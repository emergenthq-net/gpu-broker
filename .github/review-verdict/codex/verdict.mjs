#!/usr/bin/env node
// ONE Codex review verdict for a PR's CURRENT head, from every channel Codex uses. TEC-259.
//
//   node tools/codex/verdict.mjs <owner/repo> <pr>
//   -> prints one line, e.g. "CLEAN d463c3f (completed 2026-09-28T16:27:48Z)", plus any findings.
//   exit 0 CLEAN · 2 FINDINGS · 3 RUNNING · 4 NOT_PICKED_UP · 5 CODEX_OUT_OF_CREDITS · 1 error
//
// Why this exists: the fact `codex-reviews-every-pr-read-its-inline-comments-before-merging` said
// "read the inline comments", and on 2026-09-28 (analytiqa-prism#11) it was STILL misapplied three
// different ways, each giving a false reading:
//   1. matching review BODIES for "Reviewed commit" missed 6 inline P2s on the head ("no verdict");
//   2. watching only NEW comments missed that the "Codex Review Summary" is EDITED IN PLACE;
//   3. treating the summary row's PRESENCE as done fired while it still said "Running".
// And the obvious fix for (1), `commit_id == head`, is itself wrong: GitHub RE-ANCHORS old comments
// onto the new head when their lines are unchanged. Measured on prism#11: 11 Codex comments showed
// commit_id == the clean head d463c3f while having been made on older commits. The field that names
// the commit a comment was MADE on is `original_commit_id`; that is what this keys on.
import { execFileSync } from 'node:child_process';
import { isMain } from '../shared/is-main.mjs';
import { classify, finalVerdict, isCodexUser, latestRequest, latestTrigger } from './classify.mjs';
import { findingRounds, reframeBanner } from './reframe.mjs';

// Re-exported so existing importers (and the tests) keep one entry point.
export { classify, finalVerdict, latestRequest, latestTrigger, summaryRow } from './classify.mjs';

const EXIT = { CLEAN: 0, FINDINGS: 2, RUNNING: 3, NOT_PICKED_UP: 4, CODEX_OUT_OF_CREDITS: 5 };

// Ashwin 2026-09-29: "never get stalled on codex running out of review credits for the week." When Codex is
// out of usage it never reviews; it only answers with this comment, and the verdict would sit at
// NOT_PICKED_UP forever. Detect it and say so, so the caller falls back to an independent review
// (pr-review-toolkit:code-reviewer) instead of waiting.
//
// Two things must hold, both fail-closed:
//   1. `pushedAt` must be known. Unknown push time means clean evidence cannot be bound to a sha either
//      (see the pushedAt===null branch below) — the same reasoning applies here: with no anchor, a limit
//      reply cannot be proven to postdate the current head, so we do NOT report out-of-credits.
//   2. The limit reply must postdate BOTH the head push AND the latest review request (`latestRequest`,
//      shared with classify.mjs). A limit reply from a PREVIOUS week must not mask a review that was
//      re-triggered after it — if "@codex review" was posted after the limit reply, Codex may have
//      regained credits since, and the stale limit reply is no longer evidence of anything.
export function codexOutOfCredits(issueComments, pushedAt, triggers = null) {
  if (!pushedAt) return false;
  const comments = issueComments ?? [];
  const limit = comments.filter(c => isCodexUser(c?.user) && // exact bot identity (es.PUB-4097, Codex P2): a user named *codex* is not Codex
    /reached your Codex usage limits/i.test(c?.body ?? ''));
  const last = limit.map(c => c.created_at).sort().at(-1);
  if (!last) return false;
  const latestRequestAt = latestRequest(comments)?.created_at ?? null;
  // Timeline triggers too (draft->ready auto-triggers a review with NO comment; Codex P2): a limit reply older
  // than ANY recognised trigger is stale, not evidence about the review that trigger started.
  const triggerAts = Object.values(triggers ?? {}).filter(Boolean);
  return last >= pushedAt && (latestRequestAt === null || last >= latestRequestAt) && triggerAts.every(at => last >= at);
}

// Pure wrapper around the upgrade decision, split out of the CLI block so it's unit-testable without
// hitting the network. Only upgrades NOT_PICKED_UP/RUNNING to CODEX_OUT_OF_CREDITS, and only when the head
// hasn't moved during the read (same guard as the "head moved" branch in finalVerdict): an out-of-credits
// verdict about a stale head is as wrong as any other verdict about a stale head.
export function withOutOfCreditsUpgrade({ verdict, row, findings }, { head, headNow, issueComments, pushedAt, triggers }) {
  if (headNow === head && (verdict === 'NOT_PICKED_UP' || verdict === 'RUNNING') && codexOutOfCredits(issueComments, pushedAt, triggers)) {
    return { verdict: 'CODEX_OUT_OF_CREDITS', row, findings,
      message: 'Codex replied "reached your Codex usage limits" -> do NOT wait: run an independent review (pr-review-toolkit:code-reviewer), fix P1/P2, merge on its CLEAN.' };
  }
  return { verdict, row, findings, message: null };
}

// The activity URL for a branch. The ref is URL-ENCODED: a branch named `x&per_page=1&ref=other` must stay one
// opaque ref value and never add or override query parameters (es.PUB-4097 security fix; tested).
export function activityPath(repoFullName, branch) {
  return `repos/${repoFullName}/activity?per_page=100&ref=${encodeURIComponent('refs/heads/' + branch)}`;
}

// RULE 43 fallback label, decided HERE (not in workflow shell) so every consumer shares one predicate: the label
// must be present now AND its latest `labeled` event must postdate the SERVER-recorded push time of this head.
export function independentLabelValid({ labels = [], timeline = [], pushedAt = null }) {
  if (!pushedAt || !labels.some(l => l?.name === INDEPENDENT_LABEL)) return false;
  const at = timeline.filter(e => e?.event === 'labeled' && e.label?.name === INDEPENDENT_LABEL).map(e => e.created_at).sort().at(-1);
  return !!at && Date.parse(at) > Date.parse(pushedAt);
}
const INDEPENDENT_LABEL = 'independent-review-clean';

function gh(path) {
  // Manual page loop, NOT `gh api --paginate --slurp`: the self-hosted runners ship an older gh that rejects
  // `--slurp` ("unknown flag", es.PUB-4097). Each page is parsed as JSON on its own, so a comment BODY
  // containing "] [" can never be mangled by a textual splice (Codex review, P2, on the original version).
  // A page shorter than per_page is the last one. Cursor-paged endpoints (activity) use ghLines() instead;
  // a repeated first page here means the endpoint ignored `page`, which is an incomplete read -> throw
  // (fail closed), never "assume complete".
  const out = [];
  let firstPrev = null;
  for (let page = 1; page <= 50; page++) {
    const sep = path.includes('?') ? '&' : '?';
    const rows = JSON.parse(execFileSync('gh', ['api', `${path}${sep}per_page=100&page=${page}`], { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 }));
    if (!Array.isArray(rows)) throw new Error(`gh(${path}): expected an array page`);
    const first = JSON.stringify(rows[0] ?? null);
    if (page > 1 && first === firstPrev) throw new Error(`gh(${path}): endpoint ignores page; history would be incomplete`);
    firstPrev = first;
    out.push(...rows);
    if (rows.length < 100) return out;
  }
  throw new Error(`gh(${path}): more than 50 pages`);
}

// Cursor-paged endpoints (repos/.../activity: `before`/`after` cursors). `--paginate` follows the Link
// headers; `--jq '.[] | @json'` emits one object per line, which works on older gh (no --slurp needed).
function ghLines(path) {
  const out = execFileSync('gh', ['api', '--paginate', '--jq', '.[] | @json', path], { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
  return out.split('\n').filter(Boolean).map(l => JSON.parse(l));
}

if (isMain(import.meta.url)) {
  const [repo, pr] = process.argv.slice(2);
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo ?? '') || !/^\d+$/.test(pr ?? '')) {
    console.error('usage: verdict.mjs <owner/repo> <pr>'); process.exit(1);
  }
  try {
    const pull = JSON.parse(execFileSync('gh', ['api', `repos/${repo}/pulls/${pr}`], { encoding: 'utf8' }));
    const head = pull.head.sha;
    console.log(`HEAD_SHA=${head}`); // FIRST: the sha this verdict is about, printed before any evidence read can fail (callers post to THIS sha)
    // The server-side time this exact sha was pushed (latest, if re-pushed). Fork heads live in the fork's repo.
    // No `time_period`: measured 2026-09-28, the default is NOT one day (a ref's 4-day-old pushes returned;
    // `time_period=day` returned none). Pagination covers the rest.
    let pushedAt = null, pushedShas = [];
    try {
      const acts = ghLines(activityPath(pull.head.repo.full_name, pull.head.ref));
      pushedAt = acts.filter(a => a.after === head).map(a => a.timestamp).sort().at(-1) ?? null;
      pushedShas = acts.map(a => a.after).filter(Boolean);
    } catch { /* unreadable activity: pushedAt stays null, so CLEAN cannot be granted (fail closed) */ }
    // Order matters (Codex review, P1): read the COMPLETION evidence (summary, comments, reactions) FIRST
    // and the FINDINGS after it, so a review that finishes mid-read cannot pair "Completed" with a findings
    // list captured before its comments were posted. A CLEAN answer repeats the whole read.
    // A CLEAN answer re-reads EVERYTHING, not just the findings (Codex review, P2): a new "@codex review"
    // posted during the first read would otherwise leave CLEAN standing on a request/reaction snapshot that
    // predates it. The second read supersedes the first whatever it says.
    let lastInline = [], lastReviews = [];
    const read = () => {
      const issueComments = gh(`repos/${repo}/issues/${pr}/comments`);
      const request = latestRequest(issueComments);
      const trigger = latestTrigger(issueComments, gh(`repos/${repo}/issues/${pr}/timeline`));
      const requestReactions = request ? gh(`repos/${repo}/issues/comments/${request.id}/reactions`) : [];
      const inline = gh(`repos/${repo}/pulls/${pr}/comments`);
      // Codex also writes findings into the REVIEW BODY with no inline comment (mundonow-emdash#113).
      const reviews = gh(`repos/${repo}/pulls/${pr}/reviews`);
      lastInline = inline; lastReviews = reviews;
      return { ...classify({ head, inline, issueComments, reviews, requestReactions, triggers: trigger ?? {}, pushedAt, pushedShas }), triggers: trigger ?? {} };
    };
    let result = read();
    if (result.verdict === 'CLEAN') result = read();
    // A trigger posted after the second read fetched its comments would leave CLEAN on superseded evidence
    // (Codex review, P2): re-read ONLY the triggers, and demote CLEAN if any type's latest trigger moved.
    if (result.verdict === 'CLEAN') {
      const again = latestTrigger(gh(`repos/${repo}/issues/${pr}/comments`), gh(`repos/${repo}/issues/${pr}/timeline`));
      if (JSON.stringify(again ?? {}) !== JSON.stringify(result.triggers ?? {})) result = { ...result, verdict: 'NOT_PICKED_UP' };
    }
    // The head can move while the evidence is being read (Codex review, P2): a verdict is only about the
    // head it was computed for, so a moved head is reported as still RUNNING, never as that verdict.
    const headNow = JSON.parse(execFileSync('gh', ['api', `repos/${repo}/pulls/${pr}`], { encoding: 'utf8' })).head.sha;
    const { verdict, row, findings, message } = withOutOfCreditsUpgrade(finalVerdict(result, head, headNow),
      { head, headNow, issueComments: gh(`repos/${repo}/issues/${pr}/comments`), pushedAt, triggers: result.triggers });
    if (message) console.log(message);
    if (pushedAt === null) console.log(`NO PUSH RECORD for ${head.slice(0, 7)}: clean evidence cannot be bound to the full sha`);
    if (headNow !== head) console.log(`HEAD MOVED ${head.slice(0, 7)} -> ${headNow.slice(0, 7)} during the read; re-run`);
    if (verdict === 'CODEX_OUT_OF_CREDITS') {
      const ok = independentLabelValid({ labels: gh(`repos/${repo}/issues/${pr}/labels`), timeline: gh(`repos/${repo}/issues/${pr}/timeline`), pushedAt });
      console.log(`INDEPENDENT_REVIEW_LABEL=${ok ? 'valid' : 'invalid'}`);
    }
    console.log(`${verdict} ${head.slice(0, 7)}${row ? ` (${row.status} ${row.at ?? ''})` : ''}`);
    for (const f of findings) console.log(`  [${f.path}:${f.line ?? '?'}] #${f.id} ${(f.body ?? '').replace(/\s+/g, ' ').slice(0, 200)}`);
    const banner = reframeBanner(verdict, findingRounds(lastInline, lastReviews)); // es.PUB-3794
    if (banner) console.log(banner);
    process.exit(EXIT[verdict]);
  } catch (e) { console.error(`verdict.mjs: ${e.message}`); process.exit(1); }
}
