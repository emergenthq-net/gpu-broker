// The PURE Codex verdict rules for TEC-259: raw GitHub API shapes in, one verdict out, no I/O.
// The why (three misreadings of analytiqa-prism#11, and original_commit_id vs commit_id) is in verdict.mjs,
// the CLI that reads GitHub and calls these.

// The exact GitHub App identity, not "any login containing codex": on a public repo anyone could post a
// fake clean verdict or fake findings under such a name (Codex review of this tool, P1).
const CODEX_BOT = 'chatgpt-codex-connector[bot]';
export const isCodexUser = (user) => user?.login === CODEX_BOT && (user.type === undefined || user.type === 'Bot');

/** Every summary-table row for `head` (Code Review, Security Review, ...), aggregated: 'running' if ANY
 * is still running, 'completed' only when ALL are (Codex review, P2), `at` = the latest completion.
 * The summary comment is edited in place, so its CURRENT body is the status. Rows name a 7-char sha. */
export function summaryRow(issueComments, head) {
  const short = head.slice(0, 7);
  const rows = [];
  for (const c of issueComments) {
    if (!isCodexUser(c.user) || !/codex-pull-request-review-summary/.test(c.body ?? '')) continue;
    for (const line of c.body.split('\n')) {
      if (!line.startsWith('|') || !line.includes('`' + short + '`')) continue;
      const at = (/datetime="([^"]+)"/.exec(line) ?? [])[1] ?? null;
      const type = /Security Review/.test(line) ? 'security' : 'code';
      if (/Running/.test(line)) rows.push({ status: 'running', at, type });
      else if (/Completed/.test(line)) rows.push({ status: 'completed', at, type });
    }
  }
  if (!rows.length) return null;
  // Latest completion PER review type: a re-run request is satisfied only by its own type (Codex review, P2).
  const completedAt = (t) => rows.filter(r => r.type === t && r.status === 'completed').map(r => r.at).filter(Boolean).sort().at(-1) ?? null;
  const byType = { code: completedAt('code'), security: completedAt('security') };
  if (rows.some(r => r.status === 'running')) return { status: 'running', at: null, rows: rows.length, byType };
  const at = rows.map(r => r.at).filter(Boolean).sort().at(-1) ?? null;
  return { status: 'completed', at, rows: rows.length, byType };
}

/** Findings Codex writes into a PULL REQUEST REVIEW BODY rather than as inline comments: P0-P3 badge blocks
 * separated by '---', each opening with a blob link (mundonow-emdash#113 @ 9d70236: 2 P2s, 0 inline). A
 * review's commit_id is the FULL sha fixed at submission (never re-anchored), so it binds exactly. */
export function reviewBodyFindings(reviews, head) {
  return reviews.filter(rv => isCodexUser(rv.user) && rv.commit_id === head).flatMap(rv =>
    (rv.body ?? '').split(/\n---\n/).filter(part => /!\[P[0-3] Badge\]/.test(part)).map(part => {
      const m = /\/blob\/[0-9a-f]{40}\/(\S+?)#L(\d+)/.exec(part);
      return { id: rv.id, path: m ? decodeURIComponent(m[1]) : '(review body)', line: m ? Number(m[2]) : null, body: part.trim() };
    }));
}

const reviewedCommit = (body) => (/Reviewed commit:\*\* `([0-9a-f]{7,40})`/.exec(body ?? '') ?? [])[1];

/** Pure verdict. Inputs are raw GitHub API shapes (see fixtures/). `requestReactions` are reaction
 * objects ({content, user}); `requestAt` is the latest "@codex review" comment's created_at, if any.
 * `pushedAt` is GitHub's SERVER time for the push whose `after` is the FULL head sha (repo activity API).
 * Codex's clean evidence names only a 7/10-char prefix, which is grindable: a new commit sharing the old
 * head's prefix would inherit its stale "Completed" row (Codex review, P1). Evidence therefore counts only
 * if it is dated at or after that push; with no push record the head cannot be bound and is never CLEAN. */
// Timing alone cannot say WHICH full sha a run reviewed: a colliding commit pushed while the old review is
// still running is followed by that review's completion (Codex review, P1, round 2). But the attack needs two
// DISTINCT full shas sharing the evidence's prefix on this branch, and the activity history lists every
// push's full `after`. `pushedShas` is that list; any collision means the prefix evidence is ambiguous.
export function classify({ head, inline, issueComments, reviews = [], requestReactions = [], requestAt = null, requestType = 'code', triggers = null, pushedAt = null, pushedShas = [] }) {
  const findings = [...inline.filter(c => isCodexUser(c.user) && c.in_reply_to_id == null && c.original_commit_id === head),
    ...reviewBodyFindings(reviews, head)];
  const row = summaryRow(issueComments, head);
  const cleanComments = [...issueComments.filter(c => isCodexUser(c.user) && /Didn.t find any major issues/.test(c.body ?? '')
    && reviewedCommit(c.body) !== undefined && head.startsWith(reviewedCommit(c.body))),
    // A clean REVIEW BODY on the exact full sha is the same evidence, better bound (its commit_id is full).
    ...reviews.filter(rv => isCodexUser(rv.user) && rv.commit_id === head && /Didn.t find any major issues/.test(rv.body ?? ''))
      .map(rv => ({ ...rv, created_at: rv.submitted_at }))];
  // Only the Codex app's own reactions are evidence: anyone can add 👍 on a public PR (Codex review, P1).
  // A reaction is NOT bound to any commit, so it can never prove a head clean: five review rounds of rules
  // trying to bind a 👍 to a push (commit date, check suites, force-push back to an old sha) each had a
  // hole. CLEAN therefore needs evidence that NAMES the sha (a Completed summary row for it, or a "no major
  // issues" comment naming it). 👀 may still mean RUNNING: that can only delay a merge, never grant one.
  const eyes = requestReactions.some(r => isCodexUser(r.user) && r.content === 'eyes');
  // The latest clean/completed evidence for THIS head. A newer request that Codex has not acknowledged yet
  // (no 👀/👍) means that evidence is superseded: the re-run is pending, not clean (Codex review, P2).
  const evidenceAt = [row?.status === 'completed' ? row.at : null, ...cleanComments.map(c => c.created_at)].filter(Boolean).sort().at(-1) ?? null;
  // A request is answered only by evidence of the SAME review type: a security row completing after an
  // "@codex review" must not satisfy it (Codex review, P2). The "no major issues" comment is a code review.
  const typeEvidenceAt = (t) => [row?.byType?.[t] ?? null, ...(t === 'code' ? cleanComments.map(c => c.created_at) : [])]
    .filter(Boolean).sort().at(-1) ?? null;
  // GitHub's created_at is whole seconds while Codex rows carry microseconds, so a request later in the SAME
  // second as a completion would sort before it. Compare whole seconds; a tie is pending (Codex review, P2).
  const sec = (t) => Math.floor(Date.parse(t) / 1000);
  // PER TYPE: each review type's latest trigger must be answered by a later completion of that type. A later
  // security request must not hide an unanswered earlier code request (Codex review, P2). `triggers` is
  // {code?: at, security?: at} from latestTrigger(); requestAt/requestType is the single-trigger shorthand.
  const pending = triggers ?? (requestAt !== null ? Object.fromEntries([requestType].flat().map(t => [t, requestAt])) : {});
  // A trigger posted before the current head was pushed was answered (or not) on an OLDER head: it is not
  // owed by this one, or it would pin every later head at NOT_PICKED_UP (Codex review, P2). Same second = kept.
  const rerunPending = Object.entries(pending).filter(([, at]) => pushedAt === null || sec(at) >= sec(pushedAt)).some(([t, at]) => {
    const done = typeEvidenceAt(t);
    return done === null || sec(at) >= sec(done);
  });
  // Precedence: findings made ON this sha stand even during a re-run (a fix would change the sha); an
  // in-progress re-run (Running row, or 👀, which Codex clears when done) outranks OLDER clean evidence
  // for the same sha, because a real PR keeps its earlier "no major issues" comment.
  if (findings.length) return { verdict: 'FINDINGS', row, findings };
  if (row?.status === 'running' || eyes) return { verdict: 'RUNNING', row, findings };
  if (rerunPending) return { verdict: 'NOT_PICKED_UP', row, findings };
  const collision = pushedShas.some(s => s !== head && s.slice(0, 7) === head.slice(0, 7));
  const bound = !collision && pushedAt !== null && evidenceAt !== null && Date.parse(evidenceAt) >= Date.parse(pushedAt);
  if ((row?.status === 'completed' || cleanComments.length) && bound) return { verdict: 'CLEAN', row, findings };
  return { verdict: 'NOT_PICKED_UP', row, findings };
}

/** The latest Codex trigger: "@codex review" OR "@codex security review" (the summary advertises both;
 * missing the second let an older clean row answer for a security re-run, Codex review P2). */
// A trigger counts only from someone who can actually direct the bot on this repo: anyone can POST the
// words on a public PR (Codex review, P2). GitHub's own author_association is the authority.
const TRUSTED = new Set(['OWNER', 'MEMBER', 'COLLABORATOR']);
const isRequest = (c) => TRUSTED.has(c.author_association) && /^@codex (security )?review\b/i.test((c.body ?? '').trim());
export function latestRequest(issueComments) {
  return [...issueComments].reverse().find(isRequest) ?? null;
}

/** The latest trigger PER REVIEW TYPE: every trusted "@codex review" / "@codex security review" comment,
 * plus draft->ready transitions, which auto-trigger a code review with NO comment (Codex review, P2).
 * Returns {code?: at, security?: at} or null. Keeping each type separately makes the order between types
 * irrelevant (a later security request no longer hides an earlier code one), and same-second ties need no
 * special case: each type is answered on its own. `timeline` is the issue timeline API shape. */
export function latestTrigger(issueComments = [], timeline = []) {
  const out = {};
  const add = (type, at) => { if (at && (!out[type] || Date.parse(at) > Date.parse(out[type]))) out[type] = at; };
  for (const e of timeline) if (e.event === 'ready_for_review') add('code', e.created_at);
  for (const c of issueComments.filter(isRequest)) add(/^@codex security review\b/i.test((c.body ?? '').trim()) ? 'security' : 'code', c.created_at);
  return Object.keys(out).length ? out : null;
}

/** A verdict is only about the head it was computed for: if the PR head moved during the read, it is
 * reported as RUNNING (re-run), never as that head's CLEAN/FINDINGS (Codex review, P2). */
export function finalVerdict(result, head, headNow) {
  return headNow === head ? result : { ...result, verdict: 'RUNNING' };
}
