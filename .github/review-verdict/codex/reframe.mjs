// The REFRAME trigger (es.PUB-3794): a PR that keeps drawing Codex findings round after round is being PATCHED, not
// DESIGNED. The rule ("reframe as soon as the second finding of the same class appears, not the eighth", fact
// feedback_use-qslice-qflow-qthink-24-plus, from hub #253's 7+ rounds) lived in a memory file and was broken again on
// hub #278 (8 rounds, 6 of one class). Prose is not a control, so verdict.mjs prints this where the next step is taken.
import { reviewBodyFindings } from './classify.mjs';
const CODEX_BOT = 'chatgpt-codex-connector[bot]';

/** Codex findings grouped by the commit they were MADE on, oldest round first: inline comments (original_commit_id)
 *  AND findings written only in a review BODY (review.commit_id, parsed by classify's reviewBodyFindings). */
export function findingRounds(inline = [], reviews = []) {
  const rounds = new Map();
  for (const c of inline) {
    if (c.user?.login !== CODEX_BOT || c.in_reply_to_id) continue;
    const sha = c.original_commit_id;
    if (!rounds.has(sha)) rounds.set(sha, { sha, at: c.created_at, paths: [] });
    rounds.get(sha).paths.push(`${c.path}:${c.original_line ?? c.line ?? '?'}`);
  }
  for (const sha of new Set(reviews.filter((rv) => rv.user?.login === CODEX_BOT).map((rv) => rv.commit_id))) {
    const found = reviewBodyFindings(reviews, sha);
    if (!found.length) continue;
    const at = reviews.find((rv) => rv.commit_id === sha && rv.user?.login === CODEX_BOT)?.submitted_at;
    if (!rounds.has(sha)) rounds.set(sha, { sha, at, paths: [] });
    for (const f of found) rounds.get(sha).paths.push(`${f.path}:${f.line ?? '?'}`);
  }
  return [...rounds.values()].sort((a, b) => String(a.at).localeCompare(String(b.at)));
}

/** The banner, or null when no reframe is due (fewer than two rounds, or the current verdict is not FINDINGS). */
export function reframeBanner(verdict, rounds) {
  if (verdict !== 'FINDINGS' || rounds.length < 2) return null;
  const files = new Map();
  for (const r of rounds) for (const p of r.paths) { const f = p.split(':')[0]; files.set(f, (files.get(f) ?? 0) + 1); }
  const hot = [...files].sort((a, b) => b[1] - a[1]).map(([f, n]) => `${f} ×${n}`).join(', ');
  return [
    `REFRAME (round ${rounds.length} of findings): stop patching. Run the qThink outer pass before the next commit:`,
    '  1. list every finding so far and name the CLASS they share (strip the specifics);',
    '  2. change the DESIGN so that class cannot occur (structured input over parsing, one shared predicate, a deleted feature);',
    '  3. record the class on the Linear issue, then fix, then re-request review.',
    `  findings by file across ${rounds.length} rounds: ${hot}`,
  ].join('\n');
}
