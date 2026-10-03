// Runs web/static/imagejob.js in a sandbox with a fake broker, feeds it event pages the way
// dash.js's status refresh does, and prints what happened, as JSON, for tests/test_dash_js.py.
"use strict";
const fs = require("fs"), vm = require("vm");
let timers = 0, gets = 0, nextId = 1, holdPost = null, holdGet = null, postState = "queued", failGets = 0;
const listeners = {}, hooks = [], tokenHooks = [], jobs = {};
const el = { textContent: "", innerHTML: "" };
const picker = { files: [{ name: "a.png", type: "image/png" }], addEventListener: (ev, fn) => { listeners[ev] = fn; }, click() {} };
const answer = id => ({ state: jobs[id], ...(jobs[id] === "done" ? { result: { outputs: [{ url: "http://c/o.mp4", file: "o.mp4" }] } } : {}) });
const sandbox = {
  document: { createElement: () => picker, body: { appendChild() {} } },
  ACTIONS: {}, EVENT_HOOKS: hooks, TOKEN_HOOKS: tokenHooks, API: { submit: "/v1/jobs", jobs: "/v1/jobs/" }, TERMINAL: ["done", "failed", "rejected"],
  $: () => el, esc: s => String(s), cls: s => s, prompt: () => "a fox", alert() {},
  post: async () => {
    const id = `j${nextId++}`; jobs[id] = postState;
    const reply = { ok: true, data: { id, resolved: "wan-i2v", job: { state: postState } } };
    if (holdPost) await holdPost;
    return reply;
  },
  get: async path => {
    gets += 1;
    if (failGets > 0) { failGets -= 1; throw new Error("token"); }   // dash.js's get(): 401 shows the prompt
    const id = path.split("/").pop(), now = answer(id); if (holdGet) await holdGet; return now;
  },
  FileReader: class { readAsDataURL() { this.result = "data:image/png;base64,AA=="; this.onload(); } },
  setTimeout: () => { timers += 1; }, setInterval: () => { timers += 1; },
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), sandbox);
const turn = () => new Promise(r => setImmediate(r));
const flush = async () => { for (let i = 0; i < 5; i++) await turn(); };   // let every pending await settle
const submit = async () => {
  sandbox.ACTIONS.image({ model: "wan-i2v", inputs: JSON.stringify({ image: "required" }) });
  const done = listeners.change(); await flush(); return done;
};
const page = async (...evs) => { hooks.forEach(h => h(evs.map(([job_id, kind]) => ({ job_id, kind })))); await flush(); };
const shows = (...parts) => parts.every(p => el.innerHTML.includes(p));
const gate = () => { let open; const p = new Promise(r => { open = r; }); return [p, open]; };
(async () => {
  const log = {};
  await submit();                                                        // j1: one fetch after the POST
  await page(["other", "job.done"], ["j1", "broker.x"], ["j1", "job.substituted"], ["j1", "job.direct"]);
  log.untouched = [shows("j1", "queued"), gets];                         // none of these is j1's state
  await page(["j1", "job.running"], ["j1", "job.switching"]);           // any order: the latest state wins
  log.running = [shows("running"), gets];
  jobs.j1 = "done"; await page(["j1", "job.done"]);
  log.done = [shows("o.mp4"), gets];
  await page(["j1", "job.failed"]);                                      // after the end: ignored
  log.after = [shows("o.mp4"), gets];
  postState = "rejected"; await submit();                                // j2: terminal in the POST reply
  await page(["j2", "job.running"]);
  log.terminalOnPost = [shows("j2", "rejected"), gets];
  postState = "queued"; let open; [holdPost, open] = gate();             // j3: ends while the POST is in flight
  const pending = submit(); await flush(); jobs.j3 = "done"; await page(["j3", "job.done"]);
  holdPost = null; open(); await pending; await flush();
  log.endedDuringPost = [shows("j3", "o.mp4"), gets];
  await submit(); [holdGet, open] = gate();                              // j4 ends, its fetch is in flight ...
  jobs.j4 = "done"; await page(["j4", "job.done"]);
  holdGet = null; await submit();                                        // ... j5 is submitted
  open(); await flush();
  log.race = [shows("j5", "queued") && !shows("o.mp4")];
  [holdGet, open] = gate(); const late = submit(); await flush();                       // j6: its POST fetch answers late
  holdGet = null; await page(["j6", "job.substituted"]);              // not a state, nothing drawn yet
  const sawNonState = shows("substituted");
  jobs.j6 = "running"; await page(["j6", "job.running"]);
  jobs.j6 = "queued"; open(); await late; await flush();                 // "queued" arrives after "running"
  log.olderAnswer = [shows("j6", "running"), sawNonState];
  // j6 ends but its fetch fails: the state shows, the job is kept, the next events retry it.
  jobs.j6 = "done"; failGets = 1; await page(["j6", "job.done"]);
  const keptAfterFailure = shows("j6", "done") && !shows("o.mp4");
  await page();                                                          // an empty page: no retry
  const g0 = gets;
  await page(["other", "job.running"]);                                  // any event: retry, outputs drawn
  log.retryOnEvent = [keptAfterFailure, gets - g0, shows("j6", "o.mp4")];
  await page(["other", "job.done"]); log.retryOnEvent.push(gets - g0);   // followed no more: no fetch
  await submit(); jobs.j7 = "failed"; failGets = 1;                       // j7: a token is saved, once
  await page(["j7", "job.failed"]);
  const g1 = gets; tokenHooks.forEach(h => h()); await flush();
  tokenHooks.forEach(h => h()); await flush();
  log.retryOnToken = [shows("j7", "failed"), gets - g1];
  await submit(); jobs.j8 = "done"; failGets = 1;                        // j8: retries never overlap
  await page(["j8", "job.done"]);
  let open8; [holdGet, open8] = gate(); const g2 = gets;
  tokenHooks.forEach(h => h()); await flush(); await page(["other", "job.queued"]);   // one in flight
  holdGet = null; open8(); await flush();
  log.oneRetryAtATime = [gets - g2, shows("j8", "o.mp4")];
  log.timers = timers;
  console.log(JSON.stringify(log));
})();
