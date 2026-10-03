// Runs web/static/dash.js in a sandbox opened at the URL given as argv[3], with the broker
// answering every call with status argv[4] (default 401) and argv[5] (if given) already saved as
// the token, and prints, as JSON for tests/test_dash_js.py, what it did with a `#token=` fragment:
// the token it sent, what it stored, and the address it left in the bar.
"use strict";
const fs = require("fs"), vm = require("vm");
const url = new URL(process.argv[3]);
const status = Number(process.argv[4] || 401), store = process.argv[5] ? { "gpu-broker-token": process.argv[5] } : {}, sent = [];
let replaced = null;
const el = () => ({ style: {}, textContent: "", innerHTML: "", hidden: false, closest: () => ({ tHead: { rows: [{ cells: [] }] } }) });
const sandbox = {
  location: { hash: url.hash, pathname: url.pathname, search: url.search },
  history: { replaceState: (s, t, u) => { replaced = u; } },
  localStorage: { getItem: k => store[k] ?? null, setItem: (k, v) => { store[k] = v; } },
  document: { getElementById: el, addEventListener() {}, documentElement: {} },
  fetch: async (path, opt) => { sent.push(opt.headers.Authorization); return { status, ok: status >= 200 && status < 300, json: async () => ({}) }; },
  setInterval() {}, Date, Promise, URL,
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), sandbox);
setImmediate(() => process.stdout.write(JSON.stringify({ sent: sent[0], stored: store["gpu-broker-token"] ?? null, replaced })));
