// Loads web/static/index.js (argv[2]) in a sandbox with stand-ins for what dash.js provides and
// prints, as JSON for tests/test_dash_js.py, the "who has the GPU" line for each card state.
"use strict";
const fs = require("fs"), vm = require("vm");
const sandbox = {
  UI: { gpu_label: "GPU", resident_label: "the chat model" }, API: {}, ACTIONS: {},
  get: () => Promise.reject(new Error("offline")), post: async () => ({}),
  $: () => ({}), esc: s => String(s), gb: n => n, setInterval() {}, Promise,
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), sandbox);
const models = { "qwen3-8b": { kind: "llm" }, "wan2.2-5b": { kind: "video" } };
const job = { using: "wan2.2-5b", requested: "wan" };
process.stdout.write(JSON.stringify({
  resident: sandbox.gpuLine(models, { resident_llm: "qwen3-8b", running: null }),
  lent: sandbox.gpuLine(models, { resident_llm: null, running: job }),
  lentUnknown: sandbox.gpuLine({}, { resident_llm: null, running: job }),
  free: sandbox.gpuLine(models, { resident_llm: null, running: null }),
}));
