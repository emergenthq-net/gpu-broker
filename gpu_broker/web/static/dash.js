// gpu-broker dashboard: status, queue, per-model stats, events, downloads.
// The page carries no data and no secrets: it asks for the broker token once, keeps it in this
// browser's localStorage only, and sends it as a bearer header to the JSON endpoints.
"use strict";

// ---- constants -------------------------------------------------------------
const TOKEN_KEY = "gpu-broker-token";        // localStorage key
const STATUS_EVERY_MS = 5000;                // status/stats/events refresh
const EVENTS_SHOWN = 60;                     // rows in the event table (and initial backlog)
const EVENTS_PAGE = 500;
const ERROR_CHARS = 80, DETAIL_CHARS = 120;  // table cell truncation
const MIB_PER_GB = 1024;
const AGO_SECONDS_MAX = 90, AGO_MINUTES_MAX = 5400, S_PER_MIN = 60, S_PER_H = 3600, MS_PER_S = 1000;
const PERCENT = 100;
const RESIDENCY_PREFIX = "residency.";
const BAD_EVENT = /fail|error|lost/;
const DONE = "done", BAD_STATES = ["failed", "rejected"], TERMINAL = [DONE, ...BAD_STATES];
const UI_DEFAULTS = { gpu_label: "GPU", resident_label: "the default model", groups: {}, comfy_url: "",
                      power_max_w: 450, temp_max_c: 90 };
const API = { status: "/v1/status", gpu: "/v1/gpu", stats: "/v1/stats", events: "/v1/events", ui: "/v1/ui",
              metrics: "/v1/metrics", catalog: "/v1/catalog", jobs: "/v1/jobs/", sessions: "/v1/sessions",
              sessionEnd: "/v1/sessions/end" };

// ---- helpers ---------------------------------------------------------------
let tok = "";
try { tok = localStorage.getItem(TOKEN_KEY) || ""; } catch (e) { /* storage disabled */ }
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const now = () => Date.now() / MS_PER_S;
const ago = ts => {
  const s = now() - ts;
  return s < AGO_SECONDS_MAX ? Math.round(s) + "s" : s < AGO_MINUTES_MAX ? Math.round(s / S_PER_MIN) + "m" : Math.round(s / S_PER_H) + "h";
};
const cls = st => st === DONE ? "ok" : BAD_STATES.includes(st) ? "bad" : "warn";
const gb = mib => (mib / MIB_PER_GB).toFixed(1);
const authHeaders = (extra = {}) => ({ Authorization: "Bearer " + tok, ...extra });

async function get(path) {
  const r = await fetch(path, { headers: authHeaders() });
  if (r.status === 401) { $("tok").style.display = "flex"; throw new Error("token"); }
  return r.json();
}

async function post(path, body) {
  const r = await fetch(path, { method: "POST", headers: authHeaders({ "Content-Type": "application/json" }),
                                body: body === undefined ? undefined : JSON.stringify(body) });
  return { ok: r.ok, data: await r.json() };
}

// Click handling: elements carry data-action; handlers register here (no inline JS, strict CSP).
const ACTIONS = {};
document.addEventListener("click", e => {
  const el = e.target.closest("[data-action]");
  if (el && ACTIONS[el.dataset.action]) ACTIONS[el.dataset.action](el.dataset);
});

// ---- site labels -----------------------------------------------------------
let UI = UI_DEFAULTS;
async function loadUi() {
  try { UI = { ...UI_DEFAULTS, ...await get(API.ui) }; } catch (e) { return; }
  $("gpuT").textContent = UI.gpu_label !== UI_DEFAULTS.gpu_label ? `GPU (${UI.gpu_label})` : "GPU";
}

ACTIONS["save-token"] = () => {
  tok = $("t").value.trim();
  try { localStorage.setItem(TOKEN_KEY, tok); } catch (e) { /* storage disabled */ }
  $("tok").style.display = "none"; loadUi(); tick();
};

// ---- status ----------------------------------------------------------------
// A one-cell placeholder row spanning the table body `id`'s columns.
const empty = (id, text) => `<tr><td colspan=${$(id).closest("table").tHead.rows[0].cells.length} class=mute>${text}</td></tr>`;
let lastSeq = -EVENTS_SHOWN, evRows = [];

function drawStatus(st, gpu) {
  if (gpu.used_mib != null) {
    $("vram").textContent = `${gb(gpu.used_mib)} / ${(gpu.total_mib / MIB_PER_GB).toFixed(0)} GB`;
    $("vbar").style.width = (PERCENT * gpu.used_mib / gpu.total_mib).toFixed(0) + "%";
    $("util").textContent = "util " + gpu.util_pct + "%";
  } else $("util").textContent = gpu.error || "";
  $("res").textContent = st.resident_llm || "none";
  $("comfy").textContent = st.last_comfy ? "ComfyUI last ran " + st.last_comfy : "";
  const r = st.running, n = (st.inflight || []).length;
  $("run").textContent = r ? (r.resolved || r.requested) : "idle";
  $("runsub").textContent = r ? (n > 1 ? `${n} calls in parallel · ` : `${r.state} · ${ago(r.created)} · `) + r.id : "";
  $("q").innerHTML = st.queue.map((j, i) => `<tr><td>${i + 1}</td><td><code>${esc(j.id)}</code></td><td>${esc(j.requested)}</td>
    <td>${esc(j.resolved)}</td><td>${esc(j.requester)}</td><td>${ago(j.created)}</td></tr>`).join("") || empty("q", "empty");
  $("rj").innerHTML = st.recent.map(j => `<tr><td>${ago(j.created)}</td><td><code>${esc(j.id)}</code></td><td>${esc(j.requested)}</td>
    <td>${esc(j.resolved)}${j.substitution ? " ↺" : ""}</td><td class="${cls(j.state)}">${esc(j.state)}</td>
    <td>${(j.updated - j.created).toFixed(0)}</td><td class=bad>${esc((j.error || "").slice(0, ERROR_CHARS))}</td></tr>`).join("");
  $("dl").innerHTML = st.downloads.map(d => `<tr><td>${esc(d.slug)}</td><td>${esc(d.ref)}</td>
    <td class="${cls(d.state)}">${esc(d.state)}</td><td>${ago(d.updated)}</td></tr>`).join("") || empty("dl", "none");
}

function drawStats(sx) {
  let done = 0, failed = 0;
  const rows = Object.entries(sx.models).map(([m, s]) => {
    const d = s.done || {}, f = s.failed || {};
    done += d.n || 0; failed += f.n || 0;
    return `<tr><td>${esc(m)}</td><td class=ok>${d.n || 0}</td><td class="${f.n ? "bad" : "mute"}">${f.n || 0}</td>
      <td>${d.avg_s ?? "–"}</td><td>${d.max_s ?? "–"}</td></tr>`;
  });
  $("pm").innerHTML = rows.join("") || empty("pm", "no jobs");
  $("sum").innerHTML = `<span class=ok>${done}</span> ok · <span class="${failed ? "bad" : "mute"}">${failed}</span> failed`;
  $("evs").textContent = Object.entries(sx.events).map(([k, n]) => `${k.replace(RESIDENCY_PREFIX, "")} ${n}`).join(" · ");
}

function drawEvents(ev) {
  if (ev.length) { lastSeq = ev[ev.length - 1].seq; evRows = evRows.concat(ev).slice(-EVENTS_SHOWN); }
  $("ev").innerHTML = evRows.slice().reverse().map(e => `<tr><td>${ago(e.ts)}</td>
    <td class="${BAD_EVENT.test(e.kind) ? "bad" : ""}">${esc(e.kind)}</td><td><code>${esc(e.job_id || "")}</code></td>
    <td class=mute>${esc(JSON.stringify(e.data).slice(0, DETAIL_CHARS))}</td></tr>`).join("");
}

function drawSystem(sys) {\n  const s = sys.scheduler || {}, cat = sys.catalog || {}, rt = sys.runtime || {}, res = sys.resource || {};\n  const paused = Boolean(s.paused);\n  document.body.classList.toggle("paused", paused);\n  $("policy").textContent = (s.policy || "scheduler") + (paused ? " · paused" : " · active");\n  $("schedDot").className = "dot " + (paused ? "warn" : "ok");\n  $("schedBtn").dataset.paused = String(paused); $("schedBtn").textContent = paused ? "Resume" : "Quiesce";\n  $("schedSummary").textContent = paused ? "Queue paused" : ((s.queued || 0) + " queued · " + (s.inflight || 0) + " in flight");\n  $("schedDetail").textContent = (s.policy || "–") + " policy · aging " + (s.aging_s ?? "–") + "s · locality-aware";\n  $("modelCount").textContent = (cat.runnable ?? 0) + " / " + (cat.models ?? 0);\n  $("modelSummary").textContent = "runnable / catalog models";\n  $("sysDriver").textContent = rt.driver || "–"; $("sysPolicy").textContent = (s.policy || "–") + " · aging " + (s.aging_s ?? "–") + "s";\n  $("sysResource").textContent = (res.id || "gpu:0") + " · " + gb(res.vram_budget_mib || 0) + " GB budget";\n  const routes = rt.json_routes || []; $("sysRoutes").textContent = routes.length + " JSON · " + (rt.stream_routes || []).length + " streaming";\n  $("driverChip").textContent = "driver · " + (rt.driver || "–"); $("routeChip").textContent = "routes · " + routes.length;\n  const caps = Object.entries(cat.capabilities || {}).sort((a,b) => b[1] - a[1] || a[0].localeCompare(b[0]));\n  $("sysCaps").innerHTML = caps.slice(0,12).map(([cap,n]) => `<span class="chip">${esc(cap)}<strong>${n}</strong></span>`).join("") || `<span class="chip">none declared</span>`;\n}\nasync function tick() {
  try {
    const [st, gpu, sx, ev] = await Promise.all([get(API.status), get(API.gpu), get(API.stats),
                                                 get(`${API.events}?since=${lastSeq}&limit=${EVENTS_PAGE}`)]);
    drawStatus(st, gpu); drawStats(sx); drawEvents(ev);
    $("upd").textContent = "updated " + new Date().toLocaleTimeString();
  } catch (e) { $("connDot").className = "dot warn"; $("upd").textContent = e.message === "token" ? "token needed" : "error · " + e.message; }
}

if (!tok) $("tok").style.display = "flex";
loadUi(); tick(); setInterval(tick, STATUS_EVERY_MS);
