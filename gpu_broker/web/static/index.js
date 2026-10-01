// Model index: pick a ComfyUI model → the broker frees the GPU (an interactive session job) →
// ComfyUI (or the model's own front end) opens. Uses helpers and UI labels from dash.js.
"use strict";

// ---- constants -------------------------------------------------------------
const INDEX_EVERY_MS = 3000;
const KIND_ORDER = ["ui", "llm", "image", "video", "3d"];
const DEFAULT_IDLE_MIN = 15;
const WINDOW_NAME = "comfy";
const TEMPLATE_QUERY = "/?template=";
const READY = "ready", NEEDS_INTEGRATION = "needs_integration", LLM_UNIT = "llm_unit", COMFY = "comfy";
const SAFE_URL = /^https?:\/\//;

let pending = null;   // {job, model, template, open, win} while a session waits for the GPU

function badge(m) {
  if (m.status === READY) return `<span class="ok">ready</span>`;
  if (m.status === NEEDS_INTEGRATION) return `<span class="mute">downloaded · not wired</span>`;
  return `<span class="warn">${esc(m.status)}</span>`;
}

function action(k, m, res, ses) {
  if (m.runner === LLM_UNIT) return k === res ? `<span class="ok">● resident (${esc(UI.resident_label)})</span>` : "";
  if (ses && ses.model === k)
    return `<span class="ok">● in use</span> <button data-action="open" data-template="${esc(m.comfy_template || "")}"` +
           ` data-open="${esc(m.open_url || "")}">Open</button>`;
  if (pending && pending.model === k) return `<span class="warn">switching…</span>`;
  if (m.runner === COMFY && m.status === READY)
    return `<button data-action="use" data-model="${esc(k)}">${m.open_url ? "Open " + esc(k) : "Use in ComfyUI"}</button>`;
  return "";
}

function drawIndex(models, st) {
  const res = st.resident_llm, ses = st.session;
  const rank = kind => { const i = KIND_ORDER.indexOf(kind); return i < 0 ? KIND_ORDER.length : i; };
  const rows = Object.entries(models).sort(([, a], [, b]) => rank(a.kind) - rank(b.kind) || (b.quality || 0) - (a.quality || 0));
  $("idx").innerHTML = rows.map(([k, m]) => `<tr><td>${esc(m.kind)}</td><td><b>${esc(k)}</b></td>
    <td>${m.vram_mib ? gb(m.vram_mib) + " GB" : "–"}</td><td>${badge(m)}</td><td>${action(k, m, res, ses)}</td></tr>`).join("");
  if (ses) {
    const min = Math.floor(ses.ends_in_s / S_PER_MIN), sec = ses.ends_in_s % S_PER_MIN;
    $("ses").innerHTML = `<b>${esc(ses.model)}</b> has the GPU · idle ${ses.idle_for_s}s · returns to ` +
      `${esc(res || UI.resident_label)} in <b>${min}m ${sec}s</b> of no rendering ` +
      `<button data-action="end">Done — give GPU back</button>`;
  } else $("ses").textContent = pending ? "Waiting for the GPU…" : `GPU is with ${res || "nothing"} (${UI.resident_label}). Pick a model to borrow it.`;
}

function uiUrl(template, open) {
  const url = open || UI.comfy_url + (template ? TEMPLATE_QUERY + encodeURIComponent(template) : "/");
  return SAFE_URL.test(url) ? url : null;
}

ACTIONS.open = d => { const url = uiUrl(d.template, d.open); if (url) window.open(url, WINDOW_NAME); };
ACTIONS.end = async () => { await post(API.sessionEnd); idxTick(); };
ACTIONS.use = async d => {
  const key = d.model;
  const idle = prompt(`Borrow the GPU for ${key}.\nReturn it to ${UI.resident_label} after how many idle minutes?`, DEFAULT_IDLE_MIN);
  if (idle === null) return;
  const win = window.open("about:blank", WINDOW_NAME);  // open now, inside the click, so popup blockers allow it
  const r = await post(API.sessions, { model: key, idle_min: Number(idle) || DEFAULT_IDLE_MIN });
  if (!r.ok) { if (win) win.close(); alert(r.data.detail || "failed"); return; }
  pending = { job: r.data.id, model: key, template: r.data.comfy_template, open: r.data.open_url, win };
  if (win) win.document.body.textContent = `Freeing the ${UI.gpu_label} for ${key}… (queue position ` +
                                           `${r.data.queue_position}). It opens here when it's ready.`;
  idxTick();
};

async function idxTick() {
  try {
    const [models, st] = await Promise.all([get(API.catalog), get(API.status)]);
    if (pending) {
      const j = await get(API.jobs + pending.job);
      if (st.session && st.session.job === pending.job) {
        const url = uiUrl(pending.template, pending.open);
        if (url && pending.win && !pending.win.closed) pending.win.location = url;
        else if (url) window.open(url, WINDOW_NAME);
        pending = null;
      } else if (TERMINAL.includes(j.state)) { alert(`Session ${j.state}: ${j.error || ""}`); pending = null; }
    }
    drawIndex(models, st);
  } catch (e) { /* the token prompt is shown by get() */ }
}
idxTick(); setInterval(idxTick, INDEX_EVERY_MS);
