// Jobs with input files from the model index: pick an image (or several views, or a video)
// for a model that takes them — image-to-video, image edit, image/views -> 3D — type a prompt,
// and the broker queues the job (POST /v1/jobs). Uses helpers from dash.js. One persistent file
// input serves every row, because the index table is redrawn while the picker is open. The job's
// progress comes from the broker's event stream that dash.js already fetches on each status
// refresh (EVENT_HOOKS): no timer of its own.
"use strict";

const IMAGE_TYPES = "image/png,image/jpeg,image/webp";   // what the broker accepts by default
const VIDEO_TYPES = "video/mp4,video/quicktime,video/webm";
const JOB_EVENT = "job.";                     // job.<state> events from the broker
// The broker's job states, in the order a job passes through them (constants.JobState). Other
// job.* events (job.substituted, job.direct) are not states.
const JOB_STATES = ["received", "queued", "switching", "running", "done", "failed", "rejected"];
const rank = state => JOB_STATES.indexOf(state);
const SAFE_LINK = /^https?:\/\//;
const EXEC = "exec";

let inputModel = null;   // {key, inputs} the picker was opened for
// {id, model, shown, pending} while a job is followed; `shown`: rank of the state drawn;
// `pending`: it ended but fetching its outputs failed (no token yet, network): retried.
let inputJob = null;

const picker = document.createElement("input");
picker.type = "file";
picker.hidden = true;
document.body.appendChild(picker);

// Called by index.js for each row: a button when the model takes input files.
function imageButton(k, m) {
  const slots = Object.keys(m.inputs || {});
  if (!slots.length || m.status !== "ready" || !(m.template || m.runner === EXEC)) return "";
  const label = slots.includes("frames") || slots.includes("video") ? "Run with views / video…" : "Run with image…";
  return ` <button data-action="image" data-model="${esc(k)}" data-inputs="${esc(JSON.stringify(m.inputs))}"` +
         ` title="takes: ${esc(slots.join(", "))}">${label}</button>`;
}

// Synchronous on purpose: a file picker opened after an await may be blocked as not user-initiated.
ACTIONS.image = d => {
  const inputs = JSON.parse(d.inputs || "{}");
  inputModel = { key: d.model, inputs };
  picker.multiple = "frames" in inputs;
  picker.accept = [IMAGE_TYPES, ...("video" in inputs ? [VIDEO_TYPES] : [])].join(",");
  picker.value = "";
  picker.click();
};

const asDataUrl = file => new Promise((ok, fail) => {
  const reader = new FileReader();
  reader.onload = () => ok(reader.result);   // a data: URL; the broker checks its type against the bytes
  reader.onerror = () => fail(reader.error);
  reader.readAsDataURL(file);
});

// Which request field the chosen files fill: a video, several views, or one image.
function fieldFor(files, inputs) {
  if (files.length === 1 && files[0].type.startsWith("video/") && "video" in inputs) return "video";
  if ("frames" in inputs) return "frames";
  return "image";
}

picker.addEventListener("change", async () => {
  const files = [...picker.files], model = inputModel;
  if (!files.length || !model) return;
  const text = model.inputs.image || model.inputs.end_image ? prompt(`Prompt for ${model.key}:`, "") : "";
  if (text === null) return;
  const field = fieldFor(files, model.inputs);
  const data = await Promise.all(files.map(asDataUrl));
  const body = { model: model.key, prompt: text, [field]: field === "frames" ? data : data[0] };
  const r = await post(API.submit, body);
  if (!r.ok) { alert(r.data.detail || "failed"); return; }
  const watching = inputJob = { id: r.data.id, model: r.data.resolved || model.key, shown: -1 };
  // Events that arrived while the POST was in flight were not ours yet: fetch the job once.
  await update(watching, { state: r.data.job.state }, true);
});

function drawInputJob(j) {
  const el = $("imgJob");
  if (!inputJob) { el.textContent = ""; return; }
  const links = ((j.result || {}).outputs || []).filter(o => SAFE_LINK.test(o.url || ""))
    .map(o => `<a href="${esc(o.url)}" target="_blank" rel="noopener noreferrer">${esc(o.file)}</a>`).join(" · ");
  const paths = ((j.result || {}).outputs || []).filter(o => !o.url).map(o => esc(o.path || o.file)).join(" · ");
  el.innerHTML = `Job <b>${esc(inputJob.id)}</b> on ${esc(inputJob.model)}: ` +
    `<span class="${cls(j.state)}">${esc(j.state)}</span>${j.error ? " · " + esc(j.error) : ""}` +
    `${links ? " · " + links : ""}${paths ? " · " + paths : ""}`;
}

// Show state changes as their events arrive; fetch the job when it ends, for its outputs. A job
// whose final fetch failed is fetched again with the next events, or once a token is saved.
EVENT_HOOKS.push(events => {
  const watching = inputJob;
  if (!watching) return;
  const states = events.filter(e => e.job_id === watching.id && e.kind.startsWith(JOB_EVENT))
    .map(e => e.kind.slice(JOB_EVENT.length)).filter(s => rank(s) >= 0);
  if (states.length) update(watching, { state: states.reduce((a, b) => (rank(b) > rank(a) ? b : a)) }, false);
  else if (events.length) retry(watching);
});
TOKEN_HOOKS.push(() => { if (inputJob) retry(inputJob); });

function retry(watching) {
  if (watching.pending && !watching.retrying) {
    watching.retrying = true;
    update(watching, { state: JOB_STATES[watching.shown] }, true).finally(() => { watching.retrying = false; });
  }
}

// Draw `j` for the watched job unless a later state is already shown; once it is terminal, fetch
// the whole job (outputs, error), and stop watching only when that fetch succeeded. `fetch`:
// fetch it now whatever the state.
async function update(watching, j, fetch) {
  let fetched = false;
  if (fetch || TERMINAL.includes(j.state)) {
    try { j = await get(API.jobs + watching.id); fetched = true; } catch (e) { /* get() shows the token prompt */ }
  }
  if (inputJob !== watching || rank(j.state) < watching.shown) return;   // replaced, or an older answer
  watching.shown = rank(j.state);
  watching.pending = TERMINAL.includes(j.state) && !fetched;
  drawInputJob(j);
  if (TERMINAL.includes(j.state) && fetched) inputJob = null;   // outputs drawn: done following it
}
