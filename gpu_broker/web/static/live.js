// Live metrics panel: GPU samples (VRAM stacked by owner, utilisation, power, temperature) and
// per-job latency/throughput. Canvas charts, no dependencies. Uses helpers from dash.js.
"use strict";

// ---- constants -------------------------------------------------------------
const LIVE_EVERY_MS = 2000;
const GPU_WINDOW_S = 900;            // seconds of GPU history drawn
const GPU_KEEP_SLACK_S = 10;         // keep a little more than drawn, so the left edge is filled
const JOB_WINDOW_S = 3600;           // job charts and percentiles
const PALETTE = ["#3b5bdb", "#e8590c", "#2b8a3e", "#ae3ec9", "#1098ad", "#f08c00"];  // owner groups without a configured color
const CHART = { pad: 2, line: 1.5, grid: 1, dot: 3, band: 0.25, headroom: 1.1, gridAt: [0.5, 1] };
const MIN_TPS_MAX = 100, MIN_LATENCY_MAX_S = 10;
const TOKENS_PER_K = 1000;

const grpCol = (g, i) => (UI.groups[g] || {}).color || PALETTE[i % PALETTE.length];
const grpName = g => (UI.groups[g] || {}).label || g;
const cssVar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let gpuBuf = [], gpuLast = 0;

// series: [{pts: [[t, v]], color, base?}] — `base` closes a stacked band down to the series below.
function chart(id, series, opt) {
  const c = $(id); if (!c) return;
  const dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight, p = CHART.pad;
  c.width = w * dpr; c.height = h * dpr;
  const g = c.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, w, h);
  const X = t => p + (t - opt.t0) / (opt.t1 - opt.t0) * (w - 2 * p);
  const Y = v => h - p - Math.min(v, opt.max) / opt.max * (h - 2 * p);
  g.strokeStyle = cssVar("--line"); g.lineWidth = CHART.grid;
  CHART.gridAt.forEach(f => { g.beginPath(); g.moveTo(0, Y(opt.max * f)); g.lineTo(w, Y(opt.max * f)); g.stroke(); });
  for (const s of series) {
    if (!s.pts.length) continue;
    if (opt.dots) {
      g.fillStyle = s.color;
      s.pts.forEach(([t, v]) => { g.beginPath(); g.arc(X(t), Y(v), CHART.dot, 0, 2 * Math.PI); g.fill(); });
      continue;
    }
    g.beginPath(); s.pts.forEach(([t, v], i) => i ? g.lineTo(X(t), Y(v)) : g.moveTo(X(t), Y(v)));
    g.strokeStyle = s.color; g.lineWidth = CHART.line; g.stroke();
    if (s.base) {
      for (let i = s.base.length - 1; i >= 0; i--) g.lineTo(X(s.base[i][0]), Y(s.base[i][1]));
      g.closePath(); g.globalAlpha = CHART.band; g.fillStyle = s.color; g.fill(); g.globalAlpha = 1;
    }
  }
}

function drawGpu() {
  const t1 = now(), t0 = t1 - GPU_WINDOW_S, pts = gpuBuf.filter(s => s.t >= t0), last = pts[pts.length - 1];
  if (!last) return;
  const grps = [...new Set(pts.flatMap(s => Object.keys(s.by_group)))].sort();
  let base = pts.map(s => [s.t, 0]); const ser = [];
  grps.forEach((gr, gi) => {
    const top = pts.map((s, i) => [s.t, base[i][1] + (s.by_group[gr] || 0)]);
    ser.push({ pts: top, base, color: grpCol(gr, gi) }); base = top;
  });
  ser.push({ pts: pts.map(s => [s.t, s.used_mib]), color: cssVar("--mute") });
  chart("cVram", ser, { max: last.total_mib, t0, t1 });
  chart("cUtil", [{ pts: pts.map(s => [s.t, s.util_pct]), color: cssVar("--acc") }], { max: PERCENT, t0, t1 });
  chart("cPow", [{ pts: pts.map(s => [s.t, s.power_w]), color: cssVar("--power") }], { max: UI.power_max_w, t0, t1 });
  chart("cTemp", [{ pts: pts.map(s => [s.t, s.temp_c]), color: cssVar("--temp") }], { max: UI.temp_max_c, t0, t1 });
  $("lVram").innerHTML = `${gb(last.used_mib)} / ${(last.total_mib / MIB_PER_GB).toFixed(0)} GB · ` +
    Object.entries(last.by_group).map(([gr, m]) =>
      `<span style="color:${esc(grpCol(gr, grps.indexOf(gr)))}">■</span> ${esc(grpName(gr))} ${gb(m)}`).join(" · ");
  $("lUtil").textContent = `${last.util_pct}% · ${last.sm_mhz} MHz`;
  $("lPow").textContent = `${last.power_w} W`;
  $("lTemp").textContent = `${last.temp_c} °C`;
}

function drawJobs(m) {
  const t1 = now(), t0 = t1 - JOB_WINDOW_S, js = m.jobs.filter(j => j.t >= t0), llm = js.filter(j => j.gen_tps);
  chart("cTps", [{ pts: llm.map(j => [j.t, j.gen_tps]), color: cssVar("--acc") }],
        { max: Math.max(MIN_TPS_MAX, ...llm.map(j => j.gen_tps)) * CHART.headroom, t0, t1, dots: true });
  chart("cLat", [
    { pts: js.map(j => [j.t, j.total_s]), color: cssVar("--mute") },
    { pts: js.filter(j => j.ttft_s != null).map(j => [j.t, j.ttft_s]), color: cssVar("--ttft") }],
    { max: Math.max(MIN_LATENCY_MAX_S, ...js.map(j => j.total_s)) * CHART.headroom, t0, t1, dots: true });
  const s = m.summary, f = (o, u) => o && o.p50 != null ? `${o.p50}${u} <span class="mute">p95 ${o.p95}${u}</span>` : "–";
  [["kTps", s.gen_tps, ""], ["kPp", s.prompt_tps, ""], ["kTtft", s.ttft_s, "s"], ["kTot", s.total_s, "s"],
   ["kQ", s.queue_s, "s"], ["kSw", s.switch_s, "s"]].forEach(([id, v, u]) => { $(id).innerHTML = f(v, u); });
  $("kTok").textContent = `${s.jobs} jobs · ${(s.gen_tokens_total / TOKENS_PER_K).toFixed(1)}k tokens generated`;
  const lj = llm[llm.length - 1];
  $("lTps").textContent = lj ? `last ${lj.gen_tps} tok/s · ${lj.gen_tokens} tok · ${ago(lj.t)} ago` : "no LLM jobs in the window";
}

async function liveTick() {
  try {
    const m = await get(`${API.metrics}?since=${gpuLast}&window_s=${JOB_WINDOW_S}`);
    gpuBuf.push(...m.gpu);
    if (m.gpu.length) gpuLast = m.gpu[m.gpu.length - 1].t;
    gpuBuf = gpuBuf.filter(s => s.t > now() - GPU_WINDOW_S - GPU_KEEP_SLACK_S);
    $("lErr").textContent = m.gpu_error ? "GPU stream: " + m.gpu_error : "";
    drawGpu(); drawJobs(m);
  } catch (e) { /* the token prompt is shown by get() */ }
}
liveTick(); setInterval(liveTick, LIVE_EVERY_MS);
addEventListener("resize", drawGpu);
