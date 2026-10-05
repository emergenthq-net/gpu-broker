// Cloud upstreams: each provider's breaker (closed / open / half-open), the routes, and the
// failover events, refreshed with the status tick. Hidden while `upstreams:` is off.
// Uses helpers from dash.js ($, esc, ago, get, EVENT_HOOKS, TOKEN_HOOKS).
"use strict";

const UPSTREAMS_API = "/v1/upstreams";
const UPSTREAM_PREFIX = "upstream.";
const UPSTREAM_EVENTS_SHOWN = 20;
const STATE_CLASS = { closed: "ok", open: "bad", "half-open": "" };
let upEvents = [];

function drawUpstreams(v) {
  $("upCard").hidden = !v.enabled;
  if (!v.enabled) return;
  $("upProv").innerHTML = Object.entries(v.providers).map(([name, p]) => `<tr><td><b>${esc(name)}</b></td>
    <td class="${STATE_CLASS[p.state] ?? ""}">${esc(p.state)}${p.quota ? " (quota)" : ""}</td>
    <td>${p.retry_in_s != null ? `${p.retry_in_s} s` : "–"}</td><td>${p.keys_out_of_quota || 0}</td>
    <td class=mute>${esc(p.reason || "")}</td></tr>`).join("");
  $("upRoutes").textContent = Object.entries(v.routes).map(([pat, chain]) => `${pat} → ${chain.join(" → ")}`).join("   ·   ");
}

function drawUpstreamEvents() {
  $("upEv").innerHTML = upEvents.slice().reverse().map(e => `<tr><td>${ago(e.ts)}</td>
    <td class="${e.kind === "upstream.closed" ? "ok" : "bad"}">${esc(e.kind.replace(UPSTREAM_PREFIX, ""))}</td>
    <td>${esc(e.data.provider || e.data.model || "")}${e.data.to ? " → " + esc(e.data.to) : ""}</td>
    <td class=mute>${esc(e.data.reason || "")}</td></tr>`).join("") || `<tr><td colspan=4 class=mute>none yet</td></tr>`;
}

async function loadUpstreams() {
  try { drawUpstreams(await get(UPSTREAMS_API)); } catch (e) { /* needs the main token; dash.js shows that */ }
}

EVENT_HOOKS.push(ev => {
  const mine = ev.filter(e => e.kind.startsWith(UPSTREAM_PREFIX));
  if (mine.length) { upEvents = upEvents.concat(mine).slice(-UPSTREAM_EVENTS_SHOWN); drawUpstreamEvents(); }
  loadUpstreams();
});
TOKEN_HOOKS.push(loadUpstreams);
loadUpstreams(); drawUpstreamEvents();
