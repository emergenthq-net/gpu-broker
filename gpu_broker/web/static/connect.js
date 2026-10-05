// Connect apps: point AI apps at this broker, on this machine (one button each) or on another
// machine (a one-line installer with its own key). Keys can be revoked here. Main token only.
// Uses helpers from dash.js ($, esc, get, post, authHeaders, ACTIONS, TOKEN_HOOKS).
"use strict";

const CONNECT_API = { clients: "/v1/connect/clients", connect: "/v1/connect/", disconnect: "/v1/disconnect/",
                      invite: "/v1/connect/invite", keys: "/v1/keys", key: "/v1/keys/" };
const OPENWEBUI = "openwebui", CLAUDE_CODE = "claude-code";
const KEY_DATE = ts => ts ? new Date(ts * 1000).toLocaleString() : "–";

function clientAction(c) {
  if (c.connected) return `<button data-action="app-disconnect" data-name="${esc(c.name)}">Disconnect</button>`;
  if (c.found || c.name === OPENWEBUI) return `<button data-action="app-connect" data-name="${esc(c.name)}">Connect</button>`;
  return "";
}

async function loadConnect() {
  let clients, keys;
  try { [clients, keys] = await Promise.all([get(CONNECT_API.clients), get(CONNECT_API.keys)]); }
  catch (e) { $("apps").innerHTML = `<tr><td colspan=4 class=mute>needs the main broker token</td></tr>`; return; }
  $("apps").innerHTML = clients.map(c => `<tr><td><b>${esc(c.label)}</b></td>
    <td>${c.connected ? '<span class="ok">connected</span>' : c.found ? "found" : '<span class="mute">not found</span>'}</td>
    <td class="mute">${esc(c.detail)}</td><td>${clientAction(c)}</td></tr>`).join("");
  $("keys").innerHTML = keys.length ? keys.map(k => `<tr><td>${esc(k.name)}</td><td><code>${esc(k.shown)}…</code></td>
    <td>${KEY_DATE(k.created)}</td><td>${KEY_DATE(k.last_used)}</td>
    <td>${k.revoked ? '<span class="mute">revoked</span>' : `<button data-action="key-revoke" data-id="${esc(k.id)}">Revoke</button>`}</td></tr>`).join("")
    : `<tr><td colspan=5 class=mute>no client keys yet</td></tr>`;
}

function showReport(lines) { $("appsOut").textContent = (lines || []).join("\n"); }

ACTIONS["app-connect"] = async d => {
  const body = {};
  if (d.name === OPENWEBUI) {
    body.openwebui_url = prompt("Open WebUI URL (e.g. http://localhost:3000)") || "";
    body.openwebui_token = prompt("Open WebUI admin API key") || "";
    if (!body.openwebui_url || !body.openwebui_token) return;
  }
  if (d.name === CLAUDE_CODE && !confirm("Point Claude Code on this machine at the local model?")) return;
  if (d.name === CLAUDE_CODE) body.claude_code = true;
  const r = await post(CONNECT_API.connect + d.name, body);
  showReport(r.ok ? r.data.report : [r.data.detail || "failed"]); loadConnect();
};

ACTIONS["app-disconnect"] = async d => {
  const body = {};
  if (d.name === OPENWEBUI) {
    body.openwebui_url = prompt("Open WebUI URL") || "";
    body.openwebui_token = prompt("Open WebUI admin API key") || "";
  }
  const r = await post(CONNECT_API.disconnect + d.name, body);
  showReport(r.ok ? r.data.report : [r.data.detail || "failed"]); loadConnect();
};

ACTIONS["app-invite"] = async () => {
  const name = prompt("Name for the other machine (shown in the key list)", "laptop");
  if (!name) return;
  const r = await post(CONNECT_API.invite, { name });
  if (!r.ok) { showReport([r.data.detail || "failed"]); return; }
  $("invite").hidden = false;
  $("inviteCmd").textContent = r.data.command;
  $("inviteNote").textContent = `Run it on the other machine within ${Math.round(r.data.expires_in_s / 60)} minutes; it works once.`;
  loadConnect();
};

ACTIONS["key-revoke"] = async d => {
  if (!confirm("Revoke this key? Apps using it stop working.")) return;
  await fetch(CONNECT_API.key + encodeURIComponent(d.id), { method: "DELETE", headers: authHeaders() });
  loadConnect();
};

TOKEN_HOOKS.push(loadConnect);
loadConnect();
