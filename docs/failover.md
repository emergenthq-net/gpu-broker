# Cloud first, local when the cloud fails

With an `upstreams:` section, gpu-broker sits between your tools and Claude or ChatGPT. A request
for `claude-…` or `gpt-…` goes to the real provider first. When the provider stops answering
(no internet, an outage, credits or quota used up) the same request is answered by a local model
instead, with no change in the tool. When the provider answers again, requests go back to it on
their own.

This is the opposite of the [hosted fallback](drop-in.md#when-the-gpu-is-busy-hosted-fallback-optional),
which sends a request to the cloud when the GPU is busy. The two can be on at once; a model name
that matches an `upstreams` route uses its route.

## Configure it

```yaml
upstreams:
  providers:
    anthropic: {url: https://api.anthropic.com, api: anthropic}
    openai:    {url: https://api.openai.com, api: openai}
  routes:                      # the first pattern that matches the requested model wins
    "claude-*": [anthropic, my-local-llm]
    "gpt-*":    [openai, my-local-llm]
    "o3*":      [openai]       # cloud only: a 503 when OpenAI is down
```

A route is one or more providers, optionally ending in one local catalog model. `gpu-broker
check` prints the routes and reports a local name the catalog does not have; `serve` refuses to
start with one. A model name no route matches works exactly as before.

`api: anthropic` providers serve `/v1/messages`; `api: openai` providers serve
`/v1/chat/completions` and `/v1/responses`. A provider of the other API is skipped.

## Credentials

**By default the broker stores no cloud key.** The client's own key is passed through to the
official provider, in the header it arrived in: `x-api-key` (Anthropic SDKs, Claude Code with an
API key) or `Authorization: Bearer` (OpenAI SDKs, Codex, Anthropic auth tokens). The broker
credential travels in its own header, `x-gpu-broker-key`, and is never sent upstream; neither is
any header that carried a broker credential:

```sh
# Claude Code: your Anthropic key as usual, the broker key beside it
export ANTHROPIC_BASE_URL=http://gpu-host:8095
export ANTHROPIC_API_KEY=sk-ant-...
export ANTHROPIC_CUSTOM_HEADERS="x-gpu-broker-key: gbk_..."
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://gpu-host:8095/v1", api_key="sk-...",
                default_headers={"x-gpu-broker-key": "gbk_..."})
```

Optionally, a provider can carry an operator key, read from the environment variable its
`key_env` names (never from the config file):

```yaml
    anthropic: {url: https://api.anthropic.com, api: anthropic, key_env: UPSTREAM_ANTHROPIC_API_KEY}
```

It is used when the client sends no key of its own, for example a tool whose only key is the
broker's. A request with no key either way skips the provider and goes local.

**`gpu-broker connect` sets this up for you.** It asks the broker (`GET /v1/upstreams/passthrough`, which a client key may read) whether
failover is on with a provider that passes the client's key through, per API. If so, Claude Code
(`--claude-code`) keeps its own login or API key and gets the broker key in
`ANTHROPIC_CUSTOM_HEADERS` (added to any headers already there), and the Codex profile keeps
sending your `OPENAI_API_KEY` (`env_key`) with the broker key in `http_headers`. Otherwise, and for
every client that gets a separate `gpu-broker` entry (Continue, Cline, Roo, Open WebUI) or reads
its settings from the environment (the shell block), the broker key stays the only key, as before.
`gpu-broker clients` shows which each client has; `disconnect` restores every file exactly.

**Which providers see a client's key.** Only those with `pass_client_key: true`, which is the
default just for `https://api.anthropic.com` and `https://api.openai.com`. Any other provider
(an OpenAI-compatible proxy such as OpenRouter, a self-hosted gateway) gets only its own `key_env`
key, or is skipped: a key a client meant for OpenAI never reaches a third party because it speaks
the same API. Set `pass_client_key: true` on a provider only if its clients really send keys for
it.

Provider keys are never logged, never stored in the database or the event log, and never shown
on the dashboard. A client's key is held only for its own request. The quota breaker below
keeps a SHA-256 fingerprint of a key that ran out (at most 1024 of them, oldest dropped), in
memory only, and forgets it once the key works again.

**API keys only.** Pass-through is built for provider API keys. Whether a consumer subscription
login (Claude Max, ChatGPT Plus) may be proxied is a question for the provider's terms; it is
untested and unsupported here.

## When a request moves to the local model

| The provider... | What happens |
|---|---|
| cannot be reached: DNS, refused, TLS, dropped connection | local |
| sends nothing before `timeouts.first_byte_s`, or goes quiet for `idle_s` before the answer is complete | local |
| answers 5xx, 408, or Anthropic's 529 `overloaded_error` (also as a stream's first event) | local |
| says the quota or credit is used up (402; error type or code `insufficient_quota` or `billing_error`) | local, and that key is left alone for a while |
| answers any other 4xx: a bad request, a bad key, a plain rate limit | returned to the client as the provider sent it |
| breaks a stream after the client has seen part of it | the stream ends with an error event; no other model is spliced in |

A bad request is not quietly sent to another model: the client sees the provider's own error.

**Unstreamed requests go upstream as streams.** The broker adds `stream: true` (and, for chat
completions, `stream_options: {include_usage: true}`; nothing else changes) and puts the stream
back together into the provider's own unstreamed answer: the same fields, usage, stop reason,
tool calls, thinking and ids. So a provider that is down is noticed after `first_byte_s`, a long
answer is never cut off by a whole-response timeout, and a stream that stops short or carries an
error fails over before the client sees anything. A provider that refuses a stream (400, 415 or
422) is asked once more unstreamed; that refusal does not count against its breaker.
A plain 429 rate limit is the client's to back off from; its `Retry-After` is passed back. Quota
is read from the error's type and code only, never its message: OpenAI's ordinary rate-limit
message links to the billing page. Two consequences:

- Any 402, whatever its body says, counts as quota and fails over.
- A 400 `invalid_request_error` whose message mentions the credit balance (as Anthropic has
  sent for an account out of credit) is a client error: it is returned as sent and does not
  fail over. Anthropic documents running out of credit as 402 `billing_error`, which does.

## The circuit breaker

Each provider has a breaker: **closed** (requests go to it), **open** (requests skip it and go
straight to the next entry, with no waiting on a timeout), **half-open** (one trial).

- It opens after `breaker.failures` failures in a row (default 3).
- While open, a background probe sends `GET /v1/models` with the provider's operator key after
  `probe_s` (15 s), doubling after each failed probe up to `probe_max_s` (5 min). Each provider is
  probed on its own, with `timeouts.probe_s` (10 s) for the whole answer. Any answer that is not
  a server error closes the breaker, and requests go back to the cloud. A provider without an
  operator key is not probed: once its probe is due, its next real request is the trial.
- A quota or credit error opens a breaker for that key alone, at once, for `quota_probe_s`
  (15 min) or as long as the provider's `Retry-After` or rate-limit reset headers say, whichever
  is longer. Its trial is the next real request with that key (a model list answers fine with an
  empty balance, so it proves nothing). Other keys keep going to the cloud.

```yaml
upstreams:
  breaker: {failures: 3, probe_s: 15, probe_max_s: 300, quota_probe_s: 900, trial_s: 120}
  timeouts: {connect_s: 5, probe_s: 10, first_byte_s: 30, response_s: 600, idle_s: 120}
```

## Seeing it

Every answer on a routed model says who served it:

- `x-gpu-broker-served-by: anthropic` (or `openai`), or `local:my-local-llm`;
- `x-gpu-broker-fallback: <why>` when the chain moved on, for example
  `anthropic 529 overloaded_error` or `anthropic circuit open (anthropic connection failed)`.

A local answer names the local model in its `model` field, so the tool can show what ran.
The dashboard's **Cloud upstreams** card shows each breaker, its next probe, keys out of quota,
and the failover events (`upstream.failover`, `upstream.open`, `upstream.quota`,
`upstream.closed`), which are also in `GET /v1/events`. Failovers are counted: one
`upstream.failover` event per provider and local model per minute, with its `count`. `GET /v1/upstreams` (main token) returns
the same view as JSON.

## Limits

- Translation between APIs is the broker's own: a Claude request that falls back is answered
  through the existing Anthropic adapter (tools included), and likewise for OpenAI and the
  Responses API. A request is never sent to a provider of the other API.
- A Responses request that continues a conversation (`previous_response_id`) goes where that
  conversation lives: a turn the broker served continues locally; any other id goes to the cloud
  chain and, if no provider can answer, returns the error rather than falling back, since the
  local model cannot continue the provider's conversation.
- `HTTP(S)_PROXY` is not honoured for upstream calls.
- Breaker state is in memory; a restart starts every provider closed.
