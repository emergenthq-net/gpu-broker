"""Cloud first, local on failure: the `upstreams:` section (config.py), when a cloud answer
counts as failed (classify.py), one circuit breaker per provider (breaker.py), and the HTTP
client that talks to the provider (transport.py). The routes use them through web/failover.py."""
