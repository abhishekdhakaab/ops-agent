# Error Burst Investigation (Starter Runbook)

## When to use
- 5xx spike
- Error logs surge
- Sudden SLO burn

## Quick checks
1. **Logs**: error count, top messages, new error signature?
2. **Metrics**: error rate, latency (errors often raise latency too), traffic spikes
3. **Deployments**: did errors start right after deploy?
4. **Dependency health**: DB/cache/external API status

## Common signatures
- `timeout talking to upstream` → dependency issues, network, misconfigured timeouts
- `db connection pool exhausted` → pool starvation, slow queries, connection leak
- `context deadline exceeded` → latency/timeout mis-match, slow dependency
- `permission denied` / `unauthorized` → auth/role misconfig after deploy
- `schema validation failed` → incompatible contract, bad payloads

## Actions
- If correlated with deploy: rollback / disable feature flag
- If dependency down: circuit-break + degrade + escalate
- If traffic spike: rate limit + cache + autoscale (if available)
