# Latency Spike Triage (Starter Runbook)

## When to use
- Users report slowness
- p95 latency spikes
- Timeouts increase

## Quick checks (5 minutes)
1. **Metrics**: avg/p95 latency, error rate, traffic, CPU/mem, saturation.
2. **Dependencies**: upstream latency and timeout rate (DB/cache/external API).
3. **Deployments**: last deploy time; compare 30m pre vs 30m post.
4. **Logs**: top error signatures; look for timeouts and pool exhaustion.

## Evidence patterns → likely causes
- **Latency spike + timeout errors** → upstream dependency slow/down; retry storm
- **Latency spike + DB pool exhausted** → pool sizing, slow queries, connection leaks
- **Latency spike after deploy** → regression / config change / feature flag

## Safe mitigations
- Reduce concurrency / rate limit at edge
- Increase timeouts only if dependency is healthy (avoid masking)
- Enable caching / degrade non-critical features
- Rollback if strong deploy correlation

## What to record
- time window, baseline vs spike, top error messages, last deploy id/version
