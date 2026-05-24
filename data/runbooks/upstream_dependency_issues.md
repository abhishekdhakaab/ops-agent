# Upstream Dependency Latency & Timeout Issues

## Symptoms
- `timeout talking to upstream`
- `context deadline exceeded`
- Latency spikes without CPU/mem saturation

## Checks
- Identify which dependency is timing out (DB/cache/external API)
- Correlate dependency latency with service latency
- Check retry/backoff configuration (avoid retry storms)

## Fixes
- Add circuit breaker / exponential backoff
- Reduce retry count; jitter backoff
- Cache or degrade non-critical calls
- Escalate to dependency owner

## Evidence to collect
- top timeout messages
- timing correlation chart (if available)
- deployment change related to client/retry logic
