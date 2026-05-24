# Caching & Rate Limiting Playbook

## When to use
- Traffic spikes
- Hot endpoint overload
- Dependency saturation

## Checks
- Identify top endpoints / hottest keys
- Confirm cache hit rate (if you track it)
- Check rate limiter configs

## Mitigations
- Add caching for read-heavy endpoints
- Add request coalescing for hot keys
- Enable rate limiting on non-critical endpoints
- Use graceful degradation for expensive features

## Evidence to collect
- traffic trend
- error/latency trend after enabling cache/limits
