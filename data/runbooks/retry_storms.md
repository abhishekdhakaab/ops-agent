# Retry Storms & Cascading Failures

## What happens
Aggressive retries amplify load on a slow dependency, causing a feedback loop.

## Symptoms
- timeouts increase
- latency rises
- dependency sees QPS spike
- error signatures mention retry/backoff changes

## Checks
- Look for deploy notes about HTTP client / retry logic
- Check logs for repeated attempts / identical request ids
- Compare request volume pre/post incident

## Mitigation
- Reduce retries immediately; add jitter
- Rate limit at edge
- Enable caching
- Consider temporary circuit breaker

## Evidence
- deployments mentioning retry/backoff
- logs showing repeated timeout patterns
