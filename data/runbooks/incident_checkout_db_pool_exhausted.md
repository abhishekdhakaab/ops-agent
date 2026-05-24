# Incident (Fictional): checkout errors from DB pool exhaustion

## Timeline
- 09:20Z traffic increased (promo)
- 09:25Z error burst begins
- 09:27Z logs show db pool exhausted

## Evidence
- Logs: "db connection pool exhausted"
- Metrics: latency increased with errors
- Deployments: no recent deploy

## Conclusion
Resource saturation; pool too small for concurrency or slow queries/locks.

## Next actions
- Reduce concurrency + increase pool cautiously
- Investigate slow queries and lock contention
