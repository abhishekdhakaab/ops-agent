# Incident (Fictional): service-x latency spike after deploy

## Timeline
- 12:05Z Deploy dep_10291 (version 1.14.3) - HTTP client / retry tuning
- 12:10Z p95 latency rises ~300ms → ~980ms
- 12:12Z timeouts increase in logs

## Evidence
- Metrics: spike_detected=true, p95 elevated
- Logs: "timeout talking to upstream", "context deadline exceeded"
- Deploy: change mentions retry/backoff

## Conclusion
Likely retry storm or upstream dependency latency introduced/triggered after deploy.

## Next actions
- Reduce retries / add jitter
- Check upstream dependency health
- Consider rollback if impact high
