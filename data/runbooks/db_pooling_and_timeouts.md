# DB Pooling & Timeout Troubleshooting

## Symptoms
- `db connection pool exhausted`
- Increased request latency
- Spiky throughput with many timeouts

## Checks
- Pool size vs concurrency
- Connection leak indicators (open conns rising)
- Slow query logs / query latency
- Transaction duration / locks

## Fixes
- Reduce app concurrency temporarily
- Increase pool size (carefully) and DB max connections
- Add query indexes / optimize hot queries
- Ensure timeouts are aligned:
  - request timeout > DB timeout > upstream timeout (avoid orphaned work)

## Evidence to collect
- error counts for pool exhaustion
- p95 latency before/after changes
- top slow queries
