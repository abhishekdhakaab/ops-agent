# Deployment Regression Checklist (Starter Runbook)

## Goal
Determine whether a recent deployment caused performance/errors.

## Steps
1. Identify **last deploy**: time, version, author, summary.
2. Compare **pre/post windows**:
   - latency avg/p95
   - error rate
   - top log signatures
3. Look for **new signatures** post-deploy:
   - new exception type, new endpoint path, new dependency call
4. Decide:
   - strong correlation + high impact → rollback
   - uncertain → canary/feature flag off, increase logging, narrow scope

## Strong-correlation heuristic
- Spike begins within 0–30 minutes of deploy
- No concurrent traffic surge
- New error signature appears post-deploy

## Rollback checklist
- Confirm rollback procedure and blast radius
- Communicate status + timeline
- After rollback: confirm metrics return to baseline
