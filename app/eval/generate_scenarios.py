"""
Procedural Scenario Generator for the Ops Agent.

Generates diverse operational incident scenarios with:
- Randomized timing, severity, error messages
- Red herrings (e.g., recent deploy that ISN'T the cause)
- Confounding signals (e.g., spike + deploy but actually upstream issue)
- Automatic ground truth (generator knows the answer)
- Linked tool outputs (metrics/logs/deploys consistent with root cause)

Usage:
    python generate_scenarios.py              # generate 100 scenarios
    python generate_scenarios.py --count 200  # generate 200
    python generate_scenarios.py --seed 42    # reproducible
    python generate_scenarios.py --hard       # harder scenarios for training
"""

import json
import random
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "data" / "scenarios.json"


# Each root cause has a template for what metrics/logs/deploys look
# like.  The generator picks a root cause, then fills in the template

SERVICE_NAMES = [
    "order-svc", "checkout-svc", "payment-svc", "user-auth", "search-svc",
    "notification-svc", "inventory-svc", "cdn-edge", "email-worker",
    "recommendation-svc", "session-store", "image-resize", "analytics-ingest",
    "auth-proxy", "billing-svc", "file-upload", "pricing-engine",
    "shipping-calc", "feed-svc", "catalog-svc", "cart-svc", "review-svc",
    "audit-log", "rate-limiter", "config-svc", "webhook-svc", "export-svc",
    "import-worker", "cache-proxy", "api-gateway", "media-svc", "pdf-generator",
    "scheduler-svc", "queue-worker", "metrics-collector", "health-checker",
]

DEPLOY_SUMMARIES = {
    "deploy_regression": [
        "Increase retry count from 3 to {n} and reduce backoff interval",
        "Refactor request pipeline to async/await pattern",
        "Upgrade {lib} from v{v1} to v{v2}",
        "Rebuild search index with new schema v{v2}",
        "Deploy ML model v{v2} (larger embeddings, {n}GB)",
        "Add concurrent batch processing with {n} workers",
        "Migrate ORM from v{v1} to v{v2} (eager loading changes)",
        "Switch to new HTTP client library with different timeout defaults",
        "Enable gzip compression on all responses",
        "Replace connection pool implementation",
    ],
    "config_change": [
        "Flush CDN cache and update cache-control headers",
        "Enable dynamic-pricing-v{v2} feature flag",
        "Rotate TLS certificates for {domain}",
        "Update rate limit thresholds from {n} to {n2}/min",
        "Change log level from INFO to DEBUG",
        "Enable new A/B test variant for checkout flow",
    ],
    "neutral": [
        "Update UI copy and translations",
        "Routine dependency update (security patches)",
        "Add new logging fields for audit trail",
        "Update email templates for Q{q} campaign",
        "Bump CI/CD pipeline version",
        "Add health check endpoint",
    ],
}

ERROR_TEMPLATES = {
    "deploy_regression": {
        "retry_storm": [
            "timeout talking to upstream ({n}ms)",
            "retry budget exhausted: {n}/{m} retries failed",
            "context deadline exceeded after {n}s",
        ],
        "oom": [
            "OOM killed worker pid {pid}: used {n}GB of {m}GB",
            "heap usage {pct}%: GC pause {n}ms",
            "cannot allocate {n}MB for request buffer",
        ],
        "query_regression": [
            "slow query: {query} ({n}ms, expected <{m}ms)",
            "N+1 detected: {n} individual SELECTs per request",
            "db pool utilization {pct}%",
        ],
        "serialization": [
            "JSON parse error: unexpected token at position {n}",
            "schema validation failed: missing field '{field}'",
            "response size {n}MB exceeds limit {m}MB",
        ],
    },
    "upstream_dependency": [
        "connection refused from {dep}",
        "circuit breaker OPEN for {dep} (failures: {n}/{m})",
        "gateway timeout from {dep} (>{n}s)",
        "DNS resolution failed for {dep}.internal",
    ],
    "resource_exhaustion": {
        "db_pool": [
            "db pool exhausted: 0/{n} connections available",
            "waiting for connection timeout after {n}s",
            "{action} failed: db unavailable",
        ],
        "memory": [
            "heap usage {pct}%: approaching OOM",
            "GC pause {n}ms (stop-the-world)",
            "OOM killed worker pid {pid}",
        ],
        "cpu": [
            "CPU throttled: {pct}% utilization",
            "request processing timeout ({n}s)",
            "queue depth {n}: processing backlog",
        ],
        "disk": [
            "ENOSPC: no space left on device",
            "disk usage {pct}% on {vol} volume",
            "write failed: cannot write to {path}",
        ],
    },
    "infrastructure": [
        "CLUSTERDOWN: {component} cluster is down",
        "MOVED {n} {ip}:{port}",
        "consumer lag {n} messages on partition {p}",
        "rebalance triggered: {component} consumer left group",
        "connection reset by peer ({component})",
    ],
    "healthy": [
        "minor: GC pause {n}ms",
        "rate limit applied for client {client}",
        "slow query warning: {n}ms (threshold {m}ms)",
        "minor: email delivery delayed {n}s for batch {batch}",
    ],
}

DEPENDENCIES = [
    "payments-svc", "stripe-api", "postgres-primary", "redis-cluster",
    "elasticsearch", "rabbitmq", "kafka-broker", "s3-storage",
    "auth-provider", "notification-gateway", "cdn-origin",
]



def _fill(template: str, rng: random.Random) -> str:
    """Fill placeholders in a template string."""
    return template.format(
        n=rng.randint(3, 500),
        n2=rng.randint(100, 5000),
        m=rng.randint(3, 50),
        v1=rng.randint(1, 5),
        v2=rng.randint(6, 12),
        pid=rng.randint(1000, 9999),
        pct=rng.randint(85, 99),
        lib=rng.choice(["axios", "httpx", "grpc-client", "pg-driver", "redis-client"]),
        domain=rng.choice(["api.example.com", "cdn.example.com", "auth.example.com"]),
        dep=rng.choice(DEPENDENCIES),
        component=rng.choice(["redis", "kafka", "consul", "etcd"]),
        ip=f"10.0.{rng.randint(1,10)}.{rng.randint(1,200)}",
        port=rng.choice([6379, 9092, 5432, 8500]),
        p=rng.randint(0, 11),
        field=rng.choice(["user_id", "timestamp", "amount", "currency", "status"]),
        query=rng.choice(["SELECT * FROM orders WHERE user_id = ?", "UPDATE inventory SET stock = ?", "INSERT INTO audit_log"]),
        action=rng.choice(["login", "checkout", "payment", "search", "upload"]),
        vol=rng.choice(["/data", "/tmp", "/var/log"]),
        path=rng.choice(["/data/uploads", "/var/log/app", "/tmp/cache"]),
        client=f"{rng.choice(['abc','xyz','acme','beta'])}-corp",
        batch=rng.randint(1000, 9999),
        q=rng.randint(1, 4),
    )


def _pick_errors(rng: random.Random, templates: list, count: int = 3) -> List[str]:
    chosen = rng.sample(templates, min(count, len(templates)))
    return [_fill(t, rng) for t in chosen]


def generate_scenario(
    rng: random.Random,
    service: str,
    root_cause: str,
    difficulty: str = "medium",
) -> Dict[str, Any]:
    """Generate a single scenario with consistent tool data and ground truth."""

    sc_id = f"sc_{service}_{rng.randint(1000, 9999)}"

    if root_cause == "deploy_regression":
        subtype = rng.choice(["retry_storm", "oom", "query_regression", "serialization"])
        spike_start = rng.randint(10, 90)
        deploy_ago = spike_start + rng.randint(-5, 8)  # deploy slightly before spike
        has_spike = True
        error_count = rng.randint(40, 350)
        errors = _pick_errors(rng, ERROR_TEMPLATES["deploy_regression"][subtype])
        deploy_summary = _fill(rng.choice(DEPLOY_SUMMARIES["deploy_regression"]), rng)
        deploy_author = rng.choice(["alex", "sam", "morgan", "taylor", "jordan"])
        severity = rng.choice(["high", "high", "medium"])
        correct_action = "rollback"
        must_mention = ["deploy"]
        should_mention = [deploy_author, str(deploy_ago)]
        should_not_conclude = ["healthy", "stable", "no issues"]

        # Red herring for hard mode: add unrelated upstream warning in logs
        if difficulty == "hard":
            errors.append(_fill("minor: {dep} response slow ({n}ms) [not causal]", rng))

    elif root_cause == "upstream_dependency":
        dep = rng.choice(DEPENDENCIES)
        spike_start = rng.randint(5, 45)
        has_spike = True
        error_count = rng.randint(50, 400)
        errors = _pick_errors(rng, ERROR_TEMPLATES["upstream_dependency"])
        errors = [e.replace(rng.choice(DEPENDENCIES), dep) if DEPENDENCIES[0] not in e else e for e in errors]
        severity = rng.choice(["high", "high", "medium"])
        correct_action = "escalate"
        must_mention = ["error"]
        should_mention = [dep.split("-")[0], "upstream", "connection"]
        should_not_conclude = ["deploy caused", "rollback"]

        if difficulty in ("medium", "hard"):
            deploy_ago = rng.randint(15, 60)  # looks suspicious but isn't the cause
            deploy_summary = _fill(rng.choice(DEPLOY_SUMMARIES["neutral"]), rng)
            deploy_author = "ci-bot"
        else:
            deploy_ago = rng.randint(360, 2000)
            deploy_summary = _fill(rng.choice(DEPLOY_SUMMARIES["neutral"]), rng)
            deploy_author = "ci-bot"

    elif root_cause == "resource_exhaustion":
        subtype = rng.choice(["db_pool", "memory", "cpu", "disk"])
        spike_start = rng.randint(8, 120)
        has_spike = subtype != "disk"  # disk full = fast failures, no latency spike
        error_count = rng.randint(30, 350)
        errors = _pick_errors(rng, ERROR_TEMPLATES["resource_exhaustion"][subtype])
        deploy_ago = rng.randint(200, 3000)
        deploy_summary = _fill(rng.choice(DEPLOY_SUMMARIES["neutral"]), rng)
        deploy_author = "ci-bot"
        severity = rng.choice(["high", "medium"])
        correct_action = "mitigate"
        must_mention = [subtype.replace("_", " ") if "_" in subtype else subtype.upper()]
        should_mention = ["exhausted" if subtype == "db_pool" else subtype, "no deploy"]
        should_not_conclude = ["deploy caused", "healthy"]

        # RED HERRING for hard mode: ancient deploy that added caching (memory leak source)
        if difficulty == "hard" and subtype == "memory":
            deploy_ago = rng.randint(180, 480)
            deploy_summary = "Add in-memory cache for frequently accessed data"
            deploy_author = rng.choice(["alex", "sam"])
            should_mention.append("cache")

    elif root_cause == "config_change":
        spike_start = rng.randint(5, 30) if rng.random() > 0.4 else None
        has_spike = spike_start is not None
        error_count = rng.randint(10, 200)
        errors = [_fill("feature flag {field}-v{v2} causing unexpected behavior", rng),
                  _fill("config value out of range: {n}", rng),
                  _fill("TLS handshake failed: certificate expired", rng)]
        errors = rng.sample(errors, min(3, len(errors)))
        deploy_ago = rng.randint(10, 180)
        deploy_summary = _fill(rng.choice(DEPLOY_SUMMARIES["config_change"]), rng)
        deploy_author = rng.choice(["product-team", "security-bot", "sam"])
        severity = rng.choice(["high", "medium", "medium"])
        correct_action = rng.choice(["rollback", "mitigate"])
        must_mention = ["config"]
        should_mention = ["change", "flag" if "flag" in deploy_summary.lower() else "certificate"]
        should_not_conclude = ["healthy", "stable"]

    elif root_cause == "infrastructure":
        spike_start = rng.randint(3, 30)
        has_spike = True
        error_count = rng.randint(80, 500)
        errors = _pick_errors(rng, ERROR_TEMPLATES["infrastructure"])
        deploy_ago = rng.randint(1000, 5000)
        deploy_summary = _fill(rng.choice(DEPLOY_SUMMARIES["neutral"]), rng)
        deploy_author = "ci-bot"
        severity = "high"
        correct_action = "escalate"
        must_mention = ["cluster" if "cluster" in " ".join(errors).lower() else "infrastructure"]
        should_mention = ["no deploy", "down", "connection"]
        should_not_conclude = ["deploy caused", "healthy"]

    elif root_cause == "healthy":
        spike_start = None
        has_spike = False
        error_count = rng.randint(0, 8)
        errors = _pick_errors(rng, ERROR_TEMPLATES["healthy"], count=rng.randint(1, 2))
        deploy_ago = rng.randint(500, 5000)
        deploy_summary = _fill(rng.choice(DEPLOY_SUMMARIES["neutral"]), rng)
        deploy_author = "ci-bot"
        severity = "low"
        correct_action = "investigate"
        must_mention = []
        should_mention = ["healthy", "stable", "normal", "no spike"]
        should_not_conclude = ["critical", "outage", "regression", "spike"]

        # RED HERRING for hard mode: recent deploy that DID NOT cause issues
        if difficulty == "hard":
            deploy_ago = rng.randint(15, 60)
            deploy_summary = _fill(rng.choice(DEPLOY_SUMMARIES["neutral"]), rng)
            deploy_author = rng.choice(["alex", "sam"])
    else:
        raise ValueError(f"Unknown root cause: {root_cause}")

    if has_spike:
        avg_lat = round(rng.uniform(200, 900), 1)
        p95_lat = round(avg_lat * rng.uniform(1.8, 3.5), 1)
    else:
        avg_lat = round(rng.uniform(20, 120), 1)
        p95_lat = round(avg_lat * rng.uniform(1.5, 2.8), 1)

    has_recent_deploy = deploy_ago < 120

    scenario = {
        "id": sc_id,
        "service": service,
        "root_cause": root_cause,
        "severity": severity,
        "difficulty": difficulty,
        "metrics": {
            "has_latency_spike": has_spike,
            "avg_latency_ms": avg_lat,
            "p95_latency_ms": p95_lat,
            "spike_start_minutes_ago": spike_start,
        },
        "logs": {
            "error_count": error_count,
            "error_messages": errors,
            "log_level": "ERROR" if error_count > 20 else "WARN",
        },
        "deployments": {
            "has_recent_deploy": has_recent_deploy,
            "deploy_minutes_ago": deploy_ago,
            "deploy_summary": deploy_summary,
            "deploy_author": deploy_author,
        },
        "ground_truth": {
            "root_cause": root_cause,
            "severity": severity,
            "correct_action": correct_action,
            "must_mention": must_mention,
            "should_mention": should_mention,
            "should_not_conclude": should_not_conclude,
        },
    }

    return scenario


def generate_dataset(
    count: int = 100,
    seed: int = 42,
    hard_ratio: float = 0.3,
) -> List[Dict[str, Any]]:
    """Generate a full dataset of diverse scenarios."""
    rng = random.Random(seed)

    # Root cause distribution (weighted to match real-world frequency)
    root_causes = (
        ["deploy_regression"] * 25 +
        ["resource_exhaustion"] * 20 +
        ["upstream_dependency"] * 15 +
        ["config_change"] * 12 +
        ["infrastructure"] * 10 +
        ["healthy"] * 18
    )

    scenarios = []
    used_services = set()

    for i in range(count):
        available = [s for s in SERVICE_NAMES if s not in used_services]
        if not available:
            used_services.clear()
            available = SERVICE_NAMES
        service = rng.choice(available)
        used_services.add(service)

        root_cause = rng.choice(root_causes)
        difficulty = "hard" if rng.random() < hard_ratio else rng.choice(["easy", "medium", "medium"])

        sc = generate_scenario(rng, service, root_cause, difficulty)
        scenarios.append(sc)

    return scenarios



QUESTION_TEMPLATES = {
    "deploy_regression": [
        "Why is {service} slow right now?",
        "Did the latest deployment cause issues in {service}?",
        "{service} started having problems recently. What happened?",
        "Investigate {service} — users are reporting errors after a release.",
        "Is there a deployment regression in {service}?",
    ],
    "upstream_dependency": [
        "We are seeing errors in {service}. What is causing this?",
        "{service} is throwing connection errors. Investigate.",
        "Is {service} down? Users can't complete their actions.",
        "Error spike in {service} — what's going on?",
    ],
    "resource_exhaustion": [
        "{service} is running slow. Can you investigate?",
        "What's happening with {service}? It seems degraded.",
        "{service} is timing out for some users. What should we check?",
        "Performance issue in {service} — help me triage.",
    ],
    "config_change": [
        "Something changed in {service}. What's going on?",
        "{service} is behaving differently. Investigate.",
        "Are there any issues with {service} right now?",
        "Check if {service} is working correctly.",
    ],
    "infrastructure": [
        "{service} has errors but no recent deploys. What's happening?",
        "Investigate {service} — seems like a platform issue.",
        "{service} is unstable. Is it infrastructure?",
    ],
    "healthy": [
        "Is {service} having any issues right now?",
        "Quick health check on {service} please.",
        "Any problems with {service}?",
        "Status update on {service}?",
    ],
}


def expected_tools_for(scenario: dict) -> Tuple[list, list]:
    root = scenario["root_cause"]
    has_deploy = scenario["deployments"]["has_recent_deploy"]
    if root == "deploy_regression":
        return (["deployments"], ["metrics", "logs", "deployments"])
    elif root in ("upstream_dependency", "resource_exhaustion"):
        return (["logs"], ["logs", "metrics"])
    elif root == "config_change":
        return (["logs"], ["logs", "deployments"])
    elif root == "infrastructure":
        return (["logs"], ["logs", "metrics"])
    elif root == "healthy":
        return ([], ["metrics", "logs"])
    return ([], ["metrics", "logs"])


def tool_count_range(scenario: dict) -> Tuple[int, int]:
    root = scenario["root_cause"]
    if root == "healthy":
        return (1, 3)
    elif root == "deploy_regression":
        return (2, 5)
    return (1, 4)


def generate_questions(scenarios: List[Dict], rng: random.Random) -> list:
    questions = []
    for sc in scenarios:
        service = sc["service"]
        root = sc["root_cause"]
        gt = sc["ground_truth"]
        templates = QUESTION_TEMPLATES.get(root, QUESTION_TEMPLATES["healthy"])
        template = rng.choice(templates)
        question = template.format(service=service)

        exp_all, exp_any = expected_tools_for(sc)
        min_t, max_t = tool_count_range(sc)

        questions.append({
            "id": sc["id"],
            "question": question,
            "service": service,
            "scenario_id": sc["id"],
            "category": root,
            "difficulty": sc.get("difficulty", "medium"),
            "expected_tools_all": exp_all,
            "expected_tools_any": exp_any,
            "expected_min_tools": min_t,
            "expected_max_tools": max_t,
            "ground_truth": gt,
        })
    return questions



PLANNER_SYSTEM = (
    "You are an ops investigation planner. Choose the NEXT best tool action.\n"
    "Return ONLY valid JSON: {action, args, rationale}.\n"
    "Allowed actions: metrics, logs, deployments, final.\n"
    "Never invent evidence. Be evidence-driven."
)

def first_tool_for(sc: dict) -> str:
    """Pick the single best first tool for this scenario.
    
    Balanced distribution (~32/38/30 across deployments/logs/metrics)
    so that 'always pick one tool' maxes out at ~38% accuracy.
    """
    root = sc["root_cause"]
    if root == "deploy_regression":
        return "deployments"   # Check what was deployed recently
    elif root == "config_change":
        return "deployments"   # Config changes show as deployment events
    elif root == "upstream_dependency":
        return "logs"          # Upstream errors appear in error logs
    elif root == "resource_exhaustion":
        return "metrics"       # CPU/memory/disk usage shows in metrics
    elif root == "infrastructure":
        return "metrics"       # Latency spikes and availability in metrics
    elif root == "healthy":
        return "logs"          # Check logs to confirm nothing is erroring
    return "metrics"


# Rationale templates per root cause — teaches the model WHY to pick each tool
_RATIONALES = {
    "deploy_regression": "Recent deployments are a common cause of regressions. Check deployment history first to see if a recent change correlates with the issue.",
    "config_change": "Configuration changes can cause unexpected behavior. Check recent deployments and config changes to identify any modifications.",
    "upstream_dependency": "Upstream dependency failures typically produce error logs. Check logs first for connection errors, timeouts, or HTTP 5xx from external services.",
    "resource_exhaustion": "Resource exhaustion (CPU, memory, disk) shows up in system metrics. Check metrics first for spikes or sustained high utilization.",
    "infrastructure": "Infrastructure issues like network problems or DNS failures cause latency spikes. Check metrics for anomalies in latency and error rates.",
    "healthy": "Start by checking logs to verify there are no error patterns. If logs are clean, the service may be operating normally.",
}


def generate_sft(scenarios: list) -> list:
    rows = []
    for sc in scenarios:
        service = sc["service"]
        root = sc["root_cause"]
        first = first_tool_for(sc)
        rationale = _RATIONALES.get(root, f"Check {first} for initial evidence.")
        
        user_text = json.dumps({
            "question": f"Investigate {service}",
            "service": service,
            "evidence_keys": [],
            "available_tools": ["metrics", "logs", "deployments", "final"],
            "instruction": "Choose the NEXT action as JSON {action, args, rationale}.",
        })
        assistant_text = json.dumps({
            "action": first,
            "args": {"service": service},
            "rationale": rationale,
        })
        rows.append({
            "messages": [
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]
        })
    return rows



def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate ops agent scenarios")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard", action="store_true", help="More red herrings and confounders")
    args = parser.parse_args()

    hard_ratio = 0.6 if args.hard else 0.3

    scenarios = generate_dataset(count=args.count, seed=args.seed, hard_ratio=hard_ratio)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "version": "3.0",
        "description": f"Procedurally generated: {args.count} scenarios, seed={args.seed}, hard_ratio={hard_ratio}",
        "count": len(scenarios),
        "scenarios": scenarios,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(scenarios)} scenarios to {OUT_PATH}")

    rng = random.Random(args.seed)
    questions = generate_questions(scenarios, rng)
    q_path = REPO_ROOT / "app" / "eval" / "questions.json"
    q_path.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(questions)} eval questions to {q_path}")

    sft_rows = generate_sft(scenarios)
    sft_path = REPO_ROOT / "data" / "trl_planner_sft.jsonl"
    with sft_path.open("w", encoding="utf-8") as f:
        for r in sft_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(sft_rows)} SFT training rows to {sft_path}")

    by_cause = {}
    by_diff = {}
    for sc in scenarios:
        by_cause.setdefault(sc["root_cause"], []).append(sc["service"])
        by_diff.setdefault(sc["difficulty"], 0)
        by_diff[sc["difficulty"]] += 1

    print(f"\nRoot cause distribution:")
    for cause, services in sorted(by_cause.items()):
        print(f"  {cause}: {len(services)}")
    print(f"\nDifficulty: {by_diff}")


if __name__ == "__main__":
    main()
