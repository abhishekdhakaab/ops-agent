#!/usr/bin/env python3
"""
Unseen Data Evaluation for EvalLLM
====================================

Generates NEW scenarios with services the model has NEVER seen
during training, then evaluates base / SFT / GRPO models on them.

This tests whether the model learned generalizable planning
(conditional reasoning from evidence) or just memorized 100 mappings.

Usage:
    # Evaluate all three models on 30 unseen scenarios:
    python unseen_eval.py --model qwen2.5:1.5b

    # Evaluate a single model:
    python unseen_eval.py --model-path data/trained_models/grpo_on_sft_v2/final

    # Generate scenarios only (inspect before eval):
    python unseen_eval.py --generate-only

    # Custom count:
    python unseen_eval.py --model qwen2.5:1.5b --count 20

Self-contained — does not modify any existing files or data.
"""

import json
import sys
import re
import time
import random
import argparse
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = DATA_DIR / "eval_results"

# Import shared prompt format from sft_pipeline
sys.path.insert(0, str(SCRIPT_DIR / "sft_pipeline"))
from prompts import PLANNER_SYSTEM, build_planner_state, build_planner_user_message
import sys
sys.path.insert(0, str(SCRIPT_DIR / "sft_pipeline"))
from planner_policy import normalize_action, best_first_tools


# ══════════════════════════════════════════════════════════════════
# UNSEEN SCENARIO GENERATION
# ══════════════════════════════════════════════════════════════════

# Services that were NEVER in training data
UNSEEN_SERVICES = [
    "auth-gateway", "log-aggregator", "report-engine", "data-pipeline",
    "graphql-proxy", "feature-flag-svc", "ab-test-engine", "sitemap-builder",
    "fraud-detector", "address-validator", "tax-calculator", "loyalty-svc",
    "push-notifier", "video-transcoder", "comment-svc", "badge-svc",
    "leaderboard-svc", "referral-engine", "coupon-svc", "waitlist-svc",
    "chat-relay", "event-bus", "secret-manager", "compliance-checker",
    "geo-lookup", "translation-svc", "spell-checker", "digest-worker",
    "backup-agent", "migration-runner",
]

# Error message templates per root cause (different from training)
_LOG_MESSAGES = {
    "deploy_regression": [
        ["segfault in request handler: null pointer at 0x7f3a", "core dump generated: 4.2GB", "worker restart loop detected: 12 restarts in 60s"],
        ["memory leak detected: RSS grew 800MB in 10min", "GC overhead exceeded 90%: pause 120ms", "malloc failed: cannot allocate 512MB"],
        ["panic: runtime error: index out of range [5] with length 3", "stack overflow in recursive handler", "thread pool exhausted: 200/200 threads busy"],
    ],
    "resource_exhaustion": [
        ["disk usage 97%: /var/log filling up", "inode exhaustion on /data partition", "write failed: no space left on device"],
        ["thread count 847: approaching ulimit 1024", "file descriptor leak: 98% of fd limit used", "connection pool full: 500/500 connections"],
        ["memory pressure: swapping 2.1GB/s", "OOM score adjustment triggered for pid 2341", "cgroup memory limit hit: 16GB/16GB"],
    ],
    "upstream_dependency": [
        ["ETIMEDOUT connecting to postgres-replica:5432", "redis sentinel failover in progress", "grpc deadline exceeded calling inventory-api"],
        ["certificate verify failed: unable to get local issuer cert", "connection reset by peer: auth-service:443", "max retries exceeded for url: /api/v2/validate"],
        ["upstream connect error: connection refused to ml-scoring:8080", "name resolution failed for cache-primary.internal", "circuit open for payment-gateway: 50 failures in 30s"],
    ],
    "healthy": [
        ["minor: slow query 85ms (threshold 100ms)", "rate limit applied for client test-bot"],
        ["GC pause 12ms within normal range", "connection pool recycled: routine maintenance"],
        ["debug: health check passed in 3ms", "info: cache hit ratio 94%"],
    ],
    "config_change": [
        ["invalid regex in routing rule: unterminated group", "feature toggle payment-v3 returned unexpected variant", "config hot-reload failed: schema validation error"],
        ["rate limit misconfigured: 0 requests/sec for tier premium", "TLS cert rotation failed: new cert not yet valid", "environment variable DB_POOL_SIZE=0 is invalid"],
        ["CORS policy blocking requests from new frontend domain", "webhook URL changed to unreachable endpoint", "cache TTL set to 0: every request hits database"],
    ],
    "infrastructure": [
        ["etcd cluster lost quorum: 1/3 nodes responding", "kubernetes node NotReady: kubelet stopped posting status", "NFS mount stale: server not responding"],
        ["DRBD split-brain detected on volume 0", "ceph OSD down: 2 OSDs not reporting", "BGP session to upstream router expired"],
        ["hypervisor memory overcommit: balloon driver reclaiming", "SAN path failover: primary HBA offline", "DNS forwarder unreachable: all queries timing out"],
    ],
}

_DEPLOY_SUMMARIES = {
    "deploy_regression": [
        "Migrate from sync to async request handling",
        "Upgrade gRPC library from v1.40 to v1.58",
        "Rewrite connection pooling with custom allocator",
        "Switch JSON parser from simdjson to native",
        "Add request body validation middleware",
    ],
    "config_change": [
        "Update rate limiting rules for API v3",
        "Rotate TLS certificates for internal services",
        "Modify feature flag evaluation logic",
        "Change database connection pool parameters",
        "Update CORS allowed origins list",
    ],
}

_QUESTIONS = {
    "deploy_regression": [
        "{service} started crashing after a recent change. What happened?",
        "We're seeing restarts and failures in {service}. Please investigate.",
        "{service} is down. Might be related to something we shipped today.",
    ],
    "resource_exhaustion": [
        "{service} is grinding to a halt. Everything is slow.",
        "Alerts firing for {service} — looks like it's running out of resources.",
        "{service} performance has been degrading over the last hour.",
    ],
    "upstream_dependency": [
        "{service} is returning errors for some requests but not others.",
        "Intermittent failures in {service}. Clients are complaining.",
        "{service} seems to be having trouble reaching something downstream.",
    ],
    "healthy": [
        "Got a page for {service}. Is there actually a problem?",
        "Quick check on {service} — someone mentioned it might be flaky.",
        "Monitoring flagged {service}. Can you confirm it's fine?",
    ],
    "config_change": [
        "{service} is behaving weirdly since this morning. Not sure what changed.",
        "Something is off with {service}. Requests are failing in unexpected ways.",
        "{service} seems misconfigured. Can you take a look?",
    ],
    "infrastructure": [
        "{service} is completely unreachable. This looks like a major outage.",
        "Multiple alerts from {service}. Might be infrastructure-level.",
        "{service} and several other services went down simultaneously.",
    ],
}


def generate_unseen_scenarios(count: int = 30, seed: int = 42) -> list:
    """Generate new scenarios with unseen services."""
    # Verify no overlap with training scenarios
    import json as _json
    from pathlib import Path as _Path
    _scenarios_path = _Path(__file__).resolve().parent / "data" / "scenarios.json"
    if _scenarios_path.exists():
        _train_data = _json.loads(_scenarios_path.read_text())
        _train_services = {s["service"] for s in _train_data.get("scenarios", [])}
        _overlap = set(UNSEEN_SERVICES) & _train_services
        if _overlap:
            raise ValueError(
                f"unseen_eval: {len(_overlap)} service(s) overlap with training data: {_overlap}. "
                "Update UNSEEN_SERVICES to use names not present in scenarios.json."
            )
        print(f"  Verified: 0 service overlap with {len(_train_services)} training services.")

    rng = random.Random(seed)

    root_causes = ["deploy_regression", "resource_exhaustion", "upstream_dependency",
                   "healthy", "config_change", "infrastructure"]
    severities = {"deploy_regression": "high", "resource_exhaustion": "high",
                  "upstream_dependency": "high", "healthy": "low",
                  "config_change": "medium", "infrastructure": "high"}
    correct_actions = {"deploy_regression": "rollback", "resource_exhaustion": "mitigate",
                       "upstream_dependency": "escalate", "healthy": "investigate",
                       "config_change": "rollback", "infrastructure": "escalate"}

    services = list(UNSEEN_SERVICES)
    rng.shuffle(services)

    scenarios = []
    for i in range(count):
        service = services[i % len(services)]
        root_cause = root_causes[i % len(root_causes)]
        severity = severities[root_cause]

        # Metrics
        if root_cause in ("deploy_regression", "resource_exhaustion", "upstream_dependency", "infrastructure"):
            has_spike = True
            avg_lat = round(rng.uniform(200, 900), 1)
            p95_lat = round(avg_lat * rng.uniform(2.0, 3.5), 1)
            spike_start = rng.randint(10, 45)
        else:
            has_spike = False
            avg_lat = round(rng.uniform(15, 80), 1)
            p95_lat = round(avg_lat * rng.uniform(1.5, 2.5), 1)
            spike_start = None

        # Logs
        log_templates = _LOG_MESSAGES[root_cause]
        error_msgs = rng.choice(log_templates)
        if root_cause == "healthy":
            error_count = rng.randint(0, 5)
            log_level = "WARN"
        else:
            error_count = rng.randint(50, 400)
            log_level = "ERROR"

        # Deployments
        if root_cause in ("deploy_regression", "config_change"):
            has_deploy = True
            deploy_mins = rng.randint(15, 90)
            summaries = _DEPLOY_SUMMARIES.get(root_cause, _DEPLOY_SUMMARIES["deploy_regression"])
            deploy_summary = rng.choice(summaries)
            deploy_author = rng.choice(["alice", "bob", "carlos", "diana", "evan"])
        else:
            has_deploy = rng.random() < 0.15
            deploy_mins = rng.randint(500, 5000)
            deploy_summary = "Routine dependency update"
            deploy_author = "ci-bot"

        # Question
        q_templates = _QUESTIONS[root_cause]
        question = rng.choice(q_templates).format(service=service)

        # Ground truth
        if root_cause == "deploy_regression":
            must_mention = ["deploy"]
        elif root_cause == "infrastructure":
            must_mention = ["infrastructure"]
        elif root_cause == "config_change":
            must_mention = ["config"]
        elif root_cause == "upstream_dependency":
            must_mention = ["error"]
        else:
            must_mention = []

        scenario = {
            "id": f"unseen_{service}_{i}",
            "service": service,
            "root_cause": root_cause,
            "severity": severity,
            "question": question,
            "metrics": {
                "has_latency_spike": has_spike,
                "avg_latency_ms": avg_lat,
                "p95_latency_ms": p95_lat,
                "spike_start_minutes_ago": spike_start,
            },
            "logs": {
                "error_count": error_count,
                "error_messages": error_msgs,
                "log_level": log_level,
            },
            "deployments": {
                "has_recent_deploy": has_deploy,
                "deploy_minutes_ago": deploy_mins,
                "deploy_summary": deploy_summary,
                "deploy_author": deploy_author,
            },
            "ground_truth": {
                "root_cause": root_cause,
                "severity": severity,
                "correct_action": correct_actions[root_cause],
                "must_mention": must_mention,
            },
        }
        scenarios.append(scenario)

    return scenarios


# ══════════════════════════════════════════════════════════════════
# TOOL SIMULATION (for unseen scenarios)
# ══════════════════════════════════════════════════════════════════

def simulate_tool(tool_name: str, args: dict, scenario: dict) -> dict:
    """Simulate a tool call using the scenario's data."""
    service = args.get("service", scenario["service"])

    if tool_name == "metrics":
        m = scenario["metrics"]
        spike = m["has_latency_spike"]
        avg = m["avg_latency_ms"] if spike else m["avg_latency_ms"] * 0.42
        p95 = m["p95_latency_ms"] if spike else m["p95_latency_ms"] * 0.38
        return {
            "spike_detected": spike,
            "avg_latency_ms": round(avg, 1),
            "p95_latency_ms": round(p95, 1),
        }

    elif tool_name == "logs":
        lg = scenario["logs"]
        return {
            "error_count": lg["error_count"],
            "common_messages": lg["error_messages"][:3],
        }

    elif tool_name == "deployments":
        dep = scenario["deployments"]
        return {
            "has_recent_deploy": dep["has_recent_deploy"] and dep["deploy_minutes_ago"] < 120,
            "minutes_ago": dep["deploy_minutes_ago"],
            "summary": dep["deploy_summary"],
            "author": dep["deploy_author"],
        }

    return {"error": f"Unknown tool: {tool_name}"}


# ══════════════════════════════════════════════════════════════════
# REWARD + TOOL ACCURACY
# ══════════════════════════════════════════════════════════════════

def compute_reward(completion: str, scenario: dict) -> float:
    """Score a completion. Same logic as sft_eval.py."""
    reward = 0.0

    text = completion.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    parsed = None
    try:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
    except Exception:
        pass

    if parsed is None:
        return -1.0

    reward += 0.5

    has_action = "action" in parsed
    has_args = "args" in parsed

    if has_action and has_args:
        reward += 0.3
    elif has_action:
        reward += 0.1

    if not has_action:
        return reward

    action = normalize_action(parsed.get("action", ""))
    valid_actions = {"metrics", "logs", "deployments", "final"}

    if action in valid_actions:
        reward += 0.2
    else:
        reward -= 0.3
        return reward

    best = best_first_tools(scenario)
    if action in best:
        reward += 1.0
    elif action in {"metrics", "logs", "deployments"} and action != "final":
        reward += 0.3
    elif action == "final":
        reward -= 0.2

    args = parsed.get("args", {})
    if isinstance(args, dict):
        service = args.get("service", "")
        if service and service == scenario.get("service", ""):
            reward += 0.3
        elif service:
            reward += 0.1

    rationale = parsed.get("rationale", "")
    if isinstance(rationale, str) and len(rationale) > 10:
        reward += 0.2

    return round(reward, 3)


# ══════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════

def _ollama_to_hf(model_name: str) -> str:
    mapping = {
        "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen2.5:3b": "Qwen/Qwen2.5-3B-Instruct",
        "gemma3:1b": "google/gemma-3-1b-it",
        "gemma3:4b": "google/gemma-3-4b-it",
    }
    return mapping.get(model_name, model_name)


def load_model(model_path: str):
    """Load model and tokenizer."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    is_lora = (Path(model_path) / "adapter_config.json").exists() if ":" not in model_path else False

    if is_lora:
        from peft import PeftModel
        adapter_cfg = json.loads((Path(model_path) / "adapter_config.json").read_text())
        base_name = adapter_cfg.get("base_model_name_or_path", "")
        print(f"  Loading LoRA adapter: {model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(base_name, trust_remote_code=True)
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    else:
        hf_path = _ollama_to_hf(model_path)
        print(f"  Loading model: {hf_path}")
        model = AutoModelForCausalLM.from_pretrained(hf_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.to(device)
    model.eval()
    return model, tokenizer, device


# SCOPE NOTE: This eval tests only the first tool decision (step 0).
# Every prompt is built with empty evidence and no prior tool calls.
# It does not test multi-step investigation planning.
# See: first_tool_accuracy in results.

# ══════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════

def run_eval(model_path: str, scenarios: list, label: str = "model") -> dict:
    """Evaluate a model on unseen scenarios."""
    import torch
    from collections import Counter

    print(f"\n{'='*55}")
    print(f"  Unseen eval: {label}")
    print(f"  Model: {model_path}")
    print(f"  Scenarios: {len(scenarios)} (all unseen services)")
    print(f"{'='*55}\n")

    model, tokenizer, device = load_model(model_path)
    print(f"  Device: {device}")

    rewards = []
    correct_tools = 0
    valid_json = 0
    total = len(scenarios)
    per_rc = {}

    t0 = time.time()

    for i, sc in enumerate(scenarios):
        service = sc["service"]
        question = sc["question"]

        # Build prompt — SAME format as training
        state = build_planner_state(
            question=question,
            service=service,
            step=0,
            tools_called=[],
            evidence={},
        )
        user_msg = build_planner_user_message(state)

        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": user_msg},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"{PLANNER_SYSTEM}\n\nUSER:\n{user_msg}\n\nASSISTANT:\n"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        reward = compute_reward(completion, sc)
        rewards.append(reward)

        best = best_first_tools(sc)
        action = ""
        try:
            m = re.search(r"\{.*\}", completion, flags=re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
                action = parsed.get("action", "")
                if action in best:
                    correct_tools += 1
                if isinstance(parsed, dict) and "action" in parsed:
                    valid_json += 1
        except Exception:
            pass

        # Track per root cause
        rc = sc["root_cause"]
        if rc not in per_rc:
            per_rc[rc] = {"correct": 0, "total": 0, "rewards": []}
        per_rc[rc]["total"] += 1
        per_rc[rc]["rewards"].append(reward)
        if action in best:
            per_rc[rc]["correct"] += 1

        status = "✓" if action in best else "✗"
        print(f"  [{i+1}/{total}] {service:20s} ({rc:20s}) {status} → {action:12s} (expected: {best})  r={reward:+.1f}")

    elapsed = time.time() - t0

    avg_reward = sum(rewards) / max(1, total)
    tool_acc = correct_tools / max(1, total)
    json_rate = valid_json / max(1, total)
    strong = sum(1 for r in rewards if r > 1.5)

    result = {
        "label": label,
        "model_path": str(model_path),
        "eval_type": "unseen_scenarios",
        "scenarios": total,
        "first_step_avg_reward": round(avg_reward, 3),
        "first_tool_accuracy": round(tool_acc, 3),
        "json_valid_rate": round(json_rate, 3),
        "first_step_strong_pct": round(strong / max(1, total), 3),
        "total_time_s": round(elapsed, 1),
        "per_root_cause": {},
    }

    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  {label:^43s} │")
    print(f"  ├─────────────────────────────────────────────┤")
    print(f"  │  First-tool accuracy: {tool_acc*100:>6.1f}%              │")
    print(f"  │  First-step avg reward: {avg_reward:>+7.3f}            │")
    print(f"  │  Valid JSON rate:   {json_rate*100:>7.1f}%              │")
    print(f"  │  Strong (>1.5):     {strong:>3d}/{total} ({strong/max(1,total)*100:.0f}%)           │")
    print(f"  │  Time: {elapsed:.1f}s                               │")
    print(f"  ├─────────────────────────────────────────────┤")
    print(f"  │  Per root cause:                             │")

    for rc in sorted(per_rc.keys()):
        info = per_rc[rc]
        rc_acc = info["correct"] / max(1, info["total"])
        rc_reward = sum(info["rewards"]) / max(1, len(info["rewards"]))
        print(f"  │  {rc:22s} acc={rc_acc*100:5.1f}% r={rc_reward:+.2f}  │")
        result["per_root_cause"][rc] = {
            "accuracy": round(rc_acc, 3),
            "avg_reward": round(rc_reward, 3),
            "count": info["total"],
        }

    print(f"  └─────────────────────────────────────────────┘")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = label.replace(" ", "_").replace(":", "_").replace("/", "_")
    out_path = RESULTS_DIR / f"unseen_eval_{safe}.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")

    return result


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Unseen data evaluation")
    parser.add_argument("--model", default="qwen2.5:1.5b", help="Base model name")
    parser.add_argument("--model-path", default=None, help="Path to specific model")
    parser.add_argument("--count", type=int, default=30, help="Number of unseen scenarios")
    parser.add_argument("--generate-only", action="store_true", help="Just generate and print scenarios")
    args = parser.parse_args()

    # Generate unseen scenarios
    scenarios = generate_unseen_scenarios(count=args.count)

    if args.generate_only:
        print(f"\n  Generated {len(scenarios)} unseen scenarios:\n")
        from collections import Counter
        rc_dist = Counter(s["root_cause"] for s in scenarios)
        for rc, c in sorted(rc_dist.items()):
            print(f"    {rc:25s}: {c}")
        print(f"\n  Services (all unseen):")
        for s in scenarios[:10]:
            print(f"    {s['service']:20s} → {s['root_cause']:20s} Q: {s['question'][:60]}")
        if len(scenarios) > 10:
            print(f"    ... and {len(scenarios)-10} more")
        print(f"\n  Sample scenario:")
        print(json.dumps(scenarios[0], indent=2))
        return

    if args.model_path:
        # Single model eval
        run_eval(args.model_path, scenarios, label=args.model_path)
        return

    # Full comparison: base → SFT → GRPO
    results = []

    # Base
    print("\n  Evaluating BASE model on unseen data...")
    r = run_eval(args.model, scenarios, label="Base (unseen)")
    results.append(r)

    # SFT
    sft_path = str(DATA_DIR / f"trained_models/sft_{args.model.replace(':', '_').replace('/', '_')}" / "final")
    if Path(sft_path).exists():
        print("\n  Evaluating SFT model on unseen data...")
        r = run_eval(sft_path, scenarios, label="SFT (unseen)")
        results.append(r)

    # GRPO
    grpo_path = str(DATA_DIR / "trained_models" / "grpo_on_sft_v2" / "final")
    if Path(grpo_path).exists():
        print("\n  Evaluating GRPO model on unseen data...")
        r = run_eval(grpo_path, scenarios, label="GRPO (unseen)")
        results.append(r)

    # DPO (standard preference-learning fallback — evaluated if trained)
    dpo_path = str(DATA_DIR / "trained_models" / "dpo_on_sft_v1" / "final")
    if Path(dpo_path).exists():
        print("\n  Evaluating DPO model on unseen data...")
        r = run_eval(dpo_path, scenarios, label="DPO (unseen)")
        results.append(r)

    # Comparison table
    if len(results) >= 2:
        print(f"\n  ╔═══════════════════════════════════════════════════════════╗")
        print(f"  ║         UNSEEN DATA COMPARISON                            ║")
        print(f"  ║  (services the model has NEVER seen during training)       ║")
        print(f"  ╠═══════════════════════════════════════════════════════════╣")
        print(f"  ║  {'Model':<20s} {'Reward':>8s} {'ToolAcc':>8s} {'JSON%':>7s} ║")
        print(f"  ╠═══════════════════════════════════════════════════════════╣")

        for r in results:
            label = r['label'][:20]
            print(f"  ║  {label:<20s} {r['first_step_avg_reward']:>+8.3f} {r['first_tool_accuracy']*100:>7.1f}% {r['json_valid_rate']*100:>6.1f}% ║")

        print(f"  ╠═══════════════════════════════════════════════════════════╣")

        base = results[0]
        best = max(results, key=lambda r: r.get("first_tool_accuracy", r.get("tool_accuracy", 0)))
        d_reward = best["first_step_avg_reward"] - base["first_step_avg_reward"]
        d_tool = (best["first_tool_accuracy"] - base["first_tool_accuracy"]) * 100

        print(f"  ║  IMPROVEMENT: {base['label'][:10]} → {best['label'][:10]} (best by first_tool_accuracy) ║")
        print(f"  ║    Reward:     {d_reward:>+.3f}                                  ║")
        print(f"  ║    Tool acc:   {d_tool:>+.1f}pp                                  ║")
        print(f"  ╚═══════════════════════════════════════════════════════════╝")

        # Seen vs unseen comparison
        print(f"\n  ┌─────────────────────────────────────────────────────────┐")
        print(f"  │  GENERALIZATION CHECK (best model)                      │")
        print(f"  │  Seen data tool acc:   from sft_aligned_eval results    │")
        print(f"  │  Unseen data tool acc: {best['first_tool_accuracy']*100:.1f}%                           │")
        print(f"  │                                                         │")
        print(f"  │  If unseen ≥ 50% of seen → model generalized           │")
        print(f"  │  If unseen < 30% → model memorized                     │")
        print(f"  └─────────────────────────────────────────────────────────┘")


if __name__ == "__main__":
    main()
