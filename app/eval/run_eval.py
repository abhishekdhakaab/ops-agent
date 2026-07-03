"""
Eval runner for the Ops Agent.

Usage:
    python run_eval.py                                  # eval current model
    python run_eval.py --compare gemma3:1b gemma3:4b    # compare models
    python run_eval.py --report                         # view past results
    python run_eval.py --no-judge                       # skip LLM-as-judge (faster)
    python run_eval.py --count 20                       # only run first N questions
"""

import json
import os
import sys
import time
import subprocess
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List
import httpx

PROJ_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_PATH = Path(__file__).parent / "questions.json"
RESULTS_DIR = PROJ_ROOT / "data" / "eval_results"
API_URL = os.getenv("EVAL_API_URL", "http://127.0.0.1:8000/run")
TIMEOUT_S = int(os.getenv("EVAL_TIMEOUT_S", "600"))

def _now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _now_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def _read_model_from_env() -> str:
    """Read LLM_MODEL from .env file (not just process env)."""
    env_path = PROJ_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("LLM_MODEL=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip()
    return os.getenv("LLM_MODEL", "unknown")

sys.path.insert(0, str(PROJ_ROOT))



async def run_eval_async(model_name: str = "unknown", use_judge: bool = True, max_questions: int = 0) -> Dict[str, Any]:
    """Evaluate the running API and return per-question and aggregate scores."""

    # Import after environment selection so comparisons use the active model config.
    from app.eval.smart_scorer import score_full

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)
    
    if max_questions > 0:
        items = items[:max_questions]

    results = []
    scores = {
        "combined": [], "tool_all": [], "diversity": [],
        "efficiency": [], "structured": [], "llm_judge": [],
        "latencies": [], "successes": 0, "errors": 0,
    }

    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        for i, it in enumerate(items):
            qid = it["id"]
            question = it["question"]
            gt = it.get("ground_truth", {})
            print(f"  [{i+1}/{len(items)}] {qid}: {question[:55]}...", end=" ", flush=True)

            start = time.time()
            try:
                # One failed question should not stop the whole benchmark.
                resp = await client.post(API_URL, json={"question": question})
                latency_s = round(time.time() - start, 2)
            except Exception as e:
                latency_s = round(time.time() - start, 2)
                print(f"CONN_ERROR ({latency_s}s)")
                results.append({"id": qid, "error": str(e), "latency_s": latency_s})
                scores["errors"] += 1
                continue

            if resp.status_code != 200:
                print(f"HTTP_{resp.status_code} ({latency_s}s)")
                results.append({"id": qid, "error": f"HTTP {resp.status_code}", "latency_s": latency_s})
                scores["errors"] += 1
                continue

            data = resp.json()

            # The public endpoint omits internal action fields, so reconstruct only
            agent_result = {
                "tools_used": data.get("tools_used", []),
                "final_answer": data.get("final_answer", ""),
                "evidence": data.get("evidence", {}),
                "action_payload": {},
                "validation": {},
            }
            # The API stays small, so these two fields are inferred for scoring.
            answer_text = data.get("final_answer", "").lower()
            if "high" in answer_text:
                agent_result["action_payload"]["severity"] = "high"
            elif "medium" in answer_text:
                agent_result["action_payload"]["severity"] = "medium"
            else:
                agent_result["action_payload"]["severity"] = "low"
            for act in ["rollback", "escalate", "mitigate", "investigate"]:
                if act in answer_text:
                    agent_result["action_payload"]["recommended_action"] = act
                    break
            else:
                agent_result["action_payload"]["recommended_action"] = "investigate"

            score_result = await score_full(
                question=question,
                agent_result=agent_result,
                ground_truth=gt,
                tools_expected_all=it.get("expected_tools_all", []),
                tools_expected_any=it.get("expected_tools_any", []),
                expected_min_tools=it.get("expected_min_tools", 1),
                expected_max_tools=it.get("expected_max_tools", 4),
                use_llm_judge=use_judge,
            )

            combined = score_result["combined_score"]
            scores["combined"].append(combined)
            scores["tool_all"].append(score_result["tool_score"]["all_score"])
            scores["diversity"].append(score_result["diversity"]["diversity_ratio"])
            scores["efficiency"].append(score_result["efficiency"]["efficiency_score"])
            scores["structured"].append(score_result["structured"]["structured_score"])
            if score_result["llm_judge"].get("llm_judge_score") is not None:
                scores["llm_judge"].append(score_result["llm_judge"]["llm_judge_score"])
            scores["latencies"].append(latency_s)
            scores["successes"] += 1

            judge_str = ""
            if score_result["llm_judge"].get("llm_judge_score") is not None:
                judge_str = f" judge={score_result['llm_judge']['llm_judge_score']:.2f}"

            tools = data.get("tools_used", [])
            print(f"score={combined:.2f}{judge_str} tools={tools} ({latency_s}s)")

            results.append({
                "id": qid,
                "question": question,
                "category": it.get("category"),
                "difficulty": it.get("difficulty"),
                "latency_s": latency_s,
                "tools_used": tools,
                "scores": score_result,
                "reasoning": data.get("reasoning", []),
            })

    def safe_avg(lst): return round(sum(lst) / max(1, len(lst)), 4)

    summary = {
        "model": model_name,
        "timestamp": _now_iso(),
        "total_questions": len(items),
        "success_count": scores["successes"],
        "error_count": scores["errors"],
        "success_rate": round(scores["successes"] / max(1, len(items)), 4),
        # These are the stable headline metrics used by comparison reports.
        "combined_score": safe_avg(scores["combined"]),
        "tool_selection_accuracy": safe_avg(scores["tool_all"]),
        "tool_diversity": safe_avg(scores["diversity"]),
        "efficiency": safe_avg(scores["efficiency"]),
        "structured_score": safe_avg(scores["structured"]),
        "llm_judge_score": safe_avg(scores["llm_judge"]) if scores["llm_judge"] else None,
        "avg_latency_s": safe_avg(scores["latencies"]),
    }

    return {"summary": summary, "results": results}


def run_eval(model_name: str = "unknown", use_judge: bool = True, max_questions: int = 0) -> Dict[str, Any]:
    """Synchronous wrapper for CLI callers."""
    return asyncio.run(run_eval_async(model_name, use_judge, max_questions))



def save_result(result: Dict[str, Any], model_name: str) -> Path:
    """Write one timestamped evaluation result."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = model_name.replace(":", "_").replace("/", "_")
    ts = _now_stamp()
    path = RESULTS_DIR / f"eval_{safe}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {path}")
    return path


def load_all_results() -> List[Dict[str, Any]]:
    """Load timestamped API evaluations for reporting."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("eval_*.json"))]



def print_comparison(results: List[Dict[str, Any]]):
    """Print a compact comparison of evaluation summaries."""
    if not results:
        print("No eval results found.")
        return

    cols = ["Model", "OK", "Combined", "ToolSel", "Diverse", "Effic", "Struct", "Judge", "Lat(s)"]
    widths = [18, 6, 9, 8, 8, 7, 7, 7, 7]
    header = " | ".join(f"{c:<{w}}" for c, w in zip(cols, widths))
    sep = "-+-".join("-" * w for w in widths)

    print(f"\n{'='*len(header)}")
    print("  MODEL COMPARISON — Ops Agent Eval")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    for r in results:
        s = r["summary"]
        judge_str = f"{s['llm_judge_score']:.3f}" if s.get('llm_judge_score') is not None else "n/a"
        row = [
            s.get("model", "?")[:18],
            f"{s['success_count']}/{s['total_questions']}",
            f"{s['combined_score']:.3f}",
            f"{s['tool_selection_accuracy']:.3f}",
            f"{s['tool_diversity']:.3f}",
            f"{s['efficiency']:.3f}",
            f"{s['structured_score']:.3f}",
            judge_str,
            f"{s['avg_latency_s']:.1f}",
        ]
        print(" | ".join(f"{v:<{w}}" for v, w in zip(row, widths)))

    print(sep)

    if len(results) >= 2:
        best = max(results, key=lambda r: r["summary"]["combined_score"])
        worst = min(results, key=lambda r: r["summary"]["combined_score"])
        if best["summary"]["model"] != worst["summary"]["model"]:
            delta = best["summary"]["combined_score"] - worst["summary"]["combined_score"]
            pct = (delta / max(0.001, worst["summary"]["combined_score"])) * 100
            print(f"\n  {worst['summary']['model']} → {best['summary']['model']}:")
            print(f"  Combined score: {worst['summary']['combined_score']:.3f} → {best['summary']['combined_score']:.3f} (+{pct:.0f}%)")
    print()



def update_env_model(model: str):
    """Select a model in the local environment file for the next server run."""
    env = PROJ_ROOT / ".env"
    lines = env.read_text().splitlines()
    out = []
    for line in lines:
        if line.strip().startswith("LLM_MODEL=") or line.strip().startswith("##LLM_MODEL="):
            out.append(f"LLM_MODEL={model}")
        else:
            out.append(line)
    env.write_text("\n".join(out) + "\n")


def wait_for_server(timeout: int = 30) -> bool:
    """Poll the health endpoint until startup completes or times out."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get("http://127.0.0.1:8000/health", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def run_compare(models: List[str], use_judge: bool = True, max_questions: int = 0):
    """Restart the API per model and evaluate each under identical settings."""
    all_results = []
    for model in models:
        print(f"\n{'='*55}")
        print(f"  EVALUATING: {model}")
        print(f"{'='*55}")

        update_env_model(model)

        # Pulling here keeps the comparison command usable on a fresh machine.
        check = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if model not in check.stdout:
            print(f"  Pulling {model}...")
            subprocess.run(["ollama", "pull", model], check=True)

        server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=str(PROJ_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

        try:
            if not wait_for_server():
                print(f"  Server failed to start for {model}")
                server.terminate()
                continue

            result = run_eval(model_name=model, use_judge=use_judge, max_questions=max_questions)
            save_result(result, model)
            all_results.append(result)

            s = result["summary"]
            print(f"\n  {model}: combined={s['combined_score']:.3f}, tools={s['tool_selection_accuracy']:.3f}, lat={s['avg_latency_s']:.1f}s")

        finally:
            server.terminate()
            server.wait(timeout=5)
            time.sleep(2)

    print_comparison(all_results)



def main():
    """Parse CLI flags and run evaluation or historical reporting."""
    args = sys.argv[1:]

    use_judge = "--no-judge" not in args
    args = [a for a in args if a != "--no-judge"]

    max_q = 0
    if "--count" in args:
        idx = args.index("--count")
        max_q = int(args[idx + 1])
        args = args[:idx] + args[idx+2:]

    if "--report" in args:
        print_comparison(load_all_results())
        return

    if "--compare" in args:
        idx = args.index("--compare")
        models = args[idx + 1:]
        if not models:
            print("Usage: python run_eval.py --compare gemma3:1b gemma3:4b")
            sys.exit(1)
        run_compare(models, use_judge=use_judge, max_questions=max_q)
        return

    model = _read_model_from_env()
    print(f"\n  Running eval (model={model}, judge={'on' if use_judge else 'off'})...\n")
    result = run_eval(model_name=model, use_judge=use_judge, max_questions=max_q)
    save_result(result, model)

    print("\n=== Summary ===")
    for k, v in result["summary"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
