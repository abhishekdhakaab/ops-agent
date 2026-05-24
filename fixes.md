# Ops Agent — Complete Fix Instructions

> These instructions are written for an agent that will execute every fix in order.
> Follow each section exactly. Do not skip any fix. Do not reorder fixes.
> After all fixes are applied, re-run the evaluation scripts as described in the final section.

---

## Fix 1 — `best_first_tools()` inconsistency across files

### Problem
There are four different copies of `best_first_tools()` / `_best_first_tools()` across the codebase and they disagree on two root causes:

| Root cause | `train.py` | `sft_eval.py` | `unseen_eval.py` |
|---|---|---|---|
| `infrastructure` | `metrics` | `logs` | `logs` |
| `resource_exhaustion` | `metrics` only | `{metrics, logs}` | `{metrics, logs}` |
| `config_change` | `deployments` only | `{logs, metrics}` | `{logs, metrics}` |

The model was rewarded during GRPO training using `train.py`'s logic but scored during evaluation using `sft_eval.py`'s different logic. This means the published numbers (36.7% → 60% → 80%) were produced with inconsistent definitions of correctness.

### Fix

**Step 1.** Create a new shared module at `sft_pipeline/planner_policy.py` with the following content. This becomes the single canonical source of truth.

```python
"""
Canonical planner policy definitions shared across training, eval, and scoring.

Import this module instead of defining best_first_tools() locally in each file.
"""

import re
from typing import Set


VALID_ACTIONS = {"metrics", "logs", "deployments", "final"}

ACTION_ALIASES = {
    "metric": "metrics",
    "log": "logs",
    "deploy": "deployments",
    "deployment": "deployments",
    "done": "final",
    "finish": "final",
    "complete": "final",
    "end": "final",
}


def normalize_action(action: str) -> str:
    """
    Normalize a model-produced action string to a valid action.
    Handles common typos and aliases the model produces.
    Returns the normalized string, or the original if no match.
    """
    if not isinstance(action, str):
        return ""
    action = action.lower().strip()
    if action in VALID_ACTIONS:
        return action
    return ACTION_ALIASES.get(action, action)


def best_first_tools(scenario: dict) -> Set[str]:
    """
    Returns the set of acceptable first tools for a given scenario.

    Based on Phase 2 synthesis results across 100 scenarios.
    This is the canonical version — use this everywhere.

    Mapping rationale:
      deploy_regression   → deployments (question usually mentions recent change)
      config_change       → deployments OR logs (either is reasonable)
      upstream_dependency → logs (question usually mentions errors)
      resource_exhaustion → metrics OR logs (both acceptable starting points)
      infrastructure      → logs (infrastructure issues surface in logs first)
      healthy             → metrics OR logs (either reasonable for health check)
    """
    root = scenario.get("root_cause", "")
    if root == "deploy_regression":
        return {"deployments"}
    elif root == "config_change":
        return {"deployments", "logs", "metrics"}
    elif root == "upstream_dependency":
        return {"logs"}
    elif root == "resource_exhaustion":
        return {"metrics", "logs"}
    elif root == "infrastructure":
        return {"logs"}
    elif root == "healthy":
        return {"metrics", "logs"}
    return {"metrics", "logs"}


def question_hint_first_tool(question: str) -> Set[str]:
    """
    Returns preferred first tools based on question wording alone.
    Use this as a secondary signal when question wording is unambiguous.

    Only returns a non-empty set when the wording is strongly indicative.
    """
    q = question.lower()
    deploy_words = ["deploy", "release", "shipped", "changed", "rollout", "push"]
    error_words = ["error", "5xx", "fail", "crash", "exception", "throwing", "broken"]
    perf_words = ["slow", "latency", "timeout", "performance", "p95", "response time"]

    hints: Set[str] = set()
    if any(w in q for w in deploy_words):
        hints.add("deployments")
    if any(w in q for w in error_words):
        hints.add("logs")
    if any(w in q for w in perf_words):
        hints.add("metrics")
    return hints
```

**Step 2.** Update `sft_eval.py`:
- Remove the local `best_first_tools()` function definition (lines 41–57).
- Remove the local `normalize_action` logic inside `compute_reward()`.
- Add this import at the top of the file (after existing imports):
```python
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "sft_pipeline"))
from planner_policy import normalize_action, best_first_tools
```
- Inside `compute_reward()`, find the line:
```python
action = parsed.get("action", "").lower().strip()
```
Replace it with:
```python
action = normalize_action(parsed.get("action", ""))
```

**Step 3.** Update `unseen_eval.py` with the exact same changes as Step 2.

**Step 4.** Update `app/training/train.py`:
- Remove the local `_best_first_tools()` function definition entirely.
- Remove the local `compute_reward()` that references it.
- These functions are dead in `train.py` (only `run_sft()` is used). Removing them eliminates the inconsistency risk. Add a comment at the top:
```python
# NOTE: compute_reward and _best_first_tools removed.
# Canonical versions live in sft_pipeline/planner_policy.py
```

**Step 5.** Update `sft_pipeline/grpo_train.py`:
- Find the local `normalize_action` block (the dict of aliases at the top of `compute_reward()`).
- Replace it with an import:
```python
from planner_policy import normalize_action, VALID_ACTIONS
```
- Remove the local alias dict and the local normalization loop inside `compute_reward()`, replacing them with a single call:
```python
action = normalize_action(parsed.get("action", ""))
```

---

## Fix 2 — Eval only tests step 0 — rename metrics to be honest

### Problem
Both `sft_eval.py` and `unseen_eval.py` build every evaluation prompt with `step=0, tools_called=[], evidence={}`. They never simulate tool calls and never test subsequent decisions. The reported `tool_accuracy` and `avg_reward` only reflect first-tool selection, not multi-step planning. Reporting them without qualification is misleading.

### Fix

**In `sft_eval.py`**, find the `result` dict that is built at the end of `run_eval()`:
```python
result = {
    ...
    "tool_accuracy": round(tool_acc, 3),
    "avg_reward": round(avg_reward, 3),
    "strong_reward_pct": round(strong / max(1, total), 3),
    ...
}
```
Change the keys to:
```python
result = {
    ...
    "first_tool_accuracy": round(tool_acc, 3),
    "first_step_avg_reward": round(avg_reward, 3),
    "first_step_strong_pct": round(strong / max(1, total), 3),
    ...
}
```

Also update the print block to use the same labels:
```python
print(f"  │  First-tool accuracy: {tool_acc*100:>6.1f}%            │")
print(f"  │  First-step avg reward: {avg_reward:>+7.3f}          │")
```

Apply the identical key renames in `unseen_eval.py`.

Also add this comment block directly above the `run_eval()` function in both files:
```python
# SCOPE NOTE: This eval tests only the first tool decision (step 0).
# Every prompt is built with empty evidence and no prior tool calls.
# It does not test multi-step investigation planning.
# See: first_tool_accuracy in results.
```

---

## Fix 3 — Action normalization not applied in eval scripts

### Problem
`grpo_train.py` normalizes model outputs like `"deploy"` → `"deployments"` before scoring. Neither `sft_eval.py` nor `unseen_eval.py` do this. A model that outputs `"deploy"` gets rewarded during GRPO training but penalized during eval for the same output.

### Fix
This is resolved by Fix 1 Step 2 and Step 3 above. Once `normalize_action()` from `planner_policy.py` is applied to `action` inside `compute_reward()` in both eval files, normalization is consistent everywhere. Verify that the line:
```python
action = normalize_action(parsed.get("action", ""))
```
appears in `compute_reward()` in both `sft_eval.py` and `unseen_eval.py` after applying Fix 1.

---

## Fix 4 — Phase 2 rationale future-evidence leakage

### Problem
Phase 2 instructs the synthesizer: "Each rationale must reference ONLY evidence from PREVIOUS steps." But `_validate_trajectory()` does not check this. A rationale at step 0 can say "logs show OOM errors" before logs were ever called. These contaminated rationales become assistant-turn targets in the SFT dataset, teaching the model to reference evidence it hasn't seen yet.

### Fix

In `sft_pipeline/phase2_synthesize.py`, find the `_validate_trajectory()` function. After the existing checks, add the following evidence-leakage check before the final `return True, "OK"`:

```python
# Check for future-evidence leakage in rationales
EVIDENCE_KEYWORDS = {
    "logs":        ["log", "error", "oom", "exception", "crash", "traceback",
                    "timeout", "warn", "stack", "message", "stderr"],
    "metrics":     ["latency", "p95", "spike", "avg", "ms", "throughput",
                    "request rate", "rps", "error rate"],
    "deployments": ["deploy", "release", "rollout", "version", "commit",
                    "author", "diff", "change", "minutes ago"],
}

tools_called_so_far = set()
for i, step in enumerate(trajectory):
    action = step.get("action", "")
    rationale = step.get("rationale", "").lower()

    # Check if rationale mentions evidence from a tool not yet called
    for tool, keywords in EVIDENCE_KEYWORDS.items():
        if tool not in tools_called_so_far:
            for kw in keywords:
                if kw in rationale:
                    return False, (
                        f"Step {i} rationale mentions '{kw}' "
                        f"but '{tool}' has not been called yet (leakage)"
                    )

    if action != "final":
        tools_called_so_far.add(action)
```

This check runs after the existing duplicate-tool and length checks, before `return True, "OK"`.

---

## Fix 5 — Phase 3 rationale/evidence value mismatch

### Problem
Phase 3 correctly re-simulates tools to rebuild the actual evidence state at each step. But it keeps the synthesized rationale from Phase 2 as the assistant target without checking whether the rationale's numeric values match the re-simulated evidence. A rationale can say "p95 is 1900ms" while the re-simulated metrics tool returns p95 = 900ms. The model learns to state numbers that contradict the evidence it sees.

### Fix

In `sft_pipeline/phase3_format_sft.py`, find the section where each step's training example is assembled. It will look approximately like:

```python
example = {
    "messages": [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user",   "content": user_msg},
        {"role": "assistant", "content": json.dumps({
            "action": action,
            "args": sim_args,
            "rationale": rationale,
        })},
    ]
}
```

Before appending this example, add the following rationale validation function and call it:

```python
def _rationale_has_numeric_mismatch(rationale: str, evidence: dict) -> bool:
    """
    Returns True if the rationale contains numeric values (e.g. latency ms)
    that are not present anywhere in the current evidence dict.
    This catches cases where the synthesizer hallucinated specific numbers.
    """
    import re
    # Extract numbers from rationale (e.g. 1900, 900ms, 99%)
    rationale_numbers = set(re.findall(r'\b\d{2,}\b', rationale))
    if not rationale_numbers:
        return False  # No numbers to check

    # Build flat string of all evidence values
    evidence_str = json.dumps(evidence)
    evidence_numbers = set(re.findall(r'\b\d{2,}\b', evidence_str))

    # If rationale has numbers not found anywhere in evidence, flag it
    orphan_numbers = rationale_numbers - evidence_numbers
    return len(orphan_numbers) > 0


# Before appending each example:
if _rationale_has_numeric_mismatch(rationale, evidence_compressed):
    # Replace the rationale with a safe generic one derived from evidence
    if action == "metrics":
        m = evidence_compressed.get("metrics", {})
        rationale = (
            f"Metrics show avg latency {m.get('avg_latency_ms', '?')}ms, "
            f"p95 {m.get('p95_latency_ms', '?')}ms, "
            f"spike_detected={m.get('spike_detected', '?')}."
        )
    elif action == "logs":
        lg = evidence_compressed.get("logs", {})
        rationale = (
            f"Logs show {lg.get('error_count', '?')} errors. "
            f"Top message: {str(lg.get('common_messages', ['?'])[:1])}."
        )
    elif action == "deployments":
        dep = evidence_compressed.get("deployments", {})
        rationale = (
            f"Deployment check: has_recent_deploy={dep.get('has_recent_deploy', '?')}, "
            f"minutes_ago={dep.get('minutes_ago', '?')}."
        )
    # For final, keep rationale as-is (it summarizes all evidence)
```

---

## Fix 6 — Phase 1 auto-completed trajectories should be deprioritized in Phase 2

### Problem
When a Phase 1 run hits the step limit without the model saying `final`, Phase 1 injects an artificial `auto_final=True` step. Phase 2 sees these alongside naturally completed runs and treats them equally. The synthesizer can select a trajectory that was never naturally completed as the "best" one.

### Fix

In `sft_pipeline/phase2_synthesize.py`, find `_build_synthesizer_user_prompt()`. Inside the loop that formats each run, find:

```python
runs_text += f"Completed: {'Yes (reached final)' if run['completed'] else 'No (hit step limit)'}\n"
```

Change it to:

```python
auto = run.get("auto_final", False)
if run["completed"] and not auto:
    completion_label = "Yes (model reached final naturally)"
elif auto:
    completion_label = "No (hit step limit — auto-final injected, lower quality)"
else:
    completion_label = "No (hit step limit)"
runs_text += f"Completed: {completion_label}\n"
```

Then add the following instruction to `SYNTHESIZER_SYSTEM` prompt. Find:
```python
SYNTHESIZER_SYSTEM = """You are an expert SRE reviewing multiple investigation attempts...
```
After the existing rules, add:
```
- IMPORTANT: Attempts marked "auto-final injected" hit the step limit and were not naturally completed.
  Prefer trajectories from naturally completed runs. Only use auto-final runs if no natural completion exists.
```

---

## Fix 7 — GRPO `compute_reward()` crashes on missing `step_info`

### Problem
In `sft_pipeline/grpo_train.py`, line 213:
```python
expected_action = step_info["expected_action"]
```
If the prompt-to-step_info lookup fails (e.g. due to whitespace difference), `step_info` is `{}` and this raises a `KeyError`, crashing the entire training run.

### Fix

In `sft_pipeline/grpo_train.py`, find `compute_reward()`. Replace the line:
```python
expected_action = step_info["expected_action"]
```
with:
```python
expected_action = step_info.get("expected_action")
if not expected_action:
    # Cannot score without knowing the expected action — penalize as invalid
    return -1.0
```

Also find the `reward_fn` closure inside `run_grpo()`:
```python
def reward_fn(completions: List[str], prompts: List[str] = None, **kwargs) -> List[float]:
    rewards = []
    for i, completion in enumerate(completions):
        prompt = prompts[i] if prompts else ""
        info = step_info_lookup.get(prompt, {})
        if not info:
            for key, val in step_info_lookup.items():
                if key[:200] == prompt[:200]:
                    info = val
                    break
        r = compute_reward(completion, info)
        rewards.append(r)
    return rewards
```

After the fallback lookup block, add:
```python
        if not info:
            # Complete lookup failure — cannot score this completion
            rewards.append(-1.0)
            continue
```
So the full loop becomes:
```python
for i, completion in enumerate(completions):
    prompt = prompts[i] if prompts else ""
    info = step_info_lookup.get(prompt, {})
    if not info:
        for key, val in step_info_lookup.items():
            if key[:200] == prompt[:200]:
                info = val
                break
    if not info:
        rewards.append(-1.0)
        continue
    r = compute_reward(completion, info)
    rewards.append(r)
```

---

## Fix 8 — Service-name lookup bug in `app/tools/impl.py`

### Problem
`_load_scenarios()` indexes scenarios by service name:
```python
return {s["service"]: s for s in raw.get("scenarios", [])}
```
If two scenarios share the same service name, the second silently overwrites the first. The live agent can return the wrong scenario's data with no error.

### Fix

In `app/tools/impl.py`, replace the entire `_load_scenarios()` function:

```python
def _load_scenarios() -> dict:
    """
    Load scenario index keyed by service name.
    If multiple scenarios share a service name, the one with the highest
    severity is kept (most interesting for investigation simulation).
    Logs a warning when duplicates are detected.
    """
    import logging
    if not SCENARIOS_PATH.exists():
        return {}
    raw = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    index = {}
    severity_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    for s in raw.get("scenarios", []):
        svc = s["service"]
        if svc in index:
            existing_rank = severity_rank.get(index[svc].get("severity", "low"), 1)
            new_rank = severity_rank.get(s.get("severity", "low"), 1)
            if new_rank > existing_rank:
                logging.warning(
                    f"Duplicate service '{svc}' in scenarios.json — "
                    f"keeping higher-severity entry (id={s.get('id', '?')})"
                )
                index[svc] = s
            # else keep existing
        else:
            index[svc] = s
    return index
```

---

## Fix 9 — Missing `contains` input sanitization in `app/tools/impl.py`

### Problem
`app/tools/impl.py` uses `inp.contains` directly:
```python
if inp.contains and inp.contains.lower() not in msg.lower():
```
If the live agent passes `contains` as a boolean, list, or any non-string type, `.lower()` raises `AttributeError` and crashes the tool call.

`sft_pipeline/tools.py` already has defensive handling for this. Apply the same fix to the production tool.

### Fix

In `app/tools/impl.py`, find the `logs_tool()` function. Before the sample-building loop, add:

```python
# Sanitize contains — model can sometimes produce non-string types
contains = inp.contains
if contains is not None:
    if isinstance(contains, bool):
        contains = None  # boolean contains is meaningless
    elif isinstance(contains, list):
        contains = contains[0] if contains else None
    elif not isinstance(contains, str):
        contains = str(contains)
    if contains:
        contains = contains.lower()
```

Then replace the contains check inside the loop:
```python
if inp.contains and inp.contains.lower() not in msg.lower():
    continue
```
with:
```python
if contains and contains not in msg.lower():
    continue
```

---

## Fix 10 — Seeded randomness in `app/tools/impl.py` to match training distribution

### Problem
`sft_pipeline/tools.py` uses `random.Random(hash(service + "metrics"))` — same service always produces the same numbers. `app/tools/impl.py` uses plain `random.uniform` with no seed — every call produces different numbers. The model learned to reason about specific values during training but sees different values every time in production.

### Fix

In `app/tools/impl.py`, find `_build_series()`:
```python
def _build_series(avg: float, spike: bool, points: int = 10) -> list:
    if spike:
        normal = avg * 0.45
        return [round(normal + (avg - normal) * (i / points) + random.uniform(-8, 8), 1)
                for i in range(points)]
    return [round(avg * (0.93 + 0.014 * i) + random.uniform(-4, 4), 1)
            for i in range(points)]
```

Replace with:
```python
def _build_series(avg: float, spike: bool, service: str = "", points: int = 10) -> list:
    rng = random.Random(hash(service + "metrics_series"))
    if spike:
        normal = avg * 0.45
        return [round(normal + (avg - normal) * (i / points) + rng.uniform(-8, 8), 1)
                for i in range(points)]
    return [round(avg * (0.93 + 0.014 * i) + rng.uniform(-4, 4), 1)
            for i in range(points)]
```

Then find where `_build_series()` is called inside `metrics_tool()` and pass the service name:
```python
# Find the call: series = _build_series(avg, spike)
# Replace with:
series = _build_series(avg, spike, service=inp.service)
```

Also find the `deployments_tool()` function. Find:
```python
"version": f"1.{random.randint(10,20)}.{random.randint(0,9)}",
```
and:
```python
"version": f"1.{random.randint(8,12)}.0",
```
Replace both with seeded versions:
```python
_dep_rng = random.Random(hash(inp.service + "deploy_version"))
# first version line:
"version": f"1.{_dep_rng.randint(10,20)}.{_dep_rng.randint(0,9)}",
# second version line (older deploy):
"version": f"1.{_dep_rng.randint(8,12)}.0",
```

---

## Fix 11 — Evidence overwrites on repeated tool calls in the live agent graph

### Problem
In `app/agent/graph.py`, tool output is stored as:
```python
state["evidence"][plan.action] = out
```
If the agent calls `metrics` twice with different `time_range_minutes`, the second call erases the first result. The agent permanently loses the first call's evidence.

### Fix

In `app/agent/graph.py`, find the line:
```python
state["evidence"][plan.action] = out
```

Replace it with:
```python
# Build a specific evidence key so repeated tool calls don't overwrite each other
_action = plan.action
_args = getattr(plan, "args", {}) or {}
if _action == "metrics":
    _tr = _args.get("time_range_minutes", 60)
    _evidence_key = f"metrics_{_tr}min" if _tr != 60 else "metrics"
elif _action == "logs":
    _contains = _args.get("contains")
    _evidence_key = f"logs_filter_{_contains}" if _contains else "logs"
else:
    _evidence_key = _action
state["evidence"][_evidence_key] = out
```

---

## Fix 12 — `RUN_INDEX` never reloaded from `runs.jsonl` on server restart

### Problem
`run_store.py` has an append-only `runs.jsonl` log but no startup code that reads it back into `RUN_INDEX`. Every server restart produces an empty index. All previously completed or in-progress runs become invisible and return 404.

### Fix

In `app/core/run_store.py`, add the following function after the `_append()` function:

```python
def reload_from_disk() -> int:
    """
    Replay runs.jsonl into RUN_INDEX on startup.
    For each run_id, keeps only the latest record (last write wins).
    Returns the number of runs loaded.
    """
    if not RUNS_PATH.exists():
        return 0
    latest: Dict[str, Dict[str, Any]] = {}
    with RUNS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                run_id = record.get("run_id")
                if run_id:
                    latest[run_id] = record  # last write wins
            except json.JSONDecodeError:
                continue
    RUN_INDEX.update(latest)
    return len(latest)
```

Then in `app/main.py`, find the FastAPI app creation (the line with `app = FastAPI(...)`). Directly after it, add:

```python
from app.core.run_store import reload_from_disk as _reload_runs

@app.on_event("startup")
async def _on_startup():
    count = _reload_runs()
    print(f"[startup] Reloaded {count} runs from disk into RUN_INDEX")
```

---

## Fix 13 — GRPO redundancy penalty is never triggered

### Problem
`compute_reward()` in `grpo_train.py` penalizes `-0.8` for calling a tool already in `tools_called`. But GRPO prompts are built from Phase 2 trajectories which by design never have repeated tools. The `tools_called` list in every prompt is always a clean non-repeating sequence. So the `-0.8` penalty fires only on model outputs that repeat a tool — which is correct — but the model is never given prompts where redundancy would be a tempting mistake to make.

### Fix

In `sft_pipeline/grpo_train.py`, find `generate_grpo_prompts()`. After generating the normal step-level prompts from Phase 2 trajectories, add the following block to inject a controlled set of "redundancy trap" prompts:

```python
# ── Inject redundancy-trap prompts ────────────────────────────────
# These prompts show a state where one tool has already been called
# and give the model a chance to either repeat it (bad) or move on (good).
# This ensures the -0.8 redundancy penalty actually fires during training.

trap_prompts_added = 0
MAX_TRAP_PROMPTS = min(len(entries) // 4, 50)  # at most 25% of dataset

for entry in entries:
    if trap_prompts_added >= MAX_TRAP_PROMPTS:
        break
    trajectory = entry.get("trajectory", [])
    tool_steps = [s for s in trajectory if s.get("action") != "final"]
    if len(tool_steps) < 2:
        continue

    # Build a prompt where the first tool has already been called,
    # but the expected next action is the second tool (not the first again)
    first_tool = tool_steps[0]["action"]
    second_tool = tool_steps[1]["action"]
    first_args = tool_steps[0].get("args", {})
    service = entry.get("service", "unknown")

    # Simulate the first tool output as evidence
    try:
        first_output = run_tool(first_tool, first_args, entry["scenario_id"])
        evidence_so_far = {first_tool: compress_evidence(first_tool, first_output)}
    except Exception:
        continue

    state = build_planner_state(
        question=entry["question"],
        service=service,
        step=1,
        tools_called=[first_tool],
        evidence=evidence_so_far,
    )
    user_msg = build_planner_user_message(state)
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ) if hasattr(tokenizer, "apply_chat_template") else (
        f"{PLANNER_SYSTEM}\n\nUSER:\n{user_msg}\n\nASSISTANT:\n"
    )

    trap_step_info = {
        "expected_action": second_tool,
        "tools_called": [first_tool],
        "service": service,
        "step": 1,
        "is_redundancy_trap": True,
    }
    step_info_lookup[formatted] = trap_step_info
    dataset_rows.append({"prompt": formatted})
    trap_prompts_added += 1

print(f"  Added {trap_prompts_added} redundancy-trap prompts to GRPO dataset")
```

Note: This block requires that `run_tool`, `compress_evidence`, `build_planner_state`, and `build_planner_user_message` are imported and that `tokenizer` is available. Move this block to after the tokenizer is loaded in `run_grpo()`, not inside `generate_grpo_prompts()`. Alternatively, store the trap examples as additional rows in `grpo_prompts.jsonl` by extending `generate_grpo_prompts()` to also write trap rows with an `is_redundancy_trap: true` flag, and the reward function already handles them correctly.

---

## Fix 14 — `sft_eval.py` tests on training data — add explicit warning

### Problem
`sft_eval.py` runs against the same 100 scenarios used to build `sft_dataset.jsonl`. Any accuracy it reports is inflated because the model has seen these exact scenarios. The file should make this limitation explicit.

### Fix

In `sft_eval.py`, find `run_eval()`. At the very start of the function, after loading scenarios, add:

```python
import warnings
warnings.warn(
    "sft_eval.py tests on the TRAINING scenarios (scenarios.json). "
    "Results are not out-of-distribution. Use unseen_eval.py for honest evaluation.",
    UserWarning,
    stacklevel=2,
)
print("\n  WARNING: This eval uses training scenarios. Results may be inflated.")
print("  For honest evaluation, use unseen_eval.py\n")
```

Also rename the output file from:
```python
out_path = RESULTS_DIR / f"sft_aligned_eval_{safe}.json"
```
to:
```python
out_path = RESULTS_DIR / f"in_distribution_eval_{safe}.json"
```

This makes it clear in saved results that this file is in-distribution.

---

## Fix 15 — Cosine similarity silently truncates on vector length mismatch in RAG

### Problem
In `app/rag/retrieve.py`:
```python
def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x,y in zip(a,b))
```
`zip(a, b)` silently stops at the shorter list. If embeddings from two different model versions have different dimensions (e.g. 768 vs 384), the similarity score is computed over only the shared prefix — producing a wrong score with no error.

### Fix

In `app/rag/retrieve.py`, replace the `cosine()` function:

```python
def cosine(a: List[float], b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    Returns 0.0 if vectors have different lengths rather than silently truncating.
    """
    if len(a) != len(b):
        import logging
        logging.warning(
            f"cosine(): vector length mismatch ({len(a)} vs {len(b)}). "
            "Returning 0.0. This likely means embeddings were generated "
            "by different models. Re-run ingest.py to rebuild the index."
        )
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)
```

---

## Fix 16 — `unseen_eval.py` does not verify services are actually unseen

### Problem
`unseen_eval.py` claims the 30 evaluation services were never seen during training, but it never checks this. If training scenarios and unseen scenarios share service names, the 80% result is partially in-distribution.

### Fix

In `unseen_eval.py`, find `generate_unseen_scenarios()`. At the start of the function, after defining `UNSEEN_SERVICES`, add:

```python
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
```

---

## Fix 17 — "Best model" in comparison is always the last evaluated

### Problem
In `unseen_eval.py`, the comparison table computes improvement as:
```python
best = results[-1]
```
If GRPO performs worse than SFT on a given run, it still gets reported as "best." This silently misreports results.

### Fix

In `unseen_eval.py`, find the comparison block. Replace:
```python
best = results[-1]
```
with:
```python
best = max(results, key=lambda r: r.get("first_tool_accuracy", r.get("tool_accuracy", 0)))
```

Also update the comparison label from:
```python
print(f"  IMPROVEMENT ({base['label'][:10]} → {best['label'][:10]})")
```
to:
```python
print(f"  IMPROVEMENT: {base['label'][:10]} → {best['label'][:10]} (best by first_tool_accuracy)")
```

---

## Post-fix validation — re-run the evaluation pipeline

After all 17 fixes are applied, run the following commands in order to verify correctness.

### Step 1 — Verify planner_policy imports cleanly
```bash
cd <repo_root>
python -c "from sft_pipeline.planner_policy import normalize_action, best_first_tools; print('OK')"
```
Expected output: `OK`

### Step 2 — Verify action normalization
```bash
python -c "
from sft_pipeline.planner_policy import normalize_action
assert normalize_action('deploy') == 'deployments'
assert normalize_action('metric') == 'metrics'
assert normalize_action('done') == 'final'
assert normalize_action('logs') == 'logs'
assert normalize_action('garbage') == 'garbage'
print('normalize_action: all assertions passed')
"
```

### Step 3 — Verify best_first_tools canonical mapping
```bash
python -c "
from sft_pipeline.planner_policy import best_first_tools
assert 'deployments' in best_first_tools({'root_cause': 'deploy_regression'})
assert 'logs' in best_first_tools({'root_cause': 'infrastructure'})
assert 'logs' in best_first_tools({'root_cause': 'resource_exhaustion'})
assert 'metrics' in best_first_tools({'root_cause': 'resource_exhaustion'})
print('best_first_tools: all assertions passed')
"
```

### Step 4 — Verify Phase 2 leakage detection
```bash
cd sft_pipeline
python -c "
from phase2_synthesize import _validate_trajectory
bad = [
    {'action': 'metrics', 'args': {'service': 'x'}, 'rationale': 'logs show OOM errors'},
    {'action': 'final', 'args': {}, 'rationale': 'done'}
]
ok, msg = _validate_trajectory(bad, 'x')
assert not ok, f'Expected leakage to be caught, got ok=True'
print(f'Leakage detection working: {msg}')
"
```

### Step 5 — Verify cosine length mismatch is caught
```bash
python -c "
import sys; sys.path.insert(0, 'app/rag')
from retrieve import cosine
result = cosine([1.0, 2.0, 3.0], [1.0, 2.0])
assert result == 0.0, f'Expected 0.0 for mismatched vectors, got {result}'
print('cosine mismatch guard: OK')
"
```

### Step 6 — Verify unseen service overlap check
```bash
python -c "
import sys; sys.path.insert(0, '.')
# This should not raise if UNSEEN_SERVICES has no overlap with scenarios.json
from unseen_eval import generate_unseen_scenarios
scenarios = generate_unseen_scenarios(count=30, seed=42)
print(f'Generated {len(scenarios)} unseen scenarios with no overlap: OK')
"
```

### Step 7 — Run sft_eval on base model (should show WARNING)
```bash
python sft_eval.py --model qwen2.5:1.5b --count 20
```
Expected: WARNING message printed, results saved to `data/eval_results/in_distribution_eval_*.json`

### Step 8 — Run unseen_eval comparison
```bash
python unseen_eval.py --compare --model qwen2.5:1.5b --count 30
```
Expected:
- Service overlap check passes with 0 conflicts
- Three models evaluated: Base, SFT, GRPO
- Results saved to `data/eval_results/`
- Comparison table shows `first_tool_accuracy` (not `tool_accuracy`)
- "Best model" line uses actual best performer not last evaluated

### Step 9 — Verify server restart reload
```bash
# Start the server, make one request, kill it, restart it
uvicorn app.main:app &
sleep 2
curl -s -X POST http://localhost:8000/runs -H 'Content-Type: application/json' -d '{"question":"test"}' | python -m json.tool
kill %1
sleep 1
uvicorn app.main:app &
sleep 2
# Server should print: [startup] Reloaded N runs from disk into RUN_INDEX
kill %1
```

---

## Summary of files changed

| File | Changes |
|---|---|
| `sft_pipeline/planner_policy.py` | **NEW FILE** — canonical `normalize_action()` and `best_first_tools()` |
| `sft_pipeline/phase2_synthesize.py` | Add future-evidence leakage check in `_validate_trajectory()`, update auto_final prompt label |
| `sft_pipeline/phase3_format_sft.py` | Add rationale/evidence numeric mismatch check and replacement |
| `sft_pipeline/grpo_train.py` | Fix KeyError crash, import from planner_policy, add redundancy-trap prompts |
| `app/training/train.py` | Remove dead `compute_reward()` and `_best_first_tools()` |
| `sft_eval.py` | Import from planner_policy, rename metrics keys, add in-distribution warning |
| `unseen_eval.py` | Import from planner_policy, add service overlap check, fix best-model comparison |
| `app/tools/impl.py` | Fix service-name lookup, add contains sanitization, add seeded randomness |
| `app/agent/graph.py` | Fix evidence key to not overwrite repeated tool calls |
| `app/core/run_store.py` | Add `reload_from_disk()` function |
| `app/main.py` | Add startup event to call `reload_from_disk()` |
| `app/rag/retrieve.py` | Fix cosine similarity to catch vector length mismatch |
