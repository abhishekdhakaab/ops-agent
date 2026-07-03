"""Convert stored investigations into next-action SFT and GRPO examples."""

import json
from pathlib import Path
from typing import Any

def detect_repo_root() -> Path:
    """Locate the project root from script, module, or mounted-data contexts."""
    here = Path(__file__).resolve()
    candidates = [
        here.parent,
        here.parent.parent,
        here.parent.parent.parent,
        Path("/mnt/data"),
    ]
    for root in candidates:
        if (root / "data").exists():
            return root
    return here.parent


REPO_ROOT = detect_repo_root()
TRAJ = REPO_ROOT / "data" / "trajectories.jsonl"
OUT_SFT = REPO_ROOT / "data" / "trl_planner_sft.jsonl"
OUT_GRPO = REPO_ROOT / "data" / "trl_planner_grpo.jsonl"


PLANNER_SYSTEM = (
    "You are an ops planner. Choose the NEXT best tool action.\n"
    "Return ONLY valid JSON: {action,args,rationale}.\n"
    "Allowed actions: metrics, logs, deployments, final.\n"
    "Never invent evidence."
)

ALLOWED_ACTIONS = {"metrics", "logs", "deployments", "final"}
MEANINGFUL_TOOL_ACTIONS = {"metrics", "logs", "deployments"}

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read valid object rows while reporting isolated malformed records."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
            except json.JSONDecodeError as e:
                print(f"[JSON ERROR] line {i}: {e}")
    return rows

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one training object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def guess_expected_action(question: str) -> str:
    """Choose a weak first-tool label when a trajectory has no usable calls."""
    q = (question or "").lower()
    if any(tok in q for tok in ["deploy", "release", "rollback"]):
        return "deployments"
    if any(tok in q for tok in ["error", "5xx", "exception", "failed", "failure"]):
        return "logs"
    if any(tok in q for tok in ["slow", "latency", "timeout", "spike"]):
        return "metrics"
    return "metrics"

def normalize_tool_sequence(t: dict[str, Any]) -> list[str]:
    """Normalize legacy tool fields and discard unknown actions."""
    raw = t.get("tool_used")
    if raw is None:
        raw = t.get("tools_used", [])

    if isinstance(raw, str):
        raw = [raw]

    if not isinstance(raw, list):
        return []

    seq: list[str] = []
    for item in raw:
        if isinstance(item, str) and item in ALLOWED_ACTIONS:
            seq.append(item)
    return seq

def collapse_adjacent_duplicates(seq: list[str]) -> list[str]:
    """Collapse retry noise without erasing later intentional revisits."""
    out: list[str] = []
    for item in seq:
        if not out or out[-1] != item:
            out.append(item)
    return out

def meaningful_tool_sequence(t: dict[str, Any]) -> list[str]:
    """Return evidence-gathering calls suitable for supervision."""
    seq = normalize_tool_sequence(t)
    seq = collapse_adjacent_duplicates(seq)
    return [tool for tool in seq if tool in MEANINGFUL_TOOL_ACTIONS]

def extract_evidence_keys(t: dict[str, Any]) -> list[str]:
    """Return recognized evidence types from a stored trajectory."""
    evidence = t.get("evidence") or {}
    if not isinstance(evidence, dict):
        return []
    keys = [k for k in evidence.keys() if k in MEANINGFUL_TOOL_ACTIONS]
    return sorted(set(keys))

def build_state(
    question: str,
    service: str,
    used_tools_so_far: list[str],
) -> dict[str, Any]:
    """Build the leakage-free state shown before the next action."""
    return {
        "question": question,
        "service": service,
        "used_tools_so_far": used_tools_so_far,
        "available_tools": ["metrics", "logs", "deployments", "final"],
        "instruction": "Choose the NEXT action as JSON {action,args,rationale}. Avoid redundant repeated calls and choose final when enough evidence has already been collected.",
    }



def make_sft_row(
    *,
    state: dict[str, Any],
    action: str,
    service: str,
    rationale: str,
    label_source: str,
    trajectory_grounded_final: bool,
) -> dict[str, Any]:
    """Format one supervised chat example with provenance metadata."""
    assistant_text = json.dumps(
        {
            "action": action,
            "args": {"service": service} if service else {},
            "rationale": rationale,
        },
        ensure_ascii=False,
    )

    return {
        "messages": [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
            {"role": "assistant", "content": assistant_text},
        ],
        "metadata": {
            "label_source": label_source,
            "trajectory_grounded_final": trajectory_grounded_final,
        },
    }



def make_grpo_row(
    *,
    state: dict[str, Any],
    expected_action: str,
    trajectory_grounded_final: bool,
    label_source: str,
) -> dict[str, Any]:
    """Format one prompt and expected action for policy optimization."""
    return {
        "prompt": f"{PLANNER_SYSTEM}\n\nUSER:\n{json.dumps(state, ensure_ascii=False)}\n\nASSISTANT:\n",
        "expected_action": expected_action,
        "metadata": {
            "label_source": label_source,
            "trajectory_grounded_final": trajectory_grounded_final,
        },
    }



def main() -> None:
    """Convert all available trajectories and print label-source counts."""
    trajs = read_jsonl(TRAJ)
    if not trajs:
        print(f"No trajectories found at {TRAJ}")
        return

    sft_rows: list[dict[str, Any]] = []
    grpo_rows: list[dict[str, Any]] = []
    stats: dict[str, int] = {}

    for t in trajs:
        question = str(t.get("question", "") or "")
        service = str(t.get("service", "") or "")
        validation = t.get("validation") or {}
        grounded_final = bool(t.get("grounded_final", False))

        seq = meaningful_tool_sequence(t)

        # Each prefix teaches the next executed tool without exposing its result.
        for idx, next_tool in enumerate(seq):
            used_tools_so_far = seq[:idx]
            state = build_state(question, service, used_tools_so_far)
            sft_rows.append(
                make_sft_row(
                    state=state,
                    action=next_tool,
                    service=service,
                    rationale="Choose the next missing evidence source based on the investigation history so far.",
                    label_source="executed_step",
                    trajectory_grounded_final=grounded_final,
                )
            )
            grpo_rows.append(
                make_grpo_row(
                    state=state,
                    expected_action=next_tool,
                    trajectory_grounded_final=grounded_final,
                    label_source="executed_step",
                )
            )
            stats["executed_step"] = stats.get("executed_step", 0) + 1

        # Grounded runs supply an explicit stop example after their final tool.
        if grounded_final:
            state = build_state(question, service, seq)
            sft_rows.append(
                make_sft_row(
                    state=state,
                    action="final",
                    service=service,
                    rationale="Enough evidence has already been collected, so finalize instead of calling another tool.",
                    label_source="grounded_stop",
                    trajectory_grounded_final=grounded_final,
                )
            )
            grpo_rows.append(
                make_grpo_row(
                    state=state,
                    expected_action="final",
                    trajectory_grounded_final=grounded_final,
                    label_source="grounded_stop",
                )
            )
            stats["grounded_stop"] = stats.get("grounded_stop", 0) + 1

        # Ungrounded runs become correction examples when validation named a tool.
        required_next_action = validation.get("required_next_action")
        if (
            not grounded_final
            and isinstance(required_next_action, str)
            and required_next_action in MEANINGFUL_TOOL_ACTIONS
        ):
            state = build_state(question, service, seq)
            sft_rows.append(
                make_sft_row(
                    state=state,
                    action=required_next_action,
                    service=service,
                    rationale="The previous investigation was not grounded; choose the validator-requested missing evidence source next.",
                    label_source="validator_correction",
                    trajectory_grounded_final=grounded_final,
                )
            )
            grpo_rows.append(
                make_grpo_row(
                    state=state,
                    expected_action=required_next_action,
                    trajectory_grounded_final=grounded_final,
                    label_source="validator_correction",
                )
            )
            stats["validator_correction"] = stats.get("validator_correction", 0) + 1

        # Keep empty trajectories useful, but mark their heuristic provenance.
        if not seq and not grounded_final:
            fallback_action = guess_expected_action(question)
            state = build_state(question, service, [])
            sft_rows.append(
                make_sft_row(
                    state=state,
                    action=fallback_action,
                    service=service,
                    rationale="Fallback heuristic because no usable tool sequence was available.",
                    label_source="heuristic_fallback",
                    trajectory_grounded_final=grounded_final,
                )
            )
            grpo_rows.append(
                make_grpo_row(
                    state=state,
                    expected_action=fallback_action,
                    trajectory_grounded_final=grounded_final,
                    label_source="heuristic_fallback",
                )
            )
            stats["heuristic_fallback"] = stats.get("heuristic_fallback", 0) + 1

    write_jsonl(OUT_SFT, sft_rows)
    write_jsonl(OUT_GRPO, grpo_rows)

    print(f"Wrote SFT:  {OUT_SFT} ({len(sft_rows)} rows)")
    print(f"Wrote GRPO: {OUT_GRPO} ({len(grpo_rows)} rows)")
    print("Label source stats:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
