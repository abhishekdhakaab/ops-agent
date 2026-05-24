
import json
from pathlib import Path
from typing import Any

# helper function to get detailed of current directory 
def detect_repo_root() -> Path:
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

ALLOWED_ACTIONS = {"metrics", "logs", "deployments", "final"} # final is only used to stop the log
MEANINGFUL_TOOL_ACTIONS = {"metrics", "logs", "deployments"}

## used to read trajectory.jsonl file
def read_jsonl(path: Path) -> list[dict[str, Any]]:
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

## will be used to write back training_sft.jsonl and training_gpro.jsonl 
def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

## we use the words in question to estimate what action might need to be taken ( have couple of options to try out)
def guess_expected_action(question: str) -> str:
    q = (question or "").lower()
    if any(tok in q for tok in ["deploy", "release", "rollback"]):
        return "deployments"
    if any(tok in q for tok in ["error", "5xx", "exception", "failed", "failure"]):
        return "logs"
    if any(tok in q for tok in ["slow", "latency", "timeout", "spike"]):
        return "metrics"
    return "metrics"

## out all the tool_used we get those which are actually in the allowed_actions list (just for safety)
def normalize_tool_sequence(t: dict[str, Any]) -> list[str]:
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

# removes duplicate calls for same tool 
def collapse_adjacent_duplicates(seq: list[str]) -> list[str]:
    out: list[str] = []
    for item in seq:
        if not out or out[-1] != item:
            out.append(item)
    return out

# clears the tool calling track : get tools in tool list and remove duplicates
def meaningful_tool_sequence(t: dict[str, Any]) -> list[str]:
    seq = normalize_tool_sequence(t)
    seq = collapse_adjacent_duplicates(seq)
    return [tool for tool in seq if tool in MEANINGFUL_TOOL_ACTIONS]

## since evidence = { 'metrics': ... , 'logs':.... } so we store evidences for each tool call made
def extract_evidence_keys(t: dict[str, Any]) -> list[str]:
    evidence = t.get("evidence") or {}
    if not isinstance(evidence, dict):
        return []
    keys = [k for k in evidence.keys() if k in MEANINGFUL_TOOL_ACTIONS]
    return sorted(set(keys))

## create prompt for SFT training
def build_state(
    question: str,
    service: str,
    used_tools_so_far: list[str],
) -> dict[str, Any]:
    return {
        "question": question,
        "service": service,
        "used_tools_so_far": used_tools_so_far,
        "available_tools": ["metrics", "logs", "deployments", "final"],
        "instruction": "Choose the NEXT action as JSON {action,args,rationale}. Avoid redundant repeated calls and choose final when enough evidence has already been collected.",
    }



# take all the data and just create a training row (just structure) them
def make_sft_row(
    *,
    state: dict[str, Any],
    action: str,
    service: str,
    rationale: str,
    label_source: str,
    trajectory_grounded_final: bool,
) -> dict[str, Any]:
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



# takes all the data and create a single row for grpo training data
def make_grpo_row(
    *,
    state: dict[str, Any],
    expected_action: str,
    trajectory_grounded_final: bool,
    label_source: str,
) -> dict[str, Any]:
    return {
        "prompt": f"{PLANNER_SYSTEM}\n\nUSER:\n{json.dumps(state, ensure_ascii=False)}\n\nASSISTANT:\n",
        "expected_action": expected_action,
        "metadata": {
            "label_source": label_source,
            "trajectory_grounded_final": trajectory_grounded_final,
        },
    }



def main() -> None:
    trajs = read_jsonl(TRAJ) # get complete trajectory path ( we save trajectory with all the reasoning,tool call for entire request )
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

        seq = meaningful_tool_sequence(t) # remove tools which are not present in tool directory a
        ## most important since our goal is to make our Agent efficient enough to only make a single tool call once hence we will remove all the contigous occurance of that tool


        # 1) Step-level executed-tool supervision
        # Example:
        # seq = ["metrics", "logs"]
        # samples:
        # [] -> metrics
        # ["metrics"] -> logs
        for idx, next_tool in enumerate(seq): # seq is cleaned list of tool calls
            used_tools_so_far = seq[:idx]
            state = build_state(question, service, used_tools_so_far)
            ## so now we got a state that has the questoin,service we want to investigate, tools we have used so far and LLM system prompt for predicting next tool call

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

        # 2) If the run ended grounded, add an explicit stop/final sample
        # This teaches the planner to stop once enough evidence was collected.
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

        # 3) If the run ended ungrounded and validator suggested a next tool,
        # add a correction sample from the terminal state.
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

        # 4) If we learned nothing from this trajectory, add one weak fallback sample
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

















































# ## we are building the state before the tool was called, because if we want to teach next action then input should not already show that logs evidence collected
# def build_pre_action_state(t: dict[str, Any], target_action: str) -> dict[str, Any]:
#     # Best-effort approximation from the final accumulated state.
#     # We remove evidence of the target tool to approximate the state *before* that tool ran.
#     evidence_keys = extract_evidence_keys(t)
#     pre_keys = [k for k in evidence_keys if k != target_action]
#     return {
#         "question": t.get("question", ""),
#         "service": t.get("service", ""),
#         "evidence_keys": pre_keys,
#         "available_tools": ["metrics", "logs", "deployments", "final"],
#         "instruction": "Choose the NEXT action as JSON {action,args,rationale}.",
#     }


# def pick_label_and_state(t: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
#     """
#     Choose a leakage-free label/state pair.

#     Priority:
#     1) If the run ended ungrounded and validation requested a next tool, use that tool.
#        State: best-effort current state with that tool's evidence removed.
#     2) If the run ended grounded, use the last meaningful tool in the run.
#        State: best-effort pre-action state for that tool.
#     3) Otherwise, use the last meaningful tool in the run.
#        State: best-effort pre-action state for that tool.
#     4) If no meaningful tool exists, use a question heuristic from the initial state.

#     Note: true step-level state reconstruction is not possible from the current trajectory schema,
#     because only the final accumulated evidence is logged.
#     """
#     q = t.get("question", "")
#     validation = t.get("validation") or {}
#     grounded_final = bool(t.get("grounded_final", False))
#     seq = meaningful_tool_sequence(t)

#     required_next_action = validation.get("required_next_action")
#     if (not grounded_final) and required_next_action in MEANINGFUL_TOOL_ACTIONS:
#         return required_next_action, build_pre_action_state(t, required_next_action), "validator_required_next_action"

#     if grounded_final and seq:
#         target = seq[-1]
#         return target, build_pre_action_state(t, target), "last_meaningful_tool_from_grounded_run"

#     if seq:
#         target = seq[-1]
#         return target, build_pre_action_state(t, target), "last_meaningful_tool_from_run"

#     target = guess_expected_action(q)
#     return target, build_initial_state(t), "question_heuristic"


# def rationale_for_source(label_source: str) -> str:
#     mapping = {
#         "validator_required_next_action": "Based on the current evidence, gather the missing evidence type requested by validation.",
#         "last_meaningful_tool_from_grounded_run": "This tool likely provided the decisive evidence needed for a grounded conclusion.",
#         "last_meaningful_tool_from_run": "This is the strongest available tool signal in the recorded run.",
#         "question_heuristic": "Use the default first investigation step suggested by the question.",
#     }
#     return mapping[label_source]


# def main() -> None:
#     trajs = read_jsonl(TRAJ)
#     if not trajs:
#         print(f"No trajectories found at {TRAJ}")
#         return

#     sft_rows: list[dict[str, Any]] = []
#     grpo_rows: list[dict[str, Any]] = []
#     stats: dict[str, int] = {}

#     for t in trajs:
#         service = t.get("service", "")
#         target_action, state, label_source = pick_label_and_state(t)
#         stats[label_source] = stats.get(label_source, 0) + 1

#         assistant_text = json.dumps(
#             {
#                 "action": target_action,
#                 "args": {"service": service} if service else {},
#                 "rationale": rationale_for_source(label_source),
#             },
#             ensure_ascii=False,
#         )

#         sft_rows.append(
#             {
#                 "messages": [
#                     {"role": "system", "content": PLANNER_SYSTEM},
#                     {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
#                     {"role": "assistant", "content": assistant_text},
#                 ],
#                 "metadata": {
#                     "label_source": label_source,
#                     "grounded_final": bool(t.get("grounded_final", False)),
#                     "tool_sequence": meaningful_tool_sequence(t),
#                     "evidence_keys_final": extract_evidence_keys(t),
#                 },
#             }
#         )

#         grpo_rows.append(
#             {
#                 "prompt": f"{PLANNER_SYSTEM}\n\nUSER:\n{json.dumps(state, ensure_ascii=False)}\n\nASSISTANT:\n",
#                 "expected_action": target_action,
#                 "metadata": {
#                     "label_source": label_source,
#                 },
#             }
#         )

#     write_jsonl(OUT_SFT, sft_rows)
#     write_jsonl(OUT_GRPO, grpo_rows)

#     print(f"Wrote SFT:  {OUT_SFT} ({len(sft_rows)} rows)")
#     print(f"Wrote GRPO: {OUT_GRPO} ({len(grpo_rows)} rows)")
#     print("Label source stats:", json.dumps(stats, ensure_ascii=False, indent=2))


# if __name__ == "__main__":
#     main()
