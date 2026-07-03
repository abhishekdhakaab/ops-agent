"""Scenario-backed metrics, logs, deployments, and ticket tools."""

from datetime import datetime, timedelta
from pathlib import Path
import json
import random

from .contracts import (
    MetricsIn, MetricsOut,
    LogsIn, LogsOut,
    DeploymentsIn, DeploymentsOut,
    TicketIn, TicketOut,
)
from .ticket_store import create_ticket


SCENARIOS_PATH = Path(__file__).resolve().parents[2] / "data" / "scenarios.json"

def _load_scenarios() -> dict:
    """Index scenarios by service, preferring the highest-severity duplicate."""
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
            # Equal or lower severity leaves the existing representative intact.
        else:
            index[svc] = s
    return index

# Tool calls are frequent, while the scenario fixture changes only during development.
_SCENARIO_INDEX = _load_scenarios()


DEFAULT_SCENARIO = {
    "service": "unknown",
    "title": "Unknown service",
    "root_cause": "unknown",
    "severity": "low",
    "metrics": {
        "has_latency_spike": False,
        "avg_latency_ms": 110.0,
        "p95_latency_ms": 240.0,
        "spike_start_minutes_ago": None,
    },
    "logs": {
        "error_count": 1,
        "error_messages": ["minor: GC pause 80ms"],
        "log_level": "WARN",
    },
    "deployments": {
        "has_recent_deploy": False,
        "deploy_minutes_ago": 2880,
        "deploy_summary": "Routine dependency update",
        "deploy_author": "ci-bot",
    },
}


def _get_scenario(service: str) -> dict:
    return _SCENARIO_INDEX.get(service.lower(), _SCENARIO_INDEX.get(service, DEFAULT_SCENARIO))


def reload_scenarios():
    """Reload scenarios from disk (useful after editing scenarios.json)."""
    global _SCENARIO_INDEX
    _SCENARIO_INDEX = _load_scenarios()


def _build_series(avg: float, spike: bool, service: str = "", points: int = 10) -> list:
    rng = random.Random(hash(service + "metrics_series"))
    if spike:
        normal = avg * 0.45
        return [round(normal + (avg - normal) * (i / points) + rng.uniform(-8, 8), 1)
                for i in range(points)]
    return [round(avg * (0.93 + 0.014 * i) + rng.uniform(-4, 4), 1)
            for i in range(points)]


def metrics_tool(inp: MetricsIn) -> MetricsOut:
    """Return scenario latency with the requested window applied."""
    sc = _get_scenario(inp.service)
    m = sc.get("metrics", DEFAULT_SCENARIO["metrics"])
    spike = m["has_latency_spike"]

    # A query window that predates the spike should still look healthy.
    spike_start = m.get("spike_start_minutes_ago")
    if spike and spike_start and inp.time_range_minutes < spike_start:
        spike = False

    avg = m["avg_latency_ms"] if spike else m["avg_latency_ms"] * 0.42
    p95 = m["p95_latency_ms"] if spike else m["p95_latency_ms"] * 0.38
    series = _build_series(avg, spike, service=inp.service)

    if spike:
        notes = f"Latency spike detected: avg {avg:.0f}ms, p95 {p95:.0f}ms. Began ~{spike_start}min ago."
    else:
        notes = f"Latency stable: avg {avg:.0f}ms, p95 {p95:.0f}ms. No anomalies in window."

    return MetricsOut(
        service=inp.service,
        time_range_minutes=inp.time_range_minutes,
        avg_latency_ms=round(avg, 1),
        p95_latency_ms=round(p95, 1),
        spike_detected=spike,
        window_series_ms=series,
        notes=notes,
    )


def logs_tool(inp: LogsIn) -> LogsOut:
    """Return aggregate errors and filtered representative log lines."""
    sc = _get_scenario(inp.service)
    lg = sc.get("logs", DEFAULT_SCENARIO["logs"])
    error_count = lg["error_count"]
    common = lg["error_messages"]
    level = lg.get("log_level", "ERROR")

    # Model-produced filters are normalized before they reach string matching.
    contains = inp.contains
    if contains is not None:
        if isinstance(contains, bool):
            contains = None  # Boolean filters have no meaningful substring form.
        elif isinstance(contains, list):
            contains = contains[0] if contains else None
        elif not isinstance(contains, str):
            contains = str(contains)
        if contains:
            contains = contains.lower()

    base = datetime.utcnow()
    samples = []
    for i in range(min(inp.limit or 20, 6)):
        ts = (base - timedelta(minutes=i * 2 + 1)).isoformat() + "Z"
        msg = common[i % len(common)]
        if contains and contains not in msg.lower():
            continue
        samples.append(f"{ts} {level} {inp.service}: {msg}")

    return LogsOut(
        service=inp.service,
        error_count=error_count,
        common_messages=common,
        samples=samples,
    )


def deployments_tool(inp: DeploymentsIn) -> DeploymentsOut:
    """Return one scenario deployment and one stable historical baseline."""
    sc = _get_scenario(inp.service)
    dep = sc.get("deployments", DEFAULT_SCENARIO["deployments"])
    now = datetime.utcnow()

    _dep_rng = random.Random(hash(inp.service + "deploy_version"))
    recent = {
        "id": f"dep_{abs(hash(inp.service + 'r')) % 100000:05d}",
        "time": (now - timedelta(minutes=dep["deploy_minutes_ago"])).isoformat() + "Z",
        "author": dep["deploy_author"],
        "summary": dep["deploy_summary"],
        "version": f"1.{_dep_rng.randint(10,20)}.{_dep_rng.randint(0,9)}",
        "minutes_ago": dep["deploy_minutes_ago"],
    }
    older = {
        "id": f"dep_{abs(hash(inp.service + 'o')) % 100000:05d}",
        "time": (now - timedelta(hours=18)).isoformat() + "Z",
        "author": "ci-bot",
        "summary": "Routine dependency update",
        "version": f"1.{_dep_rng.randint(8,12)}.0",
        "minutes_ago": 1080,
    }
    return DeploymentsOut(service=inp.service, recent=[recent, older])


def ticket_tool(inp: TicketIn) -> TicketOut:
    """Persist an action-node ticket through the shared ticket store."""
    out = create_ticket(inp.model_dump())
    return TicketOut(**out)
