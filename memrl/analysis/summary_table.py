"""
Summary table combining main experiment results with diagnostic metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_events(path: str) -> List[Dict]:
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def last_acc(events: List[Dict]) -> float:
    """Last-epoch training success rate from update events."""
    if not events:
        return 0.0
    max_epoch = max(int(ev["epoch"]) for ev in events)
    updates = [ev for ev in events if ev["event_type"] == "update" and int(ev["epoch"]) == max_epoch]
    if not updates:
        return 0.0
    successes = [ev.get("task_success") for ev in updates if ev.get("task_success") is not None]
    return sum(successes) / len(successes) if successes else 0.0


def csr(events: List[Dict]) -> float:
    """Cumulative success rate: fraction of unique task_descs ever succeeded."""
    successes = set()
    all_tasks = set()
    for ev in events:
        if ev["event_type"] == "write":
            td = ev.get("memory_text", "")[:200]
            all_tasks.add(td)
            if ev.get("task_success"):
                successes.add(td)
    return len(successes) / max(len(all_tasks), 1)


def avg_cross_epoch_dist(events: List[Dict]) -> float:
    """Mean temporal distance of productive retrievals."""
    creation: Dict[str, int] = {}
    for ev in events:
        if ev["event_type"] == "write" and ev["memory_id"] not in creation:
            creation[ev["memory_id"]] = int(ev["epoch"])

    dists = []
    for ev in events:
        if (
            ev["event_type"] == "retrieve"
            and ev.get("retrieval_used_in_response") is True
            and ev.get("task_success") is True
            and ev["memory_id"] in creation
        ):
            dists.append(max(0, int(ev["epoch"]) - creation[ev["memory_id"]]))
    return sum(dists) / len(dists) if dists else 0.0


def future_useful_survival_median(events: List[Dict]) -> float:
    """Median survival time of future-useful memories."""
    creation: Dict[str, int] = {}
    deletion: Dict[str, int] = {}
    productive_epochs: Dict[str, List[int]] = {}
    max_epoch = max((int(ev["epoch"]) for ev in events), default=1)

    for ev in events:
        mid = ev["memory_id"]
        epoch = int(ev["epoch"])
        if ev["event_type"] == "write" and mid not in creation:
            creation[mid] = epoch
        elif ev["event_type"] == "delete":
            deletion[mid] = epoch
        elif (
            ev["event_type"] == "retrieve"
            and ev.get("retrieval_used_in_response") is True
            and ev.get("task_success") is True
        ):
            productive_epochs.setdefault(mid, []).append(epoch)

    times = []
    for mid, birth in creation.items():
        prod = productive_epochs.get(mid, [])
        if any(e - birth >= 3 for e in prod):
            death = deletion.get(mid, max_epoch + 1)
            times.append(max(1, death - birth))

    if not times:
        return float("nan")
    times_sorted = sorted(times)
    n = len(times_sorted)
    return float(times_sorted[n // 2])


def consistency_score(events: List[Dict]) -> float:
    """Measure whether the memory state is consistent after updates.

    After an override/delete/expire event on a memory, check if subsequent
    retrievals for the same task pattern correctly reflect the updated state
    (i.e., do not return a stale or deleted memory).

    Score = fraction of post-update retrievals that return the updated (not stale) memory.
    """
    # Track deleted / expired memories
    removed: set = set()
    for ev in events:
        if ev["event_type"] in ("delete", "expire", "evict"):
            removed.add(ev["memory_id"])

    # Check retrievals after removals
    if not removed:
        return 1.0  # no removals => trivially consistent

    total_retrievals = 0
    stale_retrievals = 0
    for ev in events:
        if ev["event_type"] == "retrieve":
            total_retrievals += 1
            if ev["memory_id"] in removed:
                stale_retrievals += 1

    if total_retrievals == 0:
        return 1.0
    return 1.0 - (stale_retrievals / total_retrievals)


def intervention_fidelity(events: List[Dict]) -> float:
    """After explicit interventions (refine with external trigger, or override with
    external source), measure how quickly the correction propagates.

    Score = fraction of interventions where the very next retrieval for a similar
    task returns the corrected memory.
    """
    interventions = [
        ev for ev in events if ev["event_type"] == "intervention"
    ]
    if not interventions:
        return float("nan")

    corrected_ids = {ev["memory_id"] for ev in interventions}
    # Check the first retrieval after each intervention
    fidelity_hits = 0
    for intv in interventions:
        intv_ts = intv.get("timestamp", "")
        # Find next retrieve event after this intervention
        for ev in events:
            if (
                ev["event_type"] == "retrieve"
                and ev.get("timestamp", "") > intv_ts
            ):
                if ev["memory_id"] in corrected_ids:
                    fidelity_hits += 1
                break  # only check the first retrieval after

    return fidelity_hits / len(interventions) if interventions else 0.0


def conflict_resolution_rate(events: List[Dict], window: int = 3) -> float:
    """Fraction of detected conflicts that are resolved within K epochs.

    A conflict is detected when a memory's belief_conflict increases.
    It is resolved when the memory is subsequently refined, deleted, or expired.
    """
    # Track memories with conflict events (high conflict rate at update time)
    conflicts: Dict[str, int] = {}  # memory_id -> epoch of first conflict
    resolutions: set = set()

    for ev in events:
        mid = ev.get("memory_id", "")
        epoch = int(ev.get("epoch", 0))

        # Detect conflict: update event where conflict counter is rising
        if ev["event_type"] == "update":
            bc = float(ev.get("belief_conflict", 0.0))
            if bc > 0 and mid not in conflicts:
                conflicts[mid] = epoch

        # Detect resolution
        if ev["event_type"] in ("refine", "delete", "expire", "evict"):
            if mid in conflicts and epoch - conflicts[mid] <= window:
                resolutions.add(mid)

    if not conflicts:
        return float("nan")
    return len(resolutions) / len(conflicts)


def eviction_rate(events: List[Dict]) -> float:
    """Fraction of memories that were evicted (per total memories created)."""
    created = set(ev["memory_id"] for ev in events if ev["event_type"] == "write")
    evicted = set(ev["memory_id"] for ev in events if ev["event_type"] == "evict")
    if not created:
        return 0.0
    return len(evicted) / len(created)


def budget_utilization_over_time(events: List[Dict]) -> List[Dict]:
    """Track memory count over epochs for budget-utility analysis."""
    alive: set = set()
    epoch_counts: Dict[int, int] = {}

    for ev in events:
        mid = ev.get("memory_id", "")
        epoch = int(ev.get("epoch", 0))

        if ev["event_type"] == "write":
            alive.add(mid)
        elif ev["event_type"] in ("delete", "evict"):
            alive.discard(mid)

        epoch_counts[epoch] = len(alive)

    return [{"epoch": e, "memory_count": c} for e, c in sorted(epoch_counts.items())]


def run(
    memrl_path: str = "logs/memrl_alfworld.jsonl",
    belief_path: str = "logs/belief_memrl_alfworld.jsonl",
    out_dir: str = "results/diagnostics",
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    memrl_ev = load_events(memrl_path)
    belief_ev = load_events(belief_path)

    rows = []
    for name, ev in [("MemRL", memrl_ev), ("Belief-MemRL", belief_ev)]:
        cs = consistency_score(ev)
        ifid = intervention_fidelity(ev)
        crr = conflict_resolution_rate(ev)
        er = eviction_rate(ev)
        rows.append(
            {
                "System": name,
                "Last Acc": f"{last_acc(ev):.1%}",
                "CSR": f"{csr(ev):.1%}",
                "Avg Cross-Epoch Dist": f"{avg_cross_epoch_dist(ev):.2f}",
                "Survival (med)": f"{future_useful_survival_median(ev):.1f}",
                "Consistency": f"{cs:.1%}" if not (cs != cs) else "N/A",
                "Intervention Fid.": f"{ifid:.1%}" if not (ifid != ifid) else "N/A",
                "Conflict Res.": f"{crr:.1%}" if not (crr != crr) else "N/A",
                "Eviction Rate": f"{er:.1%}",
            }
        )

    df = pd.DataFrame(rows)

    md_path = out / "summary_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Diagnostic Summary Table\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(df.to_markdown(index=False))
    print(f"\nSaved: {md_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--memrl", default="logs/memrl_alfworld.jsonl")
    p.add_argument("--belief", default="logs/belief_memrl_alfworld.jsonl")
    p.add_argument("--out_dir", default="results/diagnostics")
    args = p.parse_args()
    run(args.memrl, args.belief, args.out_dir)
