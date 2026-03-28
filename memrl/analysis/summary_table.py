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
        rows.append(
            {
                "System": name,
                "Last Acc": f"{last_acc(ev):.1%}",
                "CSR": f"{csr(ev):.1%}",
                "Avg Cross-Epoch Dist": f"{avg_cross_epoch_dist(ev):.2f}",
                "Future-Useful Survival (median epochs)": f"{future_useful_survival_median(ev):.1f}",
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
