"""
Analysis 1: Cross-Epoch Retrieval Rate by Temporal Distance.

Measures how far back (in epochs) a productively-used memory was created.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


BUCKETS = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 9999)]
BUCKET_LABELS = ["0", "1-2", "3-5", "6-10", "11-20", "20+"]


def bucket_index(dist: int) -> int:
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= dist <= hi:
            return i
    return len(BUCKETS) - 1


def load_events(path: str) -> List[Dict]:
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def analyze(events: List[Dict]) -> List[float]:
    """Return normalized retrieval rates per distance bucket."""
    # Build memory_id → creation epoch (first write event)
    creation_epoch: Dict[str, int] = {}
    for ev in events:
        if ev["event_type"] == "write" and ev["memory_id"] not in creation_epoch:
            creation_epoch[ev["memory_id"]] = int(ev["epoch"])

    # Filter: retrieve events where used_in_response==True and task_success==True
    productive = [
        ev for ev in events
        if ev["event_type"] == "retrieve"
        and ev.get("retrieval_used_in_response") is True
        and ev.get("task_success") is True
        and ev["memory_id"] in creation_epoch
    ]

    counts = [0] * len(BUCKETS)
    for ev in productive:
        dist = max(0, int(ev["epoch"]) - creation_epoch[ev["memory_id"]])
        counts[bucket_index(dist)] += 1

    total = sum(counts) or 1
    return [c / total for c in counts]


def run(
    memrl_path: str = "logs/memrl_alfworld.jsonl",
    belief_path: str = "logs/belief_memrl_alfworld.jsonl",
    out_dir: str = "results/diagnostics",
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    memrl_events = load_events(memrl_path)
    belief_events = load_events(belief_path)

    memrl_rates = analyze(memrl_events)
    belief_rates = analyze(belief_events)

    df = pd.DataFrame(
        {
            "distance_bucket": BUCKET_LABELS,
            "memrl_rate": memrl_rates,
            "belief_memrl_rate": belief_rates,
        }
    )
    csv_path = out / "cross_epoch_retrieval.csv"
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))

    # Bar chart
    x = range(len(BUCKET_LABELS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], memrl_rates, width, label="MemRL", color="#4C72B0")
    ax.bar([i + width / 2 for i in x], belief_rates, width, label="Belief-MemRL", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(BUCKET_LABELS)
    ax.set_xlabel("Temporal Distance (epochs since memory creation)")
    ax.set_ylabel("Fraction of Productive Retrievals")
    ax.set_title("Cross-Epoch Retrieval Rate by Temporal Distance")
    ax.legend()
    fig.tight_layout()
    png_path = out / "cross_epoch_retrieval.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {csv_path}, {png_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--memrl", default="logs/memrl_alfworld.jsonl")
    p.add_argument("--belief", default="logs/belief_memrl_alfworld.jsonl")
    p.add_argument("--out_dir", default="results/diagnostics")
    args = p.parse_args()
    run(args.memrl, args.belief, args.out_dir)
