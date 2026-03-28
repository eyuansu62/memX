"""
Analysis 2: Memory Survival Curves (Kaplan-Meier).

Stratifies memories by whether they were eventually useful (productive retrieval
after 3+ epoch delay) and plots 4 survival curves:
  {MemRL, Belief-MemRL} × {future-useful, not-future-useful}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter


def load_events(path: str) -> List[Dict]:
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def build_memory_table(events: List[Dict]) -> pd.DataFrame:
    """Build per-memory survival data from event log."""
    creation: Dict[str, int] = {}
    deletion: Dict[str, int] = {}
    productive_epochs: Dict[str, List[int]] = {}

    max_epoch = 0
    for ev in events:
        epoch = int(ev["epoch"])
        max_epoch = max(max_epoch, epoch)
        mid = ev["memory_id"]

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

    rows = []
    for mid, birth in creation.items():
        death = deletion.get(mid, max_epoch + 1)
        survival_time = max(1, death - birth)

        prod_epochs = productive_epochs.get(mid, [])
        # "future useful" = productively retrieved ≥3 epochs after birth
        future_useful = any(e - birth >= 3 for e in prod_epochs)

        rows.append(
            {
                "memory_id": mid,
                "birth_epoch": birth,
                "death_epoch": death,
                "survival_time": survival_time,
                "observed": mid in deletion,  # True = died (event), False = censored
                "future_useful": future_useful,
            }
        )
    return pd.DataFrame(rows)


def run(
    memrl_path: str = "logs/memrl_alfworld.jsonl",
    belief_path: str = "logs/belief_memrl_alfworld.jsonl",
    out_dir: str = "results/diagnostics",
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    memrl_events = load_events(memrl_path)
    belief_events = load_events(belief_path)

    memrl_df = build_memory_table(memrl_events)
    belief_df = build_memory_table(belief_events)

    # Save raw per-memory data
    memrl_df["system"] = "MemRL"
    belief_df["system"] = "Belief-MemRL"
    combined = pd.concat([memrl_df, belief_df], ignore_index=True)
    csv_path = out / "memory_survival.csv"
    combined.to_csv(csv_path, index=False)

    # Kaplan-Meier plot
    fig, ax = plt.subplots(figsize=(9, 6))
    styles = {
        ("MemRL", True): ("#4C72B0", "solid", "MemRL · future-useful"),
        ("MemRL", False): ("#4C72B0", "dashed", "MemRL · not-useful"),
        ("Belief-MemRL", True): ("#DD8452", "solid", "Belief-MemRL · future-useful"),
        ("Belief-MemRL", False): ("#DD8452", "dashed", "Belief-MemRL · not-useful"),
    }
    for (system, useful), (color, ls, label) in styles.items():
        df = memrl_df if system == "MemRL" else belief_df
        sub = df[df["future_useful"] == useful]
        if sub.empty:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=sub["survival_time"],
            event_observed=sub["observed"],
            label=label,
        )
        kmf.plot_survival_function(ax=ax, color=color, linestyle=ls, ci_show=False)

    ax.set_xlabel("Epochs since memory creation")
    ax.set_ylabel("Fraction of memories surviving")
    ax.set_title("Memory Survival Curves (Kaplan-Meier)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    png_path = out / "memory_survival.png"
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
