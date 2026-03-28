"""
Analysis 3: Posterior vs Q-Value Calibration.

Measures which signal (Beta posterior mean vs Q-value) better predicts
future productive retrieval. Computed at multiple snapshot points.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def load_events(path: str) -> List[Dict]:
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def calibration_at_snapshot(events: List[Dict], snapshot_pct: float) -> Dict[str, float]:
    """AUROC and Pearson r for q_value and posterior_mean at a given snapshot pct."""
    max_epoch = max(int(ev["epoch"]) for ev in events) if events else 1
    snapshot_epoch = int(snapshot_pct * max_epoch)

    # Collect living memories at snapshot with their latest q_value / belief stats
    mem_state: Dict[str, Dict] = {}
    for ev in events:
        if int(ev["epoch"]) > snapshot_epoch:
            continue
        mid = ev["memory_id"]
        if ev["event_type"] == "write" and mid not in mem_state:
            mem_state[mid] = {
                "q_value": float(ev.get("q_value") or 0),
                "belief_alpha": float(ev.get("belief_alpha") or 1),
                "belief_beta": float(ev.get("belief_beta") or 1),
                "birth_epoch": int(ev["epoch"]),
            }
        elif ev["event_type"] in ("update",) and mid in mem_state:
            mem_state[mid]["q_value"] = float(ev.get("q_value") or mem_state[mid]["q_value"])
            mem_state[mid]["belief_alpha"] = float(ev.get("belief_alpha") or mem_state[mid]["belief_alpha"])
            mem_state[mid]["belief_beta"] = float(ev.get("belief_beta") or mem_state[mid]["belief_beta"])
        elif ev["event_type"] == "delete" and mid in mem_state:
            del mem_state[mid]

    # Label: productively retrieved after snapshot
    productive_after: Dict[str, bool] = {mid: False for mid in mem_state}
    for ev in events:
        if int(ev["epoch"]) <= snapshot_epoch:
            continue
        mid = ev["memory_id"]
        if (
            mid in productive_after
            and ev["event_type"] == "retrieve"
            and ev.get("retrieval_used_in_response") is True
            and ev.get("task_success") is True
        ):
            productive_after[mid] = True

    if not mem_state:
        return {"q_auroc": 0.5, "post_auroc": 0.5, "q_pearson": 0.0, "post_pearson": 0.0}

    mids = list(mem_state.keys())
    q_vals = np.array([mem_state[m]["q_value"] for m in mids])
    alphas = np.array([mem_state[m]["belief_alpha"] for m in mids])
    betas = np.array([mem_state[m]["belief_beta"] for m in mids])
    post_means = alphas / (alphas + betas + 1e-8)
    labels = np.array([int(productive_after[m]) for m in mids])

    if labels.sum() == 0 or labels.sum() == len(labels):
        return {"q_auroc": 0.5, "post_auroc": 0.5, "q_pearson": 0.0, "post_pearson": 0.0}

    def safe_auroc(scores, y):
        try:
            return float(roc_auc_score(y, scores))
        except Exception:
            return 0.5

    def safe_pearson(x, y):
        if x.std() < 1e-8:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    return {
        "q_auroc": safe_auroc(q_vals, labels),
        "post_auroc": safe_auroc(post_means, labels),
        "q_pearson": safe_pearson(q_vals, labels),
        "post_pearson": safe_pearson(post_means, labels),
        "_mids": mids,
        "_q_vals": q_vals.tolist(),
        "_post_means": post_means.tolist(),
        "_labels": labels.tolist(),
    }


def run(
    belief_path: str = "logs/belief_memrl_alfworld.jsonl",
    out_dir: str = "results/diagnostics",
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    events = load_events(belief_path)
    snapshots = [0.25, 0.50, 0.75]
    rows = []
    scatter_data_50 = None

    for pct in snapshots:
        res = calibration_at_snapshot(events, pct)
        rows.append(
            {
                "snapshot_pct": int(pct * 100),
                "q_value_auroc": round(res["q_auroc"], 4),
                "posterior_mean_auroc": round(res["post_auroc"], 4),
                "q_value_pearson_r": round(res["q_pearson"], 4),
                "posterior_pearson_r": round(res["post_pearson"], 4),
            }
        )
        if pct == 0.50:
            scatter_data_50 = res

    df = pd.DataFrame(rows)
    csv_path = out / "posterior_calibration.csv"
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))

    # Scatter plot at 50% snapshot
    if scatter_data_50 and "_q_vals" in scatter_data_50:
        q_vals = np.array(scatter_data_50["_q_vals"])
        post_means = np.array(scatter_data_50["_post_means"])
        labels = np.array(scatter_data_50["_labels"])
        jitter = np.random.default_rng(42).uniform(-0.04, 0.04, size=len(labels))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, x_vals, x_label, auroc in [
            (axes[0], q_vals, "Q-value", scatter_data_50["q_auroc"]),
            (axes[1], post_means, "Posterior Mean (α/(α+β))", scatter_data_50["post_auroc"]),
        ]:
            for val, color, marker in [(0, "#4C72B0", "o"), (1, "#DD8452", "^")]:
                mask = labels == val
                ax.scatter(
                    x_vals[mask],
                    labels[mask] + jitter[mask],
                    alpha=0.5,
                    s=20,
                    color=color,
                    marker=marker,
                    label=f"{'success' if val else 'failure'} (n={mask.sum()})",
                )
            ax.set_xlabel(x_label)
            ax.set_ylabel("Future productive retrieval (jittered)")
            ax.set_title(f"AUROC = {auroc:.3f}")
            ax.legend(fontsize=8)

        fig.suptitle("Calibration Scatter at 50% Snapshot (Belief-MemRL)")
        fig.tight_layout()
        png_path = out / "calibration_scatter.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {csv_path}, {png_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--belief", default="logs/belief_memrl_alfworld.jsonl")
    p.add_argument("--out_dir", default="results/diagnostics")
    args = p.parse_args()
    run(args.belief, args.out_dir)
