"""
Figure 4: mean client rule fidelity (top) and mean rule count per client
(bottom, log scale), IID vs non-IID, for every dataset and classifier.

    python scripts/make_figures.py                      # shipped results
    python scripts/make_figures.py --results results_rerun

Output: figures/fig_iid_vs_noniid.png (and .pdf)
"""
import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _common import REPO_ROOT  # noqa: E402
from fedneuroxai import config  # noqa: E402

COLOR_NONIID, COLOR_IID, GRID = "#2a78d6", "#eb6834", "#d8dadd"


def summary(root, ds, m):
    path = os.path.join(root, ds, f"{m}.json")
    if not os.path.isfile(path):
        return None
    d = json.load(open(path, encoding="utf-8"))
    fids = [c["fidelity_metrics"]["fidelity"] for c in d["per_client"] if c["fidelity_metrics"]]
    return {"fid": np.mean(fids) if fids else np.nan,
            "rules": np.mean([c["n_rules"] for c in d["per_client"]])}


def panel(ax, ds, data, key, ylabel, log=False):
    x = np.arange(len(config.MODEL_KEYS))
    w = 0.36
    for off, tag, color, label in ((-w / 2, "noniid", COLOR_NONIID, "Non-IID (Dirichlet)"),
                                   (w / 2, "iid", COLOR_IID, "IID (stratified)")):
        vals = [data[tag][ds][m][key] if data[tag][ds][m] else np.nan for m in config.MODEL_KEYS]
        ax.bar(x + off, vals, w, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([config.MODEL_LABELS[m] for m in config.MODEL_KEYS], fontsize=15, fontweight="bold")
    ax.set_title(config.DATASETS[ds]["label"], fontsize=18, fontweight="bold")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel, fontsize=15, fontweight="bold")
    ax.tick_params(axis="y", labelsize=14)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=config.RESULTS_DIR)
    ap.add_argument("--out", default=config.FIGURES_DIR)
    args = ap.parse_args()

    data = {tag: {ds: {m: summary(os.path.join(args.results, tag), ds, m) for m in config.MODEL_KEYS}
                  for ds in config.DATASETS} for tag in ("noniid", "iid")}

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.4))
    for col, ds in enumerate(config.DATASETS):
        panel(axes[0, col], ds, data, "fid", "Mean client fidelity" if col == 0 else "")
        axes[0, col].set_ylim(0, 1.05)
        panel(axes[1, col], ds, data, "rules", "Mean #rules per client (log)" if col == 0 else "", log=True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02),
               prop={"weight": "bold", "size": 16})
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(args.out, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(args.out, f"fig_iid_vs_noniid.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved -> {os.path.relpath(path, REPO_ROOT)}")


if __name__ == "__main__":
    main()
