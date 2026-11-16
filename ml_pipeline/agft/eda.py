"""Exploratory data analysis, before a single model is trained.

Three questions a reviewer asks about any dataset paper, answered with
pictures and numbers rather than assertions: how imbalanced are the classes,
how badly do the two datasets differ, and how much does the training set
repeat itself.
"""

# %% CELL: eda
import collections
import json
import os

import numpy as np

from pipeline import CFG


def class_balance(labels, classes, log=print, top=8):
    counts = collections.Counter(labels)
    order = [c for c, _ in counts.most_common()]
    log(f"  {len(classes)} classes, {len(labels)} images")
    log(f"  largest  {classes[order[0]]}: {counts[order[0]]}")
    log(f"  smallest {classes[order[-1]]}: {counts[order[-1]]}")
    log(f"  imbalance ratio {counts[order[0]] / max(counts[order[-1]], 1):.1f}x")
    return counts


def duplication(hashes, log=print):
    counts = collections.Counter(hashes)
    repeated = {h: n for h, n in counts.items() if n > 1}
    log(f"  {len(counts)} distinct perceptual hashes for {len(hashes)} images")
    log(f"  {sum(repeated.values())} images share a hash with at least one other")
    log(f"  largest group of near-identical photographs: {max(counts.values())}")
    return counts


def figures(labels, classes, counts, hashes, pd_labels, cfg=CFG, log=print):
    """Class balance, duplication and the domain gap, saved to Drive."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    order = [c for c, _ in counts.most_common()]
    axes[0].bar(range(len(order)), [counts[c] for c in order], color="#4c72b0")
    axes[0].set_title("PlantVillage: images per class")
    axes[0].set_xlabel("class, commonest first")
    axes[0].set_ylabel("images")

    sizes = collections.Counter(collections.Counter(hashes).values())
    axes[1].bar(list(sizes), list(sizes.values()), color="#dd8452")
    axes[1].set_yscale("log")
    axes[1].set_title("Near-duplicates: group size")
    axes[1].set_xlabel("images sharing one perceptual hash")

    pv_share = np.bincount(labels, minlength=len(classes)) / len(labels)
    pd_share = np.bincount(pd_labels, minlength=len(classes)) / max(len(pd_labels), 1)
    axes[2].scatter(pv_share, pd_share, s=18, color="#55a868")
    lim = max(pv_share.max(), pd_share.max()) * 1.05
    axes[2].plot([0, lim], [0, lim], "--", color="grey", linewidth=1)
    axes[2].set_xlabel("share in PlantVillage (lab)")
    axes[2].set_ylabel("share in PlantDoc (field)")
    axes[2].set_title("The two datasets are differently composed")

    fig.tight_layout()
    dest = os.path.join(cfg.paths["figures"], "eda.png")
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    log(f"  saved {dest}")
    return dest


def run(labels, classes, hashes, pd_labels, cfg=CFG, log=print):
    log("Class balance")
    counts = class_balance(labels, classes, log)
    log("\nRepetition inside PlantVillage")
    duplication(hashes, log)
    log("\nFigures")
    path = figures(labels, classes, counts, hashes, pd_labels, cfg, log)
    with open(os.path.join(cfg.paths["results"], "eda.json"), "w", encoding="utf-8") as fh:
        json.dump({"classes": len(classes), "images": len(labels),
                   "distinct_hashes": len(set(hashes)),
                   "imbalance": max(counts.values()) / max(min(counts.values()), 1)}, fh,
                  indent=2)
    return path
