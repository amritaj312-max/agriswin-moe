"""Exploratory data analysis, before a single model is trained.

Three questions a reviewer asks about any dataset paper, answered with
pictures and numbers rather than assertions: how imbalanced are the classes,
how badly do the two datasets differ, and how much does the training set
repeat itself.
"""

# %% CELL: eda
import collections
import os

import numpy as np

from pipeline import CFG, group_ids, write_json


def class_balance(labels, classes, log=print, top=8):
    counts = collections.Counter(labels)
    order = [c for c, _ in counts.most_common()]
    log(f"  {len(classes)} classes, {len(labels)} images")
    log(f"  largest  {classes[order[0]]}: {counts[order[0]]}")
    log(f"  smallest {classes[order[-1]]}: {counts[order[-1]]}")
    log(f"  imbalance ratio {counts[order[0]] / max(counts[order[-1]], 1):.1f}x")
    return counts


def duplication(groups, pairs, bits, log=print):
    """How much the dataset repeats itself, and how that depends on the
    threshold -- the table that justifies the threshold in the paper."""
    counts = collections.Counter(groups)
    linked = [n for n in counts.values() if n > 1]
    log(f"  at <= {bits} bits: {sum(linked)} images in {len(linked)} near-duplicate "
        f"groups, the largest of {max(counts.values())}")
    sensitivity = []
    log("  sensitivity -- bits allowed to differ, and what each threshold links:")
    for t in range(max([d for _, _, d in pairs], default=0) + 1):
        sizes = [n for n in collections.Counter(group_ids(len(groups), pairs, t)).values()
                 if n > 1]
        sensitivity.append({"bits": t, "groups": len(sizes), "images": sum(sizes),
                            "largest": max(sizes, default=1)})
        log(f"    <= {t}: {sum(sizes):6d} images in {len(sizes):5d} groups, "
            f"largest {max(sizes, default=1)}{'   <- used' if t == bits else ''}")
    return counts, sensitivity


def figures(labels, classes, counts, groups, pd_labels, cfg=CFG, log=print):
    """Class balance, duplication and the domain gap, saved to Drive.

    Drawn on a standalone Figure rather than through pyplot: switching
    pyplot's backend here would switch it for the whole notebook, and every
    later plt.show() would silently display nothing.
    """
    from matplotlib.figure import Figure

    fig = Figure(figsize=(16, 4.2))
    axes = fig.subplots(1, 3)
    order = [c for c, _ in counts.most_common()]
    axes[0].bar(range(len(order)), [counts[c] for c in order], color="#4c72b0")
    axes[0].set_title("PlantVillage: images per class")
    axes[0].set_xlabel("class, commonest first")
    axes[0].set_ylabel("images")

    sizes = collections.Counter(collections.Counter(groups).values())
    axes[1].bar(list(sizes), list(sizes.values()), color="#dd8452")
    axes[1].set_yscale("log")
    axes[1].set_title("Near-duplicates: group size")
    axes[1].set_xlabel(f"images per group (hashes within {cfg.dup_bits} bits)")

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
    log(f"  saved {dest}")
    return dest


def run(labels, classes, groups, pairs, pd_labels, cfg=CFG, log=print):
    log("Class balance")
    counts = class_balance(labels, classes, log)
    log("\nRepetition inside PlantVillage")
    _, sensitivity = duplication(groups, pairs, cfg.dup_bits, log)
    log("\nFigures")
    path = figures(labels, classes, counts, groups, pd_labels, cfg, log)
    write_json({"classes": len(classes), "images": len(labels),
                "groups": len(set(groups)), "dup_bits": cfg.dup_bits,
                "duplicate_sensitivity": sensitivity,
                "imbalance": max(counts.values()) / max(min(counts.values()), 1)},
               os.path.join(cfg.paths["results"], "eda.json"))
    return path
