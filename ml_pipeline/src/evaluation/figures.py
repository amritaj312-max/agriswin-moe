"""
The paper's evaluation figures, drawn from results/eval/*.json.

Three figures, each answering a question PlantVillage accuracy cannot:

  reliability.png      is the model's confidence honest? A perfectly calibrated
                       model's bars sit on the diagonal. The gap between bar and
                       diagonal, weighted by how many images fall in each bin,
                       is exactly the ECE printed beside it.
  risk_coverage.png    what does abstaining buy? Accuracy against the fraction
                       of images the model answers, most confident first. A
                       model that knows what it does not know climbs steeply
                       to the left; a confidently wrong one stays flat.
  uncertainty.png      does uncertainty actually separate right from wrong, and
                       lab images from field images? Predictive entropy, split
                       by correctness, on PlantVillage against PlantDoc.

Figures are written to paper/figures/ as 300 dpi PNG and as PDF, which is what
LaTeX should include -- vector text stays sharp at any size.

Usage:
    python ml_pipeline/src/evaluation/figures.py
    python ml_pipeline/src/evaluation/figures.py --runs swinv2_tiny efficientnet_b0
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
import torch

matplotlib.use("Agg")                    # no display on a training box
import matplotlib.pyplot as plt          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
import paths  # noqa: E402
from uncertainty import risk_coverage  # noqa: E402

EVAL_DIR = os.path.join(paths.REPO_ROOT, "results", "eval")
FIGURE_DIR = os.path.join(paths.REPO_ROOT, "paper", "figures")
SPLIT_LABEL = {"plantvillage_test": "PlantVillage (test)", "plantdoc": "PlantDoc (field)"}


def load(runs):
    out = []
    for path in sorted(glob.glob(os.path.join(EVAL_DIR, "*.json"))):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if runs is None or data["run_name"] in runs:
            out.append(data)
    if not out:
        raise SystemExit(f"no evaluations in {EVAL_DIR}. Run:\n"
                         f"  python ml_pipeline/src/evaluation/evaluate.py")
    return out


def save(fig, name):
    os.makedirs(FIGURE_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGURE_DIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  paper/figures/{name}.png and .pdf")


def reliability_figure(evals):
    """One column per dataset, one row per run: confidence against accuracy."""
    splits = list(SPLIT_LABEL)
    fig, axes = plt.subplots(len(evals), len(splits),
                             figsize=(4.0 * len(splits), 3.4 * len(evals)), squeeze=False)
    for row, data in enumerate(evals):
        for col, split in enumerate(splits):
            ax = axes[row][col]
            entry = data["splits"][split]
            bins = entry["mc_dropout"]["reliability"]
            total = sum(b["count"] for b in bins)
            ax.plot([0, 1], [0, 1], "--", color="0.6", linewidth=1, label="perfect calibration")
            centres = [(b["lower"] + b["upper"]) / 2 for b in bins]
            heights = [b["accuracy"] if b["count"] else 0 for b in bins]
            weights = [b["count"] / total for b in bins]
            ax.bar(centres, heights, width=1 / len(bins) * 0.9, color="#3573b9",
                   edgecolor="white", linewidth=0.5, label="accuracy in bin")
            # how many images each bin holds -- a bar far off the diagonal
            # holding three images means nothing, and the ECE weights it that way
            twin = ax.twinx()
            twin.bar(centres, weights, width=1 / len(bins) * 0.9, color="#d1495b", alpha=0.25)
            twin.set_ylim(0, 1)
            twin.set_yticks([])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"{data['run_name']} -- {SPLIT_LABEL[split]}\n"
                         f"acc {entry['deterministic']['acc']:.1%}, "
                         f"ECE {entry['deterministic']['ece']:.1%}", fontsize=9)
            if col == 0:
                ax.set_ylabel("accuracy")
            if row == len(evals) - 1:
                ax.set_xlabel("confidence")
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Reliability: is the model's confidence honest?\n"
                 "(faint red: share of images in each confidence bin)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "reliability")


def risk_coverage_figure(evals):
    """Accuracy against coverage: what abstention buys on each dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), squeeze=False)
    for col, split in enumerate(SPLIT_LABEL):
        ax = axes[0][col]
        for data in evals:
            mc = data["splits"][split]["mc_dropout"]
            # rebuilt from the per-sample columns rather than stored: it is a
            # cumulative mean of them, and evaluate.py keeps results/ small
            per = mc["per_sample"]
            curve = risk_coverage(torch.tensor(per["correct"]).bool(),
                                  torch.tensor(per["confidence"], dtype=torch.float64))
            # thin the curve: 8,155 points draw slowly and look identical
            step = max(1, len(curve["coverage"]) // 500)
            aurc = mc["aurc_by_confidence"]
            # PlantVillage AURCs are ~1e-4; three decimals would print them all as 0.000
            aurc_text = f"{aurc:.1e}" if aurc < 0.01 else f"{aurc:.3f}"
            ax.plot(curve["coverage"][::step], curve["accuracy"][::step], linewidth=1.5,
                    label=f"{data['run_name']} (AURC {aurc_text})")
        ax.set_xlim(0, 1)
        ax.set_xlabel("coverage (fraction of images answered, most confident first)")
        ax.set_title(SPLIT_LABEL[split], fontsize=10)
        ax.grid(alpha=0.3)
        if col == 0:
            ax.set_ylabel("accuracy on the answered images")
        ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("Risk-coverage: what does letting the model abstain buy?", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "risk_coverage")


def uncertainty_figure(evals):
    """Predictive entropy, split by correctness, on both datasets."""
    fig, axes = plt.subplots(1, len(evals), figsize=(3.3 * len(evals), 3.8), squeeze=False)
    for col, data in enumerate(evals):
        ax = axes[0][col]
        groups, labels, colours = [], [], []
        for split, colour in (("plantvillage_test", "#3573b9"), ("plantdoc", "#d1495b")):
            per = data["splits"][split]["mc_dropout"]["per_sample"]
            right = [e for e, c in zip(per["total_entropy"], per["correct"]) if c]
            wrong = [e for e, c in zip(per["total_entropy"], per["correct"]) if not c]
            for name, values in (("right", right), ("wrong", wrong)):
                if values:
                    groups.append(values)
                    labels.append(f"{'PV' if split.startswith('plantv') else 'PD'}\n{name}")
                    colours.append(colour)
        parts = ax.violinplot(groups, showmedians=True, widths=0.8)
        for body, colour in zip(parts["bodies"], colours):
            body.set_facecolor(colour)
            body.set_alpha(0.55)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(data["run_name"], fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        if col == 0:
            ax.set_ylabel("predictive entropy (nats)")
    fig.suptitle("Does uncertainty rise where the model is wrong, and off-distribution?",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "uncertainty")


def main():
    parser = argparse.ArgumentParser(description="Draw the evaluation figures")
    parser.add_argument("--runs", nargs="+", help="run names (default: every evaluated run)")
    args = parser.parse_args()

    evals = load(set(args.runs) if args.runs else None)
    print(f"Drawing figures for {len(evals)} run(s):")
    reliability_figure(evals)
    risk_coverage_figure(evals)
    uncertainty_figure(evals)


if __name__ == "__main__":
    main()
