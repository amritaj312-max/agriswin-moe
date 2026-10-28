"""
Print every finished run as one table -- the quantitative rows of Table II --
and check that the rows are comparable.

Rows may share a table only if they were scored on the same images and trained
with the same recipe, so this also compares every run's split fingerprints and
recipe settings. train.py stops a paper run from overriding the recipe, but a
recipe.yaml edited and committed between two runs can only be caught here.

Usage:
    python ml_pipeline/src/evaluation/summarize_results.py
    python ml_pipeline/src/evaluation/summarize_results.py --dev    # include dev_* runs
"""

import argparse
import glob
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
import paths  # noqa: E402

RESULTS_DIR = os.path.join(paths.REPO_ROOT, "results")
RECIPE = os.path.join(paths.REPO_ROOT, "ml_pipeline", "configs", "recipe.yaml")
# recipe keys that legitimately differ per run: a second seed, or fewer loader
# workers, does not change what is being compared
PER_RUN = {"seed", "num_workers"}


def duration(seconds):
    return f"{seconds / 3600:.1f}h" if seconds >= 3600 else f"{seconds / 60:.0f}m"


def main():
    parser = argparse.ArgumentParser(description="Summarise finished training runs")
    parser.add_argument("--dev", action="store_true", help="include dev_* runs")
    args = parser.parse_args()

    runs = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        if os.path.basename(path).startswith("dev_") and not args.dev:
            continue
        with open(path, encoding="utf-8") as fh:
            runs.append(json.load(fh))

    if not runs:
        print("No finished runs in results/ yet."
              + ("" if args.dev else "  (dev_* runs are hidden; --dev shows them)"))
        return

    header = (f"{'run':<26}{'params':>8}{'epochs':>9}{'test acc':>10}{'macro-F1':>10}"
              f"{'ECE':>8}{'NLL':>8}{'time':>7}   notes")
    print(header)
    print("-" * len(header))
    for r in runs:
        test, tr = r["test"], r["training"]
        notes = []
        if r["config"].get("subset"):
            notes.append(f"subset {r['config']['subset']}")
        if r["environment"].get("git_dirty"):
            notes.append("uncommitted code")
        if r["status"] == "early_stopped":
            notes.append("early stop")
        print(f"{r['run_name']:<26}{r['model']['params'] / 1e6:>7.2f}M"
              f"{tr['best_epoch']:>5}/{tr['epochs_run']:<3}"
              f"{test['acc']:>10.2%}{test['macro_f1']:>10.2%}{test['ece']:>8.2%}"
              f"{test['nll']:>8.4f}{duration(tr['train_time_s']):>7}   {', '.join(notes)}")

    print("\nepochs = best epoch / epochs run.  ECE and NLL: lower is better.")

    # subset runs differ by design, so they are left out of both checks
    full = [r for r in runs if not r["config"].get("subset")]
    if not full:
        return
    comparable = True

    groups = {}
    for r in full:
        groups.setdefault(json.dumps(r["data"]["fingerprint"], sort_keys=True), []).append(r["run_name"])
    if len(groups) == 1:
        print(f"Splits: all {len(full)} full-data run(s) were scored on identical images.")
    else:
        comparable = False
        print("WARNING: these groups of runs were scored on DIFFERENT images and must not "
              "share a table:")
        for names in groups.values():
            print("   ", ", ".join(names))

    with open(RECIPE, encoding="utf-8") as fh:
        recipe_keys = set(yaml.safe_load(fh)) - PER_RUN
    differing = sorted(k for k in recipe_keys
                       if len({json.dumps(r["config"].get(k)) for r in full}) > 1)
    if not differing:
        print(f"Recipe: all {len(full)} full-data run(s) were trained with the same recipe.")
    else:
        comparable = False
        print(f"WARNING: runs were trained with different recipes ({', '.join(differing)}), "
              f"so their rows compare recipes, not models:")
        for r in full:
            print(f"    {r['run_name']}: " + ", ".join(f"{k}={r['config'].get(k)}" for k in differing))

    if not comparable:
        sys.exit(1)


if __name__ == "__main__":
    main()
