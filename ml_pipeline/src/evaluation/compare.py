"""
Compare evaluated runs -- and refuse to call a difference real when it is not.

The baselines finished within five test images of each other on PlantVillage,
and three EfficientNet-B0 seeds spread 4.2 points on PlantDoc. So the question
this file answers is not "which number is larger" but "is this difference
bigger than what changing the seed alone produces".

Two tools, and both are needed:

  the seed floor   the spread across runs of one model that differ only in
                   seed. Any gap between two *models* smaller than this says
                   nothing: re-running either one could erase it.
  McNemar          an exact paired test on the same images (Dietterich, 1998).
                   It asks only about the images the two models disagree on,
                   which is far more sensitive than comparing two accuracies --
                   but it tests one pair of runs, so on its own it will happily
                   report a seed change as significant. Read it together with
                   the seed floor, never instead of it.

Usage:
    python ml_pipeline/src/evaluation/compare.py
    python ml_pipeline/src/evaluation/compare.py --split plantdoc
"""

import argparse
import glob
import json
import os
import statistics as st
import sys

from scipy.stats import binomtest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
import paths  # noqa: E402
from classes import PLANTVILLAGE_CLASSES  # noqa: E402

EVAL_DIR = os.path.join(paths.REPO_ROOT, "results", "eval")
CORN_PAIR = (PLANTVILLAGE_CLASSES.index("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"),
             PLANTVILLAGE_CLASSES.index("Corn_(maize)___Northern_Leaf_Blight"))


def load():
    evals = []
    for path in sorted(glob.glob(os.path.join(EVAL_DIR, "*.json"))):
        with open(path, encoding="utf-8") as fh:
            evals.append(json.load(fh))
    if not evals:
        raise SystemExit(f"no evaluations in {EVAL_DIR}. Run:\n"
                         f"  python ml_pipeline/src/evaluation/evaluate.py")
    return evals


def correctness(data, split):
    entry = data["splits"][split]
    return [p == t for p, t in zip(entry["predictions"], entry["labels"])]


def corn_pair_errors(data, split):
    """Confusions between the two corn leaf spots -- the hardest pair, and the
    only place the baselines genuinely differ on PlantVillage."""
    gls, nlb = CORN_PAIR
    entry = data["splits"][split]
    return sum(1 for p, t in zip(entry["predictions"], entry["labels"])
               if (t, p) in ((gls, nlb), (nlb, gls)))


def seed_floor(evals, split, metric):
    """{model: (mean, sd, spread, n)} over runs that differ only in seed."""
    groups = {}
    for data in evals:
        groups.setdefault(data["model"], []).append(metric(data, split))
    return {model: (st.mean(v), st.stdev(v) if len(v) > 1 else None, max(v) - min(v), len(v))
            for model, v in groups.items() if len(v) > 1}


def main():
    parser = argparse.ArgumentParser(description="Compare evaluated runs honestly")
    parser.add_argument("--split", default="plantdoc", choices=["plantdoc", "plantvillage_test"],
                        help="which split to compare on (default plantdoc)")
    args = parser.parse_args()
    evals = load()
    split = args.split
    nice = "PlantDoc" if split == "plantdoc" else "PlantVillage test"
    n_images = evals[0]["splits"][split]["images"]

    print(f"{len(evals)} evaluated run(s) on {nice} ({n_images:,} images)\n")
    header = (f"{'run':<24}{'model':<26}{'seed':>5}{'acc':>8}{'ECE':>8}"
              f"{'AURC':>8}{'acc@50%':>9}{'corn pair':>11}")
    print(header)
    print("-" * len(header))
    for data in sorted(evals, key=lambda d: (d["model"], d["seed"])):
        entry = data["splits"][split]
        mc = entry["mc_dropout"]
        print(f"{data['run_name']:<24}{data['model']:<26}{data['seed']:>5}"
              f"{entry['deterministic']['acc']:>8.2%}{entry['deterministic']['ece']:>8.2%}"
              f"{mc['aurc_by_confidence']:>8.3f}{mc['acc_at_coverage']['0.5']:>9.2%}"
              f"{corn_pair_errors(data, split):>11}")

    # --- the seed floor ------------------------------------------------------
    print(f"\nSeed floor on {nice} -- runs of one model differing only in seed:")
    floors = {}
    for label, metric, fmt in (
            ("accuracy", lambda d, s: d["splits"][s]["deterministic"]["acc"], "{:.2%}"),
            ("ECE", lambda d, s: d["splits"][s]["deterministic"]["ece"], "{:.2%}"),
            ("corn-pair errors", lambda d, s: float(corn_pair_errors(d, s)), "{:.1f}")):
        floor = seed_floor(evals, split, metric)
        if not floor:
            continue
        floors[label] = floor
        for model, (mean, sd, spread, n) in floor.items():
            sd_text = fmt.format(sd) if sd is not None else "n/a"
            print(f"  {label:<18}{model:<26}{n} seeds: mean {fmt.format(mean)}, "
                  f"sd {sd_text}, spread {fmt.format(spread)}")
    if not floors:
        print("  none: no model here has been run with more than one seed.")
        print("  Without that, nothing below can be separated from seed noise.")

    acc_floor = max((spread for _, _, spread, _ in floors.get("accuracy", {}).values()),
                    default=None)
    if acc_floor is not None:
        print(f"\n  => Treat any accuracy gap below {acc_floor:.2%} between two models on "
              f"{nice}\n     as indistinguishable from re-running one of them with another seed.")

    # --- paired comparisons --------------------------------------------------
    print(f"\nMcNemar, exact, paired per image on {nice}:")
    print(f"  {'pair':<50}{'only A':>8}{'only B':>8}{'p':>10}   reading")
    for i, a in enumerate(evals):
        for b in evals[i + 1:]:
            ca, cb = correctness(a, split), correctness(b, split)
            only_a = sum(1 for x, y in zip(ca, cb) if x and not y)
            only_b = sum(1 for x, y in zip(ca, cb) if y and not x)
            n = only_a + only_b
            p = binomtest(min(only_a, only_b), n, 0.5).pvalue if n else 1.0
            same_model = a["model"] == b["model"]
            gap = abs(sum(ca) - sum(cb)) / len(ca)
            if same_model:
                reading = "SEED CHANGE ONLY -- any 'significance' here is the noise floor"
            elif acc_floor is not None and gap < acc_floor:
                reading = f"gap {gap:.2%} is inside the seed floor -- not a model difference"
            elif p < 0.05:
                reading = f"gap {gap:.2%} exceeds the seed floor and p < 0.05"
            else:
                reading = "not significant"
            print(f"  {a['run_name'] + ' vs ' + b['run_name']:<50}{only_a:>8}{only_b:>8}"
                  f"{p:>10.3g}   {reading}")

    print("\nMcNemar counts only the images the two runs disagree on, so it is far more")
    print("sensitive than comparing two accuracies -- sensitive enough to call a seed")
    print("change significant. A difference is real only if it also clears the seed floor.")


if __name__ == "__main__":
    main()
