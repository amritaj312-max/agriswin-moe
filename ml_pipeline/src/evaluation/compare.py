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


def configuration(data):
    """What a run is, apart from its seed: the model, and the GPU it trained on.

    The GPU belongs in it. "Seed 42" on another machine is not a repeat of the
    same run -- a different GPU, PyTorch build and worker count change the
    augmentation stream, so it is a fresh draw. Pooling machines would call
    that spread seed noise and inflate the floor every comparison is judged
    against.
    """
    gpu = (data["checks"].get("trained_on_gpu") or "unknown GPU")
    return data["model"], gpu.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")


def pair_floor(floor, a, b):
    """The seed noise a comparison of these two runs has to clear, and whether
    it had to be borrowed from a model that is not one of them.

    A gap is only as trustworthy as the noisier of the two things compared, so
    this is the larger of the two configurations' own spreads -- never the
    widest spread in the table. Judging one model's ablation by another
    model's floor is how a real effect gets thrown away: on PlantDoc three
    EfficientNet-B0 seeds spread 4.16 points while three Swin-V2-T seeds
    spread 0.27, so EfficientNet's floor would bury any ST-DAM gain worth
    having.

    How much of the pair the floor covers is returned with it, as a marker: a
    floor measured on one side cannot say whether the other side is noisier,
    and one borrowed from a third model says less still. Both are usable, and
    both are marked rather than passed off as the real thing.
    """
    own = [floor[c][2] for c in (configuration(a), configuration(b)) if c in floor]
    if len(own) == 2:
        return max(own), ""
    if len(own) == 1:
        return own[0], "~"           # only one of the two has been run twice
    return max((spread for _, _, spread, _ in floor.values()), default=None), "*"


def seed_floor(evals, split, metric):
    """{(model, gpu): (mean, sd, spread, n)} over runs differing only in seed."""
    groups = {}
    for data in evals:
        groups.setdefault(configuration(data), []).append(metric(data, split))
    return {key: (st.mean(v), st.stdev(v) if len(v) > 1 else None, max(v) - min(v), len(v))
            for key, v in groups.items() if len(v) > 1}


def main():
    parser = argparse.ArgumentParser(description="Compare evaluated runs honestly")
    parser.add_argument("--split", default="plantdoc", choices=["plantdoc", "plantvillage_test"],
                        help="which split to compare on (default plantdoc)")
    args = parser.parse_args()
    evals = load()
    split = args.split
    nice = "PlantDoc" if split == "plantdoc" else "PlantVillage test"
    n_images = evals[0]["splits"][split]["images"]

    # runs scored on different images cannot share a table, however close the
    # numbers look -- the same rule summarize_results.py applies to PlantVillage
    key = "plantdoc_fingerprint" if split == "plantdoc" else None
    if key:
        groups = {}
        for data in evals:
            groups.setdefault(data.get(key, "missing (evaluated before the frozen set)"),
                              []).append(data["run_name"])
        if len(groups) > 1:
            print(f"WARNING: these runs were scored on DIFFERENT {nice} images and must not "
                  f"be compared -- re-run evaluate.py on all of them:")
            for fp, names in groups.items():
                print(f"    {fp}: {', '.join(names)}")
            sys.exit(1)

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
        for (model, gpu), (mean, sd, spread, n) in sorted(floor.items()):
            sd_text = fmt.format(sd) if sd is not None else "n/a"
            print(f"  {label:<18}{model + ' on ' + gpu:<44}{n} seeds: mean {fmt.format(mean)}, "
                  f"sd {sd_text}, spread {fmt.format(spread)}")
    if not floors:
        print("  none: no model here has been run with more than one seed.")
        print("  Without that, nothing below can be separated from seed noise.")

    accuracy_floors = floors.get("accuracy", {})
    if accuracy_floors:
        print(f"\n  => A gap on {nice} must clear the floor of the two configurations being")
        print("     compared -- the larger of their own spreads, not the widest in this")
        print("     table. Each reading below names the floor it used, with ~ when only one")
        print("     side has been run twice and * when the number is borrowed from a model")
        print("     that is neither of them.")

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
            same_config = configuration(a) == configuration(b)
            cross_machine = a["model"] == b["model"] and not same_config
            gap = abs(sum(ca) - sum(cb)) / len(ca)
            limit, mark = pair_floor(accuracy_floors, a, b)
            if same_config:
                reading = "SEED CHANGE ONLY -- any 'significance' here is the noise floor"
            elif cross_machine:
                reading = "SAME MODEL, DIFFERENT GPU -- not a model comparison"
            elif limit is None:
                reading = f"gap {gap:.2%}, but no seed floor exists yet -- cannot be judged"
            elif gap < limit:
                reading = f"gap {gap:.2%} is inside the {limit:.2%}{mark} floor -- not a model difference"
            elif p < 0.05:
                reading = f"gap {gap:.2%} clears the {limit:.2%}{mark} floor and p < 0.05"
            else:
                reading = f"gap {gap:.2%} clears the {limit:.2%}{mark} floor, but p = {p:.3g}"
            print(f"  {a['run_name'] + ' vs ' + b['run_name']:<50}{only_a:>8}{only_b:>8}"
                  f"{p:>10.3g}   {reading}")

    print("\nMcNemar counts only the images the two runs disagree on, so it is far more")
    print("sensitive than comparing two accuracies -- sensitive enough to call a seed")
    print("change significant. A difference is real only if it also clears the seed floor.")


if __name__ == "__main__":
    main()
