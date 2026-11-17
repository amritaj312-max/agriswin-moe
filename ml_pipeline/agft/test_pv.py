"""PlantVillage end to end on the real data: what cells 6 and 7 do on Colab.

Loads and cleans the full dataset, hashes every photograph, makes the grouped
split, checks it for leakage, and draws the grouped CV folds -- timing each
step, because these are the ones that run before any GPU work starts.

Run from this folder:  python test_pv.py [path to an existing dataset cache]
"""
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eda  # noqa: E402
import pipeline as P  # noqa: E402
import stages  # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")


def main():
    root = tempfile.mkdtemp(prefix="agft_pv_")
    cache = sys.argv[1] if len(sys.argv) > 1 else ""
    cfg = P.Config(root=root, workers=0, hf_cache=cache)
    P.prepare_dirs(cfg)
    say = lambda m: print("   ", m.strip())  # noqa: E731

    print("1. Load and clean (cell 6)")
    tick = time.time()
    pv, classes = P.load_plantvillage(cfg, log=say)
    print(f"    {time.time() - tick:.0f}s")
    labels = [int(v) for v in pv["label"]]
    check("38 classes once the background class is dropped", len(classes) == 38,
          str(len(classes)))
    check("54,304 images", len(pv) == 54304, str(len(pv)))
    check("labels run 0..37 with none missing", sorted(set(labels)) == list(range(38)))
    check("the background class is gone", "Background_without_leaves" not in classes)
    row = pv[0]
    check("rows decode to an image and an integer label",
          hasattr(row["image"], "convert") and isinstance(row["label"], int),
          f"{type(row['image']).__name__} {row['image'].size}, label {row['label']}")

    print("\n2. Hash every photograph (cell 7)")
    tick = time.time()
    hashes = P.group_hashes(pv, cfg, log=say)
    print(f"    {time.time() - tick:.0f}s")
    check("one hash per image", len(hashes) == len(pv))
    again = P.group_hashes(pv, cfg, log=say)
    check("the second call reuses the cache", again == hashes)

    print("\n3. Near-duplicate groups, the split, the leakage check (cell 7)")
    tick = time.time()
    pairs = P.near_duplicate_pairs(hashes, max_bits=6, log=say)
    groups = P.group_ids(len(hashes), pairs, cfg.dup_bits)
    linked = len(groups) - len(set(groups))
    print(f"    {time.time() - tick:.0f}s")
    check("near-duplicates found at 4 bits", linked > 0,
          f"{len(set(groups))} groups for {len(groups)} images")
    splits = P.grouped_split(labels, groups, cfg, log=say)
    check("every image in exactly one split",
          sum(len(v) for v in splits.values()) == len(pv))
    tick = time.time()
    check("no two images in different splits within 4 bits",
          P.check_no_leakage(splits, hashes, cfg.dup_bits, log=say))
    print(f"    leakage check {time.time() - tick:.0f}s")
    shares = {k: len(v) / len(pv) for k, v in splits.items()}
    check("roughly 70/15/15", abs(shares["train"] - 0.70) < 0.02,
          ", ".join(f"{k} {v:.1%}" for k, v in shares.items()))
    again = P.grouped_split(labels, groups, cfg, log=say)
    check("the frozen split is reused", all(np.array_equal(again[k], splits[k]) for k in splits))

    print("\n4. EDA on the real data (cell 8)")
    pd_data, _ = P.load_plantdoc(classes, cfg, log=say)
    path = eda.run(labels, classes, groups, pairs, [int(v) for v in pd_data["label"]], cfg,
                   log=say)
    check("the EDA figure is written", os.path.isfile(path))
    report = P.read_json(os.path.join(cfg.paths["results"], "eda.json"))
    at4 = next(r for r in report["duplicate_sensitivity"] if r["bits"] == cfg.dup_bits)
    check("the threshold table matches the measurement", at4["images"] == 883,
          f"{at4['images']} images in {at4['groups']} groups at {cfg.dup_bits} bits")

    print("\n5. Grouped CV folds over the training split (run_arm)")
    tick = time.time()
    cfg5 = P.Config(root=root, folds=5)
    folds = stages.cv_folds(splits["train"], labels, groups, cfg5)
    print(f"    {time.time() - tick:.1f}s for {len(folds)} folds")
    shared = sum(len({groups[i] for i in a} & {groups[i] for i in b}) for a, b in folds)
    check("no near-duplicate group straddles a fold", shared == 0, f"{shared} shared")
    sizes = [len(b) for _, b in folds]
    check("folds are balanced in size", max(sizes) / min(sizes) < 1.1, str(sizes))

    print(f"\n{'PlantVillage path OK' if ok else 'PLANTVILLAGE CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":      # the guard Windows needs before any worker starts
    sys.exit(main())
