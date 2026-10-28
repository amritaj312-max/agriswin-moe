"""
Build frozen, stratified, duplicate-aware train/val/test splits for PlantVillage.

Why this is not just train_test_split:

PlantVillage contains many near-identical images (the same leaf photographed
repeatedly, and rotated/mirrored copies). A naive random split puts copies of
the same leaf in both train and test, which inflates test accuracy by several
points and is a known reason published PlantVillage numbers look unrealistically
high. This script groups near-duplicates first and keeps every group entirely
inside one split, so the test set measures generalisation rather than memory.

Duplicates are found with a 64-bit dHash:
  1. images with an identical hash are grouped directly (exact / near-exact), then
  2. remaining hashes are bucketed by 16-bit bands (LSH) and any pair within
     HAMMING_THRESHOLD bits is merged via union-find.

The output CSVs are committed to the repo so every experiment -- baselines,
ablations, the student model -- is evaluated on exactly the same images.

Usage:
    python ml_pipeline/src/data/build_splits.py
    python ml_pipeline/src/data/build_splits.py --no-dedup      # ablate the effect
    python ml_pipeline/src/data/build_splits.py --limit 2000    # quick smoke test
"""

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classes import CLASS_TO_IDX, PLANTVILLAGE_CLASSES, crop_of, expert_of  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
IMAGE_ROOT = os.path.join(REPO_ROOT, "data", "raw", "plantvillage", "raw", "color")
SPLIT_DIR = os.path.join(REPO_ROOT, "data", "splits")

SEED = 42
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
HAMMING_THRESHOLD = 6          # bits; <=6 of 64 is a near-duplicate
HASH_BANDS = 4                 # 4 bands x 16 bits for the LSH pass
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


# --- duplicate detection ----------------------------------------------------

def dhash(path, size=8):
    """64-bit difference hash. Returns None if the image cannot be read."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
            px = list(im.getdata())
    except Exception:
        return None

    bits = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits = (bits << 1) | (1 if px[base + col] > px[base + col + 1] else 0)
    return bits


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def group_duplicates(hashes):
    """Map each image index to a group id. Images with no duplicate get their own group."""
    uf = UnionFind(len(hashes))

    # pass 1: identical hashes
    by_hash = defaultdict(list)
    for i, h in enumerate(hashes):
        if h is not None:
            by_hash[h].append(i)
    for members in by_hash.values():
        for other in members[1:]:
            uf.union(members[0], other)

    # pass 2: near-duplicates among the distinct hashes only
    distinct = [(h, members[0]) for h, members in by_hash.items()]
    band_bits = 64 // HASH_BANDS
    mask = (1 << band_bits) - 1

    buckets = defaultdict(list)
    for h, rep in distinct:
        for b in range(HASH_BANDS):
            buckets[(b, (h >> (b * band_bits)) & mask)].append((h, rep))

    for members in buckets.values():
        if len(members) < 2 or len(members) > 200:   # skip degenerate buckets
            continue
        for i in range(len(members)):
            hi, ri = members[i]
            for j in range(i + 1, len(members)):
                hj, rj = members[j]
                if bin(hi ^ hj).count("1") <= HAMMING_THRESHOLD:
                    uf.union(ri, rj)

    return [uf.find(i) for i in range(len(hashes))]


# --- split assignment -------------------------------------------------------

def assign_splits(groups_by_class, rng):
    """
    Greedy, quota-driven assignment. Groups are shuffled, then each group goes to
    whichever split is currently furthest below its target share for that class.
    Keeping it per-class is what makes the result stratified.
    """
    assignment = {}

    for class_name, groups in groups_by_class.items():
        group_items = list(groups.items())
        rng.shuffle(group_items)
        # large groups first so they cannot overshoot a quota at the end
        group_items.sort(key=lambda kv: -len(kv[1]))

        total = sum(len(v) for _, v in group_items)
        target = {s: total * r for s, r in RATIOS.items()}
        current = {s: 0 for s in RATIOS}

        for group_id, paths in group_items:
            deficit = {s: (target[s] - current[s]) / max(target[s], 1) for s in RATIOS}
            split = max(deficit, key=deficit.get)
            assignment[(class_name, group_id)] = split
            current[split] += len(paths)

    return assignment


# --- main -------------------------------------------------------------------

def scan_images(limit=None):
    if not os.path.isdir(IMAGE_ROOT):
        print(f"ERROR: image root not found: {IMAGE_ROOT}")
        print("Run: python ml_pipeline/src/data/download.py --dataset plantvillage")
        sys.exit(1)

    records = []
    found_classes = sorted(d for d in os.listdir(IMAGE_ROOT)
                           if os.path.isdir(os.path.join(IMAGE_ROOT, d)))

    unknown = [c for c in found_classes if c not in CLASS_TO_IDX]
    missing = [c for c in PLANTVILLAGE_CLASSES if c not in found_classes]
    if unknown:
        print(f"WARNING: {len(unknown)} directories not in classes.py, e.g. {unknown[:3]}")
    if missing:
        print(f"WARNING: {len(missing)} expected classes missing, e.g. {missing[:3]}")

    for class_name in found_classes:
        if class_name not in CLASS_TO_IDX:
            continue
        class_dir = os.path.join(IMAGE_ROOT, class_name)
        files = sorted(f for f in os.listdir(class_dir) if f.lower().endswith(IMAGE_EXTS))
        if limit:
            files = files[:max(1, limit // len(found_classes))]
        for fname in files:
            records.append((os.path.join(class_dir, fname), class_name))

    return records


def main():
    parser = argparse.ArgumentParser(description="Build frozen PlantVillage splits")
    parser.add_argument("--no-dedup", action="store_true",
                        help="skip duplicate grouping (for measuring its effect)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap total images, for a fast smoke test")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    rng = random.Random(SEED)

    print("Scanning images...")
    records = scan_images(args.limit)
    print(f"  found {len(records)} images across {len(set(r[1] for r in records))} classes")

    if args.no_dedup:
        group_ids = list(range(len(records)))
        print("Duplicate grouping: SKIPPED (--no-dedup)")
    else:
        print(f"Hashing {len(records)} images ({args.workers} threads)...")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            hashes = list(pool.map(lambda r: dhash(r[0]), records))

        unreadable = sum(1 for h in hashes if h is None)
        if unreadable:
            print(f"  WARNING: {unreadable} images could not be read and are kept ungrouped")

        print("Grouping near-duplicates...")
        group_ids = group_duplicates(hashes)
        n_groups = len(set(group_ids))
        dup_images = len(records) - n_groups
        pct = 100.0 * dup_images / max(len(records), 1)
        print(f"  {len(records)} images -> {n_groups} groups "
              f"({dup_images} duplicates, {pct:.1f}%)")

    # bucket paths by (class, group)
    groups_by_class = defaultdict(lambda: defaultdict(list))
    for (path, class_name), gid in zip(records, group_ids):
        groups_by_class[class_name][gid].append(path)

    print("Assigning splits (stratified, group-aware)...")
    assignment = assign_splits(groups_by_class, rng)

    rows = {s: [] for s in RATIOS}
    for class_name, groups in groups_by_class.items():
        for gid, paths in groups.items():
            split = assignment[(class_name, gid)]
            for path in paths:
                rows[split].append({
                    "path": os.path.relpath(path, REPO_ROOT).replace("\\", "/"),
                    "class_name": class_name,
                    "class_idx": CLASS_TO_IDX[class_name],
                    "crop": crop_of(class_name),
                    "expert_idx": expert_of(class_name),
                    "group_id": gid,
                })

    os.makedirs(SPLIT_DIR, exist_ok=True)
    for split, split_rows in rows.items():
        split_rows.sort(key=lambda r: r["path"])
        out = os.path.join(SPLIT_DIR, f"plantvillage_{split}.csv")
        with open(out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(split_rows[0].keys()))
            writer.writeheader()
            writer.writerows(split_rows)
        print(f"  wrote {out}  ({len(split_rows)} images)")

    # --- report ---
    total = sum(len(v) for v in rows.values())
    print(f"\n{'Split':<8}{'Images':>9}{'Share':>9}")
    for split in ("train", "val", "test"):
        n = len(rows[split])
        print(f"{split:<8}{n:>9}{100.0 * n / total:>8.1f}%")

    # no group may appear in more than one split
    seen = {}
    leaks = 0
    for split, split_rows in rows.items():
        for r in split_rows:
            key = (r["class_name"], r["group_id"])
            if seen.setdefault(key, split) != split:
                leaks += 1
    print(f"\nGroup leakage across splits: {leaks} (must be 0)")

    # smallest class in the test set -- a class with ~0 test images breaks macro-F1
    per_class_test = defaultdict(int)
    for r in rows["test"]:
        per_class_test[r["class_name"]] += 1
    worst = sorted(per_class_test.items(), key=lambda kv: kv[1])[:3]
    print("Smallest test classes:", ", ".join(f"{c.split('___')[-1]}={n}" for c, n in worst))

    if leaks:
        sys.exit(1)
    print("\nSplits frozen. Commit data/splits/*.csv so every run uses the same images.")


if __name__ == "__main__":
    main()
