"""
Prove this machine's images are byte-for-byte the ones the results were made on.

The split CSVs say WHICH images each split holds, by path, and every run's
fingerprint proves runs used the same paths. Neither proves the files behind
those paths are the same: a dataset downloaded on another machine, or on another
day, could hold different bytes under the same names. Training on Colab makes
that a real question -- PlantVillage's upstream repository last changed in
February 2026 -- so this checks the bytes themselves.

data/splits/plantvillage_sha256.json records, per split and per class, a digest
of every image's SHA-256. It was written once on the machine that trained the
Phase 2 baselines; any other machine must reproduce it exactly before its
results can share a table with those. The PlantDoc evaluation set is already
held by content (plantdoc.py), so it is checked in the same pass.

Usage:
    python ml_pipeline/src/data/verify_data.py            # check this machine (~1-2 min)
    python ml_pipeline/src/data/verify_data.py --write    # record the reference (once)
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
from classes import PLANTVILLAGE_CLASSES  # noqa: E402

REFERENCE = os.path.join(paths.SPLIT_DIR, "plantvillage_sha256.json")
SPLITS = ("train", "val", "test")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(paths.long_path(path), "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_of(lines):
    return hashlib.sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest()


def measure():
    """{split: {"images", "digest", "classes": {class: digest}}} for this machine."""
    root = paths.image_root("plantvillage")
    out = {}
    for split in SPLITS:
        with open(os.path.join(paths.SPLIT_DIR, f"plantvillage_{split}.csv"),
                  newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        start = time.time()
        # hashing is I/O-bound, so threads overlap the file reads
        with ThreadPoolExecutor(max_workers=8) as pool:
            hashes = list(pool.map(lambda r: file_sha256(os.path.join(root, r["path"])), rows))
        lines = [f"{r['path']}\t{r['class_idx']}\t{h}" for r, h in zip(rows, hashes)]
        by_class = {}
        for row, line in zip(rows, lines):
            by_class.setdefault(PLANTVILLAGE_CLASSES[int(row["class_idx"])], []).append(line)
        out[split] = {"images": len(rows), "digest": digest_of(lines),
                      "classes": {c: digest_of(v) for c, v in sorted(by_class.items())}}
        print(f"  {split:<6}{len(rows):>7} images hashed in {time.time() - start:.0f}s", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Check this machine's images against the reference")
    parser.add_argument("--write", action="store_true",
                        help="record this machine's images as the reference")
    args = parser.parse_args()

    print("PlantVillage:")
    measured = measure()

    if args.write:
        with open(REFERENCE, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"note": "SHA-256 of every split image, as trained on for the Phase 2 "
                               "baselines; see verify_data.py",
                       "splits": measured}, fh, indent=1)
            fh.write("\n")
        print(f"\nWrote {os.path.relpath(REFERENCE, paths.REPO_ROOT)}")
        return

    with open(REFERENCE, encoding="utf-8") as fh:
        reference = json.load(fh)["splits"]
    bad = []
    for split in SPLITS:
        ref, got = reference[split], measured[split]
        if ref["digest"] == got["digest"]:
            print(f"  {split:<6} identical to the reference")
            continue
        differing = [c for c in ref["classes"] if ref["classes"][c] != got["classes"].get(c)]
        bad.append(split)
        print(f"  {split:<6} DIFFERS in {len(differing)} class(es): {', '.join(differing[:6])}"
              + (" ..." if len(differing) > 6 else ""))

    print("\nPlantDoc evaluation set:")
    from plantdoc import PlantDocDataset, fingerprint
    try:
        ds = PlantDocDataset()
        print(f"  all {len(ds)} images present by content, fingerprint {fingerprint()}")
    except FileNotFoundError as exc:
        bad.append("plantdoc")
        print(f"  {exc}")

    if bad:
        raise SystemExit(f"\nNOT the same data as the reference ({', '.join(bad)}). Results made "
                         f"on this machine would not be comparable; do not train until fixed.")
    print("\nOK: every image is byte-for-byte the one the results were made on.")


if __name__ == "__main__":
    main()
