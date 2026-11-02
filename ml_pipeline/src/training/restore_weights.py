"""
Put trained weights downloaded from a GitHub release back where training left them.

checkpoints/ is git-ignored, so trained weights are backed up separately, as
assets of a GitHub release (the Phase 2 baselines: tag weights-phase2). A
release asset uploaded from the browser had to stay under 10 MB, so most files
were uploaded in parts: swinv2_tiny.pt as swinv2_tiny.pt.part01 to .part12,
and so on. This joins the parts, checks every file against the release's
SHA256SUMS.txt, and writes each as checkpoints/<run>/best.pt -- the one file
the later phases (evaluation, distillation) start from.

A file whose checksum does not match is never written. An existing best.pt is
left alone unless it is identical, or --force says to replace it.

Usage:
    python ml_pipeline/src/training/restore_weights.py <folder with the downloads>
    python ml_pipeline/src/training/restore_weights.py <folder> --force

Download SHA256SUMS.txt and the parts of whichever runs you need; runs whose
files are missing are skipped, so restoring only swinv2_tiny is fine.
"""

import argparse
import glob
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
import paths  # noqa: E402

CHECKPOINT_DIR = os.path.join(paths.REPO_ROOT, "checkpoints")


def read_sums(path):
    """{file name: sha256} from sha256sum output ('<hash> *<name>' or '<hash>  <name>')."""
    sums = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                digest, name = line.split(maxsplit=1)
                sums[name.strip().lstrip("*")] = digest.lower()
    return sums


def assemble(folder, name):
    """(bytes, None) whole or joined from parts; (None, None) if nothing is
    there; (None, reason) if parts are missing from the middle. A missing last
    part cannot be seen from the numbering, but the checksum catches it."""
    whole = os.path.join(folder, name)
    if os.path.isfile(whole):
        with open(whole, "rb") as fh:
            return fh.read(), None
    # zero-padded part numbers, so name order is part order
    parts = sorted(glob.glob(glob.escape(whole) + ".part[0-9][0-9]"))
    if not parts:
        return None, None
    numbers = [int(p[-2:]) for p in parts]
    missing = sorted(set(range(1, max(numbers) + 1)) - set(numbers))
    if missing:
        return None, "part " + ", ".join(f"{n:02d}" for n in missing) + " not downloaded"
    chunks = []
    for p in parts:
        with open(p, "rb") as fh:
            chunks.append(fh.read())
    return b"".join(chunks), None


def main():
    parser = argparse.ArgumentParser(description="Restore checkpoints/<run>/best.pt from release downloads")
    parser.add_argument("folder", help="folder holding SHA256SUMS.txt and the downloaded files or parts")
    parser.add_argument("--force", action="store_true", help="replace a best.pt that differs")
    args = parser.parse_args()

    sums_path = os.path.join(args.folder, "SHA256SUMS.txt")
    if not os.path.isfile(sums_path):
        raise SystemExit(f"{sums_path} not found: download SHA256SUMS.txt from the release too")

    restored = failed = 0
    for name, expected in read_sums(sums_path).items():
        run = name[:-len(".pt")] if name.endswith(".pt") else name
        data, problem = assemble(args.folder, name)
        if problem:
            print(f"  {run:<24} {problem} -- not written")
            failed += 1
            continue
        if data is None:
            print(f"  {run:<24} not downloaded -- skipped")
            continue
        if hashlib.sha256(data).hexdigest() != expected:
            print(f"  {run:<24} CHECKSUM MISMATCH -- not written; download it again")
            failed += 1
            continue

        target = os.path.join(CHECKPOINT_DIR, run, "best.pt")
        if os.path.isfile(target):
            with open(target, "rb") as fh:
                if hashlib.sha256(fh.read()).hexdigest() == expected:
                    print(f"  {run:<24} already in place")
                    continue
            if not args.force:
                print(f"  {run:<24} {target} exists and differs -- kept; --force replaces it")
                continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target + ".tmp", "wb") as fh:
            fh.write(data)
        os.replace(target + ".tmp", target)
        print(f"  {run:<24} restored, checksum verified ({len(data) / 1e6:.1f} MB)")
        restored += 1

    print(f"\n{restored} restored, {failed} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
