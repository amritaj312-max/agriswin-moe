"""
Download the datasets into the data root (see paths.py).

PlantVillage is the primary training set; PlantDoc is the real-world field set
used only for the cross-dataset generalisation study (never for training).

The PlantVillage repo ships three copies of every image -- colour, greyscale and
segmented -- so a plain clone costs roughly 6 GB for the 2 GB that is actually
wanted. This fetches only the colour tree, using a blobless partial clone plus
sparse-checkout, which also skips the blobs for the other two copies rather than
downloading and then discarding them.

Each dataset goes into its own directory and nothing is executed from those
directories -- they are image data, treated as untrusted.

Usage:
    python ml_pipeline/src/data/download.py --dataset all
    python ml_pipeline/src/data/download.py --dataset plantvillage --data-root D:/agriswin-data
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

DATASETS = {
    "plantvillage": {
        "url": "https://github.com/spMohanty/PlantVillage-Dataset.git",
        "dest": "plantvillage",
        # only this subtree is checked out; the greyscale and segmented copies
        # are never fetched
        "sparse": ["raw/color"],
        "expected_classes": 38,
        "approx_gb": 2.0,
    },
    "plantdoc": {
        "url": "https://github.com/pratikkayal/PlantDoc-Dataset.git",
        "dest": "plantdoc",
        "sparse": None,          # small enough to take whole
        "expected_classes": 28,   # repo ships 28 train dirs; the paper says 27
        "approx_gb": 0.7,
    },
}


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd).returncode == 0


# Characters NTFS forbids in a filename. PlantDoc's images are web-scraped and
# some names keep the query string ("IMG_1629.JPG?1507122477.jpg"), so a normal
# checkout aborts on Windows at the first one and leaves the tree empty.
WINDOWS_ILLEGAL = '<>:"|?*'

# written only once a checkout has finished; its absence means "resume"
COMPLETE_MARKER = ".agriswin_complete"


def _safe_component(name, max_len=120):
    """Make one path component legal on Windows.

    Besides the illegal characters, PlantDoc has filenames long enough to blow
    past MAX_PATH (some are whole sentences scraped from a page). Over-long
    names are truncated with a hash suffix so they stay unique.
    """
    cleaned = "".join("_" if ch in WINDOWS_ILLEGAL else ch for ch in name)
    cleaned = cleaned.rstrip(" .") or "_"

    if len(cleaned) > max_len:
        stem, ext = os.path.splitext(cleaned)
        digest = hashlib.sha1(cleaned.encode("utf-8", "replace")).hexdigest()[:8]
        keep = max(1, max_len - len(ext) - 9)
        cleaned = f"{stem[:keep]}_{digest}{ext}"
    return cleaned


def _long_path(path):
    """Windows extended-length form, which lifts the 260-character limit."""
    if os.name == "nt":
        path = os.path.abspath(path)
        if not path.startswith("\\\\?\\"):
            return "\\\\?\\" + path
    return path


def sanitized_checkout(name, dest):
    """Materialise HEAD manually, renaming paths Windows cannot represent.

    Used when `git checkout` refuses the tree. Blobs are streamed from a single
    `git cat-file --batch` process rather than one process per file.
    """
    listing = subprocess.run(
        ["git", "ls-tree", "-r", "HEAD", "--format=%(objectname) %(path)"],
        cwd=dest, capture_output=True, text=True,
    )
    if listing.returncode != 0:
        print(f"[{name}] could not list the tree: {listing.stderr.strip()[:200]}")
        return 0

    entries = []
    for line in listing.stdout.splitlines():
        if " " not in line:
            continue
        sha, path = line.split(" ", 1)
        entries.append((sha, path))

    if not entries:
        return 0

    print(f"[{name}] checkout failed; writing {len(entries)} files with sanitised names")

    written = renamed = 0
    batch = subprocess.Popen(["git", "cat-file", "--batch"], cwd=dest,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for sha, path in entries:
            batch.stdin.write((sha + "\n").encode())
            batch.stdin.flush()

            header = batch.stdout.readline().decode(errors="replace").split()
            if len(header) < 3:
                continue
            size = int(header[2])
            blob = batch.stdout.read(size)
            batch.stdout.read(1)          # trailing newline

            parts = path.split("/")
            safe_parts = [_safe_component(p) for p in parts]
            if safe_parts != parts:
                renamed += 1

            target = _long_path(os.path.join(dest, *safe_parts))
            # resumable: a re-run skips files already written in full
            if os.path.exists(target) and os.path.getsize(target) == size:
                written += 1
                continue

            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as fh:
                fh.write(blob)
            written += 1
    finally:
        batch.stdin.close()
        batch.wait()

    print(f"[{name}] wrote {written} files ({renamed} renamed to remove {WINDOWS_ILLEGAL})")
    return written


def clone(name, spec):
    dest = os.path.join(paths.raw_dir(), spec["dest"])

    marker = os.path.join(dest, COMPLETE_MARKER)

    if os.path.isdir(os.path.join(dest, ".git")):
        # Objects are present. A previous run may still have died partway
        # through the checkout, so only the marker proves it finished.
        if os.path.isfile(marker):
            print(f"[{name}] already complete at {dest} -- skipping")
        else:
            print(f"[{name}] incomplete checkout -- resuming")
            if sanitized_checkout(name, dest) > 0:
                open(marker, "w").close()
        return dest

    if os.path.isdir(dest) and os.listdir(dest):
        print(f"[{name}] {dest} exists but is not a git clone; leaving it alone.")
        return dest

    os.makedirs(paths.raw_dir(), exist_ok=True)
    print(f"[{name}] cloning into {dest}  (~{spec['approx_gb']} GB)")

    if spec["sparse"]:
        # blobless + sparse: fetch only the blobs inside the sparse paths
        ok = run(["git", "clone", "--depth", "1", "--filter=blob:none",
                  "--sparse", spec["url"], dest])
        if ok:
            print(f"[{name}] restricting checkout to {spec['sparse']}")
            ok = run(["git", "sparse-checkout", "set"] + spec["sparse"], cwd=dest)
    else:
        ok = run(["git", "clone", "--depth", "1", spec["url"], dest])

    if not ok:
        # The objects may have arrived and only the checkout failed -- that is
        # the normal outcome for PlantDoc on Windows. Recover from the object
        # store instead of throwing away a good download.
        if os.path.isdir(os.path.join(dest, ".git")):
            if sanitized_checkout(name, dest) > 0:
                open(marker, "w").close()
                print(f"[{name}] recovered via sanitised checkout")
                return dest

        print(f"[{name}] FAILED -- see git output above")
        # a partial clone left behind would be mistaken for a good one later
        if os.path.isdir(dest):
            print(f"[{name}] removing incomplete clone at {dest}")
            shutil.rmtree(dest, ignore_errors=True)
        return None

    open(marker, "w").close()
    print(f"[{name}] clone complete")
    return dest


def verify(name, spec, dest):
    """Check the image tree exists and holds roughly the expected classes."""
    if dest is None:
        return False

    image_dir = paths.image_root(name)
    if not os.path.isdir(image_dir):
        print(f"[{name}] WARNING: expected image directory not found: {image_dir}")
        return False

    class_dirs = sorted(d for d in os.listdir(image_dir)
                        if os.path.isdir(os.path.join(image_dir, d)))
    n_images = sum(
        len([f for f in os.listdir(os.path.join(image_dir, d))
             if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        for d in class_dirs
    )

    print(f"[{name}] image dir : {image_dir}")
    print(f"[{name}] classes   : {len(class_dirs)} (expected {spec['expected_classes']})")
    print(f"[{name}] images    : {n_images}")

    if len(class_dirs) != spec["expected_classes"]:
        print(f"[{name}] WARNING: class count differs from expected.")
    return n_images > 0


def main():
    parser = argparse.ArgumentParser(description="Download plant disease datasets")
    parser.add_argument("--dataset", choices=["plantvillage", "plantdoc", "all"],
                        default="all")
    parser.add_argument("--data-root", help="where to store datasets; persisted for later runs")
    args = parser.parse_args()

    if args.data_root:
        root = paths.set_data_root(args.data_root)
        print(f"data root set to {root} (saved to ml_pipeline/configs/data_root.txt)")

    if shutil.which("git") is None:
        print("ERROR: git is not on PATH.")
        sys.exit(1)

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    needed = sum(DATASETS[n]["approx_gb"] for n in names)
    free = paths.free_gb()

    print(f"data root : {paths.data_root()}")
    print(f"free space: {free:.1f} GB   needed: ~{needed:.1f} GB")
    if free < needed * 1.3:
        print(f"\nERROR: not enough free space. Need ~{needed * 1.3:.1f} GB "
              f"(datasets plus git overhead), have {free:.1f} GB.")
        print("Re-run with --data-root pointing at a drive with more room, e.g.:")
        print("  python ml_pipeline/src/data/download.py --data-root D:/agriswin-data")
        sys.exit(1)

    ok = True
    for name in names:
        spec = DATASETS[name]
        print(f"\n{'=' * 60}\n{name.upper()}\n{'=' * 60}")
        ok = verify(name, spec, clone(name, spec)) and ok

    print(f"\n{'=' * 60}")
    if ok:
        print("All requested datasets are in place.")
        print("Next: python ml_pipeline/src/data/build_splits.py")
    else:
        print("One or more datasets need attention -- see warnings above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
