"""
Download the datasets into data/raw/.

PlantVillage is the primary training set; PlantDoc is the real-world field set
used only for the cross-dataset generalisation study (never for training).

Each dataset is cloned into its own directory under data/raw/ and nothing is
executed from those directories -- they are image data, treated as untrusted.

Usage:
    python ml_pipeline/src/data/download.py --dataset plantvillage
    python ml_pipeline/src/data/download.py --dataset plantdoc
    python ml_pipeline/src/data/download.py --dataset all

Both repositories are large (PlantVillage ~2 GB, PlantDoc ~700 MB); a shallow
clone is used to avoid pulling the full history.
"""

import argparse
import os
import shutil
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")

DATASETS = {
    "plantvillage": {
        "url": "https://github.com/spMohanty/PlantVillage-Dataset.git",
        "dest": "plantvillage",
        # the colour images live here; the repo also ships greyscale/segmented copies
        "image_subdir": os.path.join("raw", "color"),
        "expected_classes": 38,
    },
    "plantdoc": {
        "url": "https://github.com/pratikkayal/PlantDoc-Dataset.git",
        "dest": "plantdoc",
        "image_subdir": "train",
        "expected_classes": 27,
    },
}


def have_git():
    return shutil.which("git") is not None


def clone(name, spec):
    dest = os.path.join(RAW_DIR, spec["dest"])

    if os.path.isdir(os.path.join(dest, ".git")):
        print(f"[{name}] already present at {dest} -- skipping clone")
        return dest

    if os.path.isdir(dest) and os.listdir(dest):
        print(f"[{name}] {dest} exists but is not a git clone; leaving it alone.")
        print(f"[{name}] Delete it manually if you want a fresh download.")
        return dest

    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"[{name}] cloning {spec['url']}")
    print(f"[{name}] -> {dest}   (this is a large download, be patient)")

    result = subprocess.run(
        ["git", "clone", "--depth", "1", spec["url"], dest],
        cwd=RAW_DIR,
    )
    if result.returncode != 0:
        print(f"[{name}] FAILED (git exit {result.returncode})")
        return None

    print(f"[{name}] clone complete")
    return dest


def verify(name, spec, dest):
    """Sanity-check the download: does the image directory exist and hold the expected classes?"""
    if dest is None:
        return False

    image_dir = os.path.join(dest, spec["image_subdir"])
    if not os.path.isdir(image_dir):
        print(f"[{name}] WARNING: expected image directory not found: {image_dir}")
        print(f"[{name}] Inspect {dest} and update image_subdir in this script.")
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
    return True


def main():
    parser = argparse.ArgumentParser(description="Download plant disease datasets")
    parser.add_argument("--dataset", choices=["plantvillage", "plantdoc", "all"],
                        default="all", help="which dataset to download")
    args = parser.parse_args()

    if not have_git():
        print("ERROR: git is not on PATH. Install git, or download the datasets manually:")
        for name, spec in DATASETS.items():
            print(f"  {name}: {spec['url']}")
        sys.exit(1)

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]

    ok = True
    for name in names:
        spec = DATASETS[name]
        print(f"\n{'=' * 60}\n{name.upper()}\n{'=' * 60}")
        dest = clone(name, spec)
        ok = verify(name, spec, dest) and ok

    print(f"\n{'=' * 60}")
    if ok:
        print("All requested datasets are in place.")
        print("Next: python ml_pipeline/src/data/build_splits.py")
    else:
        print("One or more datasets need attention -- see warnings above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
