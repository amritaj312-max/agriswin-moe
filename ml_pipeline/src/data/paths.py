"""
Where the data lives.

By default the datasets live inside the repo at <repo>/data/raw, which .gitignore
excludes, so the images sit next to the code without ever being committed. The
root stays configurable for the cases where that is not possible -- a small system
drive, a shared dataset cache, or Colab/Kaggle -- and is resolved in this order:

    1. $AGRISWIN_DATA_ROOT
    2. ml_pipeline/configs/data_root.txt   (git-ignored, written by set_data_root)
    3. <repo>/data                          (the default)

Split CSVs always stay in the repo and store paths *relative to the image root*,
never absolute. That is what lets the same frozen split be used here, on Colab
and on Kaggle without rewriting anything.
"""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CONFIG_FILE = os.path.join(REPO_ROOT, "ml_pipeline", "configs", "data_root.txt")

# committed, small: the frozen splits and the results tables
SPLIT_DIR = os.path.join(REPO_ROOT, "data", "splits")


def data_root():
    """Resolve the data root. See module docstring for precedence."""
    env = os.environ.get("AGRISWIN_DATA_ROOT")
    if env:
        return os.path.abspath(env)

    if os.path.isfile(CONFIG_FILE):
        with open(CONFIG_FILE, encoding="utf-8") as fh:
            configured = fh.read().strip()
        if configured:
            return os.path.abspath(configured)

    return os.path.join(REPO_ROOT, "data")


def set_data_root(path):
    """Persist the data root so later runs and later sessions pick it up."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    os.makedirs(path, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as fh:
        fh.write(path + "\n")
    return path


def raw_dir():
    return os.path.join(data_root(), "raw")


def long_path(path):
    """Windows extended-length form, which lifts the 260-character path limit.

    PlantDoc's filenames are scraped web page titles; three of them sit in
    paths over 260 characters, and Windows then refuses to open them by name
    even though os.listdir lists them. Prefixing the absolute path with \\?\
    bypasses the limit. Everything that opens a dataset image goes through
    here, so no loader silently skips those three.
    """
    if os.name == "nt":
        path = os.path.abspath(path)
        if not path.startswith("\\\\?\\"):
            return "\\\\?\\" + path
    return path


def image_root(dataset="plantvillage"):
    """Directory holding the class subdirectories for a dataset."""
    roots = {
        # PlantVillage ships colour, greyscale and segmented copies of every
        # image; only colour is checked out (see download.py).
        "plantvillage": os.path.join(raw_dir(), "plantvillage", "raw", "color"),
        "plantdoc": os.path.join(raw_dir(), "plantdoc", "train"),
    }
    return roots[dataset]


def free_gb(path=None):
    """Free space in GB on the volume holding `path` (default: the data root)."""
    target = path or data_root()
    while not os.path.exists(target):
        parent = os.path.dirname(target)
        if parent == target:
            break
        target = parent
    return os.statvfs(target).f_bavail * os.statvfs(target).f_frsize / 1024**3 \
        if hasattr(os, "statvfs") else \
        __import__("shutil").disk_usage(target).free / 1024**3


if __name__ == "__main__":
    print(f"repo root  : {REPO_ROOT}")
    print(f"data root  : {data_root()}")
    print(f"raw dir    : {raw_dir()}")
    print(f"splits     : {SPLIT_DIR}")
    print(f"free space : {free_gb():.1f} GB")
    for ds in ("plantvillage", "plantdoc"):
        root = image_root(ds)
        state = "present" if os.path.isdir(root) else "not downloaded"
        print(f"  {ds:<14} {state:<15} {root}")
