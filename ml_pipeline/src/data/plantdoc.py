"""
PlantDoc, mapped onto the PlantVillage label space, as a frozen evaluation set.

PlantDoc is photographs of leaves in real fields -- varied backgrounds,
daylight, several leaves per frame -- against PlantVillage's single leaf on a
uniform background. It is never trained on. It is the cross-dataset test that
says what a model trained on lab images does when it meets a field image, and
the evidence for the paper's honest claim about domain shift (ACTION_PLAN 3.3):
not "we fix it", but "under domain shift the model knows that it does not know".

Which images, with which labels
-------------------------------
The evaluation set is frozen in data/splits/plantdoc_eval.csv, exactly as the
PlantVillage splits are, and for the same reason: every row of a table must be
scored on the same images. It holds 2,551 of the repository's 2,578 files, and
the 27 left out are left out on purpose:

  - 18 extra copies of 18 images that PlantDoc ships twice under one label
    (often once in its train/ and once in its test/ folder). One copy is kept;
    the other would count the same answer twice.
  - all 18 copies of 9 images that PlantDoc ships twice under two DIFFERENT
    labels: 3 corn photos filed as both Gray Leaf Spot and Leaf Blight, 5 potato
    photos as both early and late blight, 1 tomato as both Septoria and
    bacterial spot. Whatever a model answers, one copy is "wrong", so they
    measure nothing but PlantDoc's label noise -- and the corn ones would land
    on the very pair this project's ablations are judged on.

Images are identified by the SHA-256 of their bytes, not by filename. PlantDoc's
names are scraped web page titles: on Windows some must be renamed (they contain
'?' or exceed the path limit), on Linux they are not, so the same image has a
different name on the laptop and on Colab. Its bytes are the same everywhere.

The label mapping
-----------------
PlantDoc's 28 class folders carry their own names, mapped below onto the
38-class PlantVillage space. Two judgement calls a reviewer may ask about:

  - "Corn leaf blight" is PlantDoc's name for Northern Leaf Blight.
  - A bare crop name ("Apple leaf", "Tomato leaf") means a healthy leaf of that
    crop, and maps to that crop's ___healthy class.

Ten PlantVillage classes have no PlantDoc equivalent, so a model is scored
against 28 of its 38 outputs. evaluate.py reports accuracy twice: over all 38
logits, which is what a deployed model faces, and restricted to the 28
reachable classes, an upper bound.

Usage:
    python ml_pipeline/src/data/plantdoc.py            # check every image is here
    python ml_pipeline/src/data/plantdoc.py --build    # rebuild the manifest (not normally needed)
"""

import argparse
import csv
import hashlib
import json
import os
import sys

from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
from classes import CLASS_TO_IDX, NUM_CLASSES, PLANTVILLAGE_CLASSES  # noqa: E402

MANIFEST = os.path.join(paths.SPLIT_DIR, "plantdoc_eval.csv")
MANIFEST_FIELDS = ["sha256", "class_idx", "plantdoc_class", "example_path"]

# PlantDoc directory name -> PlantVillage class name.
PLANTDOC_TO_PLANTVILLAGE = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}

# PlantVillage class indices PlantDoc can actually produce
MAPPED_CLASS_IDX = sorted(CLASS_TO_IDX[v] for v in PLANTDOC_TO_PLANTVILLAGE.values())


def plantdoc_root():
    """The dataset directory, holding the train/ and test/ class folders."""
    return os.path.dirname(paths.image_root("plantdoc"))


def _sha256(path):
    digest = hashlib.sha256()
    with open(paths.long_path(path), "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def local_files():
    """{relative path: sha256} for every file in the local PlantDoc copy.

    Hashing 2,600 photos takes a few seconds, so hashes are cached next to the
    data (data/raw is git-ignored) and recomputed only for files whose size or
    modification time changed.
    """
    root = plantdoc_root()
    if not os.path.isdir(root):
        raise FileNotFoundError(f"{root} not found. Run:\n"
                                f"  python ml_pipeline/src/data/download.py --dataset plantdoc")
    cache_path = os.path.join(root, ".sha256_cache.json")
    try:
        with open(cache_path, encoding="utf-8") as fh:
            cache = json.load(fh)
    except (OSError, ValueError):
        cache = {}

    found, fresh = {}, {}
    for split in ("train", "test"):
        for folder in sorted(os.listdir(os.path.join(root, split))):
            for name in sorted(os.listdir(os.path.join(root, split, folder))):
                rel = f"{split}/{folder}/{name}"
                full = os.path.join(root, split, folder, name)
                stat = os.stat(paths.long_path(full))
                key = f"{stat.st_size}:{int(stat.st_mtime)}"
                entry = cache.get(rel)
                digest = entry["sha256"] if entry and entry["key"] == key else _sha256(full)
                fresh[rel] = {"key": key, "sha256": digest}
                found[rel] = digest
    if fresh != cache:
        try:
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(fresh, fh)
        except OSError:
            pass          # a read-only copy still works, just without the cache
    return found


def read_manifest():
    if not os.path.isfile(MANIFEST):
        raise FileNotFoundError(f"{MANIFEST} not found. It is committed to the repository; "
                                f"run `git pull`, or rebuild it with plantdoc.py --build")
    with open(MANIFEST, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fingerprint(rows=None):
    """Which images, with which labels, the evaluation set holds -- by content,
    so it is the same on Windows and Linux. Equal fingerprints prove two
    evaluations scored the same PlantDoc images."""
    rows = read_manifest() if rows is None else rows
    lines = sorted(f"{r['sha256']}\t{r['class_idx']}" for r in rows)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


def build_manifest():
    """Deduplicate the local copy by content and write the frozen manifest.
    Returns (kept rows, extra copies dropped, conflicting images dropped)."""
    groups = {}
    for rel, digest in local_files().items():
        folder = rel.split("/")[1]
        if folder not in PLANTDOC_TO_PLANTVILLAGE:
            raise KeyError(f"PlantDoc folder {folder!r} has no entry in PLANTDOC_TO_PLANTVILLAGE")
        groups.setdefault(digest, []).append((folder, rel))

    rows, extra, conflicts = [], 0, []
    for digest, copies in groups.items():
        folders = sorted({folder for folder, _ in copies})
        if len(folders) > 1:
            conflicts.append(folders)
            continue
        extra += len(copies) - 1
        folder = folders[0]
        rows.append({"sha256": digest,
                     "class_idx": CLASS_TO_IDX[PLANTDOC_TO_PLANTVILLAGE[folder]],
                     "plantdoc_class": folder,
                     # informative only: the name differs between Windows and
                     # Linux, which is why images are matched by sha256
                     "example_path": sorted(rel for _, rel in copies)[0]})
    rows.sort(key=lambda r: (r["plantdoc_class"], r["example_path"]))
    os.makedirs(paths.SPLIT_DIR, exist_ok=True)
    with open(MANIFEST, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows, extra, conflicts


class PlantDocDataset(Dataset):
    """The frozen PlantDoc evaluation set, labelled in PlantVillage's 38 classes.

    Each manifest row is found on disk by its SHA-256, so the same 2,551 images
    are used whatever this machine called them. A missing image is an error,
    never a silent skip: a smaller set would quietly change every number.
    """

    def __init__(self, transform=None):
        self.transform = transform
        manifest = read_manifest()
        by_hash = {}
        for rel, digest in local_files().items():
            by_hash.setdefault(digest, rel)
        missing = [r for r in manifest if r["sha256"] not in by_hash]
        if missing:
            examples = "\n".join(f"    {r['plantdoc_class']}: {r['example_path']}" for r in missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} of the {len(manifest)} PlantDoc images in the manifest are not "
                f"on this machine (matched by content), e.g.:\n{examples}\n"
                f"Re-download with: python ml_pipeline/src/data/download.py --dataset plantdoc")
        root = plantdoc_root()
        self.rows = [{"path": os.path.join(root, by_hash[r["sha256"]]),
                      "class_idx": int(r["class_idx"]),
                      "plantdoc_class": r["plantdoc_class"]} for r in manifest]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        # long_path: some images sit in paths over Windows' 260-character
        # limit and cannot be opened by plain name (paths.long_path says why)
        with Image.open(paths.long_path(row["path"])) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, row["class_idx"]

    def class_counts(self):
        counts = [0] * NUM_CLASSES
        for row in self.rows:
            counts[row["class_idx"]] += 1
        return counts


def main():
    parser = argparse.ArgumentParser(description="PlantDoc evaluation set: check or rebuild")
    parser.add_argument("--build", action="store_true",
                        help="rebuild data/splits/plantdoc_eval.csv from the local copy")
    args = parser.parse_args()

    targets = list(PLANTDOC_TO_PLANTVILLAGE.values())
    assert len(set(targets)) == len(targets), "two PlantDoc folders map to one PlantVillage class"
    assert set(targets) <= set(PLANTVILLAGE_CLASSES), "a mapping target is not a PlantVillage class"

    if args.build:
        rows, extra, conflicts = build_manifest()
        print(f"Wrote {os.path.relpath(MANIFEST, paths.REPO_ROOT)}: {len(rows)} images")
        print(f"  dropped {extra} extra copies of images PlantDoc ships twice under one label")
        print(f"  dropped {len(conflicts)} images PlantDoc ships under two different labels:")
        for folders in sorted(conflicts):
            print(f"    {' vs '.join(folders)}")

    ds = PlantDocDataset()
    counts = ds.class_counts()
    print(f"\nPlantDoc evaluation set: {len(ds)} images, mapped onto {len(MAPPED_CLASS_IDX)} of "
          f"{NUM_CLASSES} PlantVillage classes, fingerprint {fingerprint()}\n")
    print(f"  {'PlantVillage class':<50}{'images':>7}")
    for idx in MAPPED_CLASS_IDX:
        print(f"  {PLANTVILLAGE_CLASSES[idx]:<50}{counts[idx]:>7}")

    print("\nOpening every image...")
    unreadable = []
    for row in ds.rows:
        try:
            with Image.open(paths.long_path(row["path"])) as im:
                im.verify()
        except Exception as exc:  # noqa: BLE001 -- web-scraped files; report, don't crash
            unreadable.append(f"{os.path.basename(row['path'])}: {type(exc).__name__}")
    print(f"  {len(ds) - len(unreadable)} of {len(ds)} readable")
    for problem in unreadable:
        print(f"    {problem}")
    if unreadable:
        raise SystemExit("some PlantDoc images could not be opened")
    print("\nOK: every image in the frozen PlantDoc set is present, readable and mapped.")


if __name__ == "__main__":
    main()
