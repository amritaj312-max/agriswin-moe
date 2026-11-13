"""Does the field collapse come from the scene, or from the leaf itself?

A PlantVillage image is one leaf filling a plain frame. A PlantDoc image is a
photograph of a plant: the largest annotated leaf covers a median 30% of it,
and the rest is soil, sky, hands, other leaves. So a model scoring 22% on
PlantDoc is being asked two questions at once -- find the leaf, then name its
disease -- and failing the pair tells us nothing about which it failed.

This separates them. Every image is scored twice, on the same checkpoints and
the same 2,425 frozen photos: once whole, once cropped to the leaf box that
PlantDoc's object-detection release publishes for it. The gap between the two
says where the collapse lives:

  a large gap    the model can read a field leaf once it is handed one, and
                 the problem is framing and clutter -- segmentation, detection
                 or background work is the fix;
  no gap         handing it the leaf changes nothing, so the loss is inside
                 the leaf pixels, where Noyan (2022) found PlantVillage's
                 capture bias survives background removal. Then no amount of
                 background work helps, and what has to be removed is the
                 photo-session signature itself.

The boxes are frozen in data/splits/plantdoc_boxes.csv, keyed by content hash
like every other split here, and rebuilt by --build. Four measures are
reported, because overall accuracy hides what changed:

  accuracy                 over all 38 classes, what a deployed model faces
  crop accuracy            is it even the right plant?
  within-crop balanced     the crop is given (a farmer knows it), the model
                           chooses among that crop's diseases, and every class
                           counts equally -- so always answering a crop's
                           commonest disease scores 1/n, not the 44.5% such a
                           guess earns when classes are counted by frequency
  sink rate                share of images predicted "Pepper bell healthy",
                           21.9% of field images at full frame

Usage:
    python ml_pipeline/src/evaluation/boxed_leaves.py
    python ml_pipeline/src/evaluation/boxed_leaves.py --runs swinv2_tiny --margin 0 0.15
    python ml_pipeline/src/evaluation/boxed_leaves.py --build   # refreeze the boxes
"""

import argparse
import collections
import csv
import json
import os
import re
import sys
import urllib.request

import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
for sub in ("data", "models"):
    sys.path.insert(0, os.path.join(SRC, sub))

import paths  # noqa: E402
import plantdoc  # noqa: E402
from classes import PLANTVILLAGE_CLASSES  # noqa: E402
from classifier import Classifier  # noqa: E402
from dataset import eval_transform  # noqa: E402

BOX_FILE = os.path.join(paths.SPLIT_DIR, "plantdoc_boxes.csv")
# PlantDoc ships its boxes in a second repository, pinned as the images are.
# Only the two label files are fetched; its copies of the photographs are not
# needed, and would be a gigabyte of images this project already has.
BOX_REPO = "pratikkayal/PlantDoc-Object-Detection-Dataset"
BOX_COMMIT = "4730a233a555b30ee98e0879c63ad25d82407455"
IMAGE_EXT = re.compile(r"(\.(jpe?g|png))+$", re.IGNORECASE)
CHECKPOINT_DIR = os.path.join(paths.REPO_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(paths.REPO_ROOT, "results")
NUM_CLASSES = len(PLANTVILLAGE_CLASSES)
CROP = [name.split("___")[0] for name in PLANTVILLAGE_CLASSES]
CROPS = sorted(set(CROP))
CROP_CLASSES = {crop: [i for i, c in enumerate(CROP) if c == crop] for crop in CROPS}
SINK = PLANTVILLAGE_CLASSES.index("Pepper,_bell___healthy")
BATCH = 32


def published_boxes(log):
    """{filename stem: [box, ...]} from the pinned detection release.

    The two datasets name the same photograph differently -- the
    classification copy carries a doubled extension, "X.jpg.jpg", from the
    scraper that built it -- so the stem is matched with image extensions
    stripped. Degenerate rows (a frame or a box of zero area) are dropped.
    """
    boxes, dropped = collections.defaultdict(list), 0
    for split in ("train", "test"):
        url = f"https://raw.githubusercontent.com/{BOX_REPO}/{BOX_COMMIT}/{split}_labels.csv"
        log(f"  reading {split}_labels.csv from {BOX_COMMIT[:7]}")
        with urllib.request.urlopen(url) as response:
            text = response.read().decode("utf-8")
        for row in csv.DictReader(text.splitlines()):
            width, height = int(row["width"]), int(row["height"])
            x0, y0, x1, y1 = (int(row[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
            if width <= 0 or height <= 0 or x1 <= x0 or y1 <= y0:
                dropped += 1
                continue
            stem = IMAGE_EXT.sub("", os.path.basename(row["filename"]))
            boxes[stem].append((x0, y0, min(x1, width), min(y1, height), width, height, row["class"]))
    log(f"  {sum(len(v) for v in boxes.values())} boxes on {len(boxes)} images, {dropped} row(s) dropped")
    return boxes


def build(log):
    """Freeze the largest leaf box for every frozen PlantDoc image that has one.

    An image is kept only when its pixel dimensions on this machine match the
    ones the detection release recorded. Where they differ the two repositories
    hold different copies of a same-named photograph, and the box would crop
    the wrong part of it.
    """
    boxes = published_boxes(log)
    by_hash = {}
    for rel, digest in plantdoc.local_files().items():
        by_hash.setdefault(digest, rel)
    root = plantdoc.plantdoc_root()

    rows, skipped = [], collections.Counter()
    for entry in plantdoc.read_manifest():
        stem = IMAGE_EXT.sub("", os.path.basename(entry["example_path"]))
        if stem not in boxes:
            skipped["no box published"] += 1
            continue
        with Image.open(paths.long_path(os.path.join(root, by_hash[entry["sha256"]]))) as handle:
            size = handle.size
        box = max(boxes[stem], key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        if size != (box[4], box[5]):
            skipped["different copy of the photograph"] += 1
            continue
        rows.append({"sha256": entry["sha256"], "class_idx": entry["class_idx"],
                     "boxes": len(boxes[stem]), "xmin": box[0], "ymin": box[1],
                     "xmax": box[2], "ymax": box[3], "width": box[4], "height": box[5],
                     "box_class": box[6]})
    with open(BOX_FILE, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    area = sorted((r["xmax"] - r["xmin"]) * (r["ymax"] - r["ymin"]) / (r["width"] * r["height"])
                  for r in rows)
    log(f"  kept {len(rows)}; skipped " + ", ".join(f"{n} {why}" for why, n in skipped.items()))
    log(f"  the largest leaf covers a median {area[len(area) // 2]:.0%} of the frame")
    log(f"  wrote {os.path.relpath(BOX_FILE, paths.REPO_ROOT)}")


class BoxedPlantDoc(torch.utils.data.Dataset):
    """The frozen boxed images, whole or cropped to the leaf.

    `margin` grows the box by that fraction of its own size before cropping,
    clipped to the frame: a detection box is drawn tight to the leaf, while the
    models were trained on leaves with a little space around them.
    """

    def __init__(self, transform, margin=None):
        self.transform, self.margin = transform, margin
        by_hash = {}
        for rel, digest in plantdoc.local_files().items():
            by_hash.setdefault(digest, rel)
        root = plantdoc.plantdoc_root()
        with open(BOX_FILE, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        missing = [r for r in rows if r["sha256"] not in by_hash]
        if missing:
            raise FileNotFoundError(f"{len(missing)} boxed images are not on this machine; "
                                    f"re-download with download.py --dataset plantdoc")
        self.rows = [{"path": os.path.join(root, by_hash[r["sha256"]]),
                      "class_idx": int(r["class_idx"]),
                      "box": tuple(int(r[k]) for k in ("xmin", "ymin", "xmax", "ymax")),
                      "size": (int(r["width"]), int(r["height"]))} for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(paths.long_path(row["path"])) as handle:
            image = handle.convert("RGB")
        if self.margin is not None:
            x0, y0, x1, y1 = row["box"]
            width, height = row["size"]
            pad_x, pad_y = (x1 - x0) * self.margin / 2, (y1 - y0) * self.margin / 2
            image = image.crop((max(0, round(x0 - pad_x)), max(0, round(y0 - pad_y)),
                                min(width, round(x1 + pad_x)), min(height, round(y1 + pad_y))))
        return self.transform(image), row["class_idx"]


def measures(predictions, labels):
    """The four numbers, for one set of predictions."""
    correct = (predictions == labels)
    crop_of = torch.tensor([CROPS.index(CROP[i]) for i in range(NUM_CLASSES)])
    crop_right = (crop_of[predictions] == crop_of[labels])
    return {"accuracy": correct.float().mean().item(),
            "crop_accuracy": crop_right.float().mean().item(),
            "sink_rate": (predictions == SINK).float().mean().item()}


def within_crop_balanced(logits, labels):
    """Mean per-class recall when the crop is given, over crops with 2+ classes.

    Every class weighs the same, so answering a crop's commonest disease every
    time scores 1/n for that crop however the test set is distributed. That is
    the baseline the full-frame numbers could not clear.
    """
    recalls = []
    for crop in CROPS:
        members = CROP_CLASSES[crop]
        if len(members) < 2:
            continue                      # naming the crop already names the class
        index = torch.tensor(members)
        for position, class_idx in enumerate(members):
            rows = (labels == class_idx)
            if not rows.any():
                continue                  # PlantDoc has no image of this class
            chosen = logits[rows][:, index].argmax(1)
            recalls.append((chosen == position).float().mean().item())
    return sum(recalls) / len(recalls), len(recalls)


@torch.no_grad()
def score(run, margins, log):
    best_path = os.path.join(CHECKPOINT_DIR, run, "best.pt")
    if not os.path.isfile(best_path):
        log(f"  {run}: no checkpoint on this machine -- skipped")
        return None
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    cfg = best["config"]
    model = Classifier(cfg["model"], NUM_CLASSES, pretrained=False,
                       drop_rate=cfg["drop_rate"], drop_path_rate=cfg["drop_path_rate"])
    model.load_state_dict(best["model"])
    model.cuda().eval()
    transform = eval_transform(cfg["img_size"], tuple(best["mean"]), tuple(best["std"]))
    amp_dtype = torch.float16 if cfg["amp_dtype"] == "float16" else torch.bfloat16

    out = {"model": cfg["model"], "seed": cfg["seed"], "views": {}}
    for margin in margins:
        dataset = BoxedPlantDoc(transform, margin)
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=False,
                                             num_workers=4, pin_memory=True)
        logits, labels = [], []
        for images, y in loader:
            with torch.autocast("cuda", dtype=amp_dtype):
                logits.append(model(images.cuda(non_blocking=True)).float().cpu())
            labels.append(y)
        logits, labels = torch.cat(logits), torch.cat(labels)
        entry = measures(logits.argmax(1), labels)
        entry["within_crop_balanced"], entry["classes_counted"] = within_crop_balanced(logits, labels)
        name = "whole image" if margin is None else f"leaf box +{margin:.0%}"
        out["views"][name] = entry
        log(f"    {name:16s} acc {entry['accuracy']:6.2%}  crop {entry['crop_accuracy']:6.2%}  "
            f"within-crop {entry['within_crop_balanced']:6.2%}  sink {entry['sink_rate']:6.2%}")
    del model
    torch.cuda.empty_cache()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", nargs="+", help="run names (default: every local checkpoint)")
    parser.add_argument("--margin", nargs="+", type=float, default=[0.0, 0.15],
                        help="box margins to score, as a fraction of box size")
    parser.add_argument("--build", action="store_true",
                        help="refreeze data/splits/plantdoc_boxes.csv and stop")
    args = parser.parse_args()

    if args.build:
        build(lambda message: print(message, flush=True))
        return

    runs = args.runs or sorted(
        name for name in os.listdir(CHECKPOINT_DIR)
        if not name.startswith("dev_") and os.path.isfile(os.path.join(CHECKPOINT_DIR, name, "best.pt"))
        and os.path.isfile(os.path.join(RESULTS_DIR, f"{name}.json")))
    margins = [None] + list(args.margin)
    lines = []

    def log(message):
        print(message, flush=True)
        lines.append(message)

    with open(BOX_FILE, encoding="utf-8") as fh:
        boxed = sum(1 for _ in fh) - 1
    log(f"Scene or leaf? {boxed} frozen PlantDoc images, whole and cropped to "
        f"the published leaf box\n")
    results = {}
    for run in runs:
        log(f"  {run}")
        scored = score(run, margins, log)
        if scored:
            results[run] = scored

    dest = os.path.join(RESULTS_DIR, "boxed_leaves.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    log(f"\nsaved {os.path.relpath(dest, paths.REPO_ROOT)}")


if __name__ == "__main__":
    main()
