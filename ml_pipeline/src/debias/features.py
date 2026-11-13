"""Frozen backbone features, cached once, for the debiasing experiments.

Everything downstream -- estimating a photo session's signature, erasing it,
refitting a head -- is linear algebra on a few hundred megabytes of vectors.
Only this file needs a GPU, and only once per view: about three minutes an
L4 for all 54,305 PlantVillage images.

The backbone is ImageNet-pretrained Swin-V2-T, frozen and never fine-tuned
here. That is deliberate. A backbone fine-tuned on PlantVillage has already
absorbed the session shortcut, so it cannot answer whether the shortcut can
be removed; a frozen one gives a clean linear question, and a frozen-backbone
probe is a baseline the method has to beat anyway.

Four views are cached:

  colour        PlantVillage as it ships -- the leaf and the bench it was
                photographed on
  leaf          the same leaf cut out with PlantVillage's own mask and
                composited onto a flat random colour. The background then
                carries no session information at all, so a signature learned
                from these features is a signature carried by leaf pixels --
                which is the only interesting kind. Noyan (2022) found the
                bias survives background removal, and this is how to study
                what survives.
  plantdoc      the frozen 2,551-image field set, whole
  plantdoc_box  the same photographs cropped to their published leaf box

Usage:
    python ml_pipeline/src/debias/features.py                 # all four views
    python ml_pipeline/src/debias/features.py --views leaf --limit 200
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import timm
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
for sub in ("data", "evaluation"):
    sys.path.insert(0, os.path.join(SRC, sub))

import paths  # noqa: E402
import plantdoc  # noqa: E402
import sources  # noqa: E402

BACKBONE = "swinv2_tiny_window8_256"
CACHE_DIR = os.path.join(paths.REPO_ROOT, "results", "features")
BOX_FILE = os.path.join(paths.SPLIT_DIR, "plantdoc_boxes.csv")
BOX_MARGIN = 0.15
BATCH = 64
MASK_FLOOR = 12          # a segmented pixel this dark is background, not leaf


def mask_index(root):
    """{(class, tag): path} for every background-removed copy.

    The two trees do not always agree on a filename. In four classes the
    colour copy dropped the UUID that the segmented copy kept, so
    "RS_Rust 1563.JPG" has to find "<uuid>___RS_Rust 1563_final_masked.jpg".
    What both always carry is the tag after '___' -- the session code and
    frame number -- so that is the key. It matches all 54,305 images, where
    matching on the whole filename misses 1,196 of them.
    """
    index, folder = {}, os.path.join(root, "segmented")
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"{folder} is missing. Fetch it with:\n"
            f"  python ml_pipeline/src/data/download.py --segmented")
    for class_name in os.listdir(folder):
        for name in os.listdir(os.path.join(folder, class_name)):
            stem = name.rsplit(".", 1)[0].removesuffix("_final_masked")
            index[(class_name, stem.split("___", 1)[-1])] = os.path.join(folder, class_name, name)
    return index


def mask_key(colour_relative):
    """The (class, tag) a colour image's mask is filed under."""
    class_name, name = colour_relative.replace("\\", "/").split("/")[-2:]
    return class_name, name.rsplit(".", 1)[0].split("___", 1)[-1]


def leaf_on_flat_colour(colour, segmented, seed):
    """The leaf, cut out by PlantVillage's mask, on one random flat colour.

    A flat colour is chosen on purpose. Pasting a background taken from
    another photograph would smuggle that photograph's session into the
    image, which is the very thing being measured; noise would add texture a
    texture-sensitive backbone could read as a session. A single colour,
    drawn from this image's own index, carries nothing and is reproducible.

    A few masks were saved at a different resolution from the photograph they
    annotate, so the mask is resampled to the image when they disagree. The
    mask only decides which pixels are leaf; the pixels kept are always the
    colour image's own, so resampling costs a little edge precision and
    nothing else.
    """
    if segmented.size != colour.size:
        segmented = segmented.resize(colour.size, Image.NEAREST)
    mask = np.asarray(segmented.convert("L")) >= MASK_FLOOR
    out = np.asarray(colour.convert("RGB")).copy()
    rng = np.random.default_rng(seed)
    out[~mask] = rng.integers(0, 256, size=3, dtype=np.uint8)
    return Image.fromarray(out)


class PlantVillageView(torch.utils.data.Dataset):
    """PlantVillage rows, served either as shipped or as leaf-on-colour."""

    def __init__(self, rows, transform, leaf_only):
        self.rows, self.transform, self.leaf_only = rows, transform, leaf_only
        self.root = os.path.join(paths.data_root(), "raw", "plantvillage", "raw")
        self.masks = mask_index(self.root) if leaf_only else {}
        missing = [r["path"] for r in rows if mask_key(r["path"]) not in self.masks]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images have no mask, e.g. {missing[0]}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(paths.long_path(os.path.join(self.root, "color", row["path"]))) as handle:
            image = handle.convert("RGB")
        if self.leaf_only:
            with Image.open(paths.long_path(self.masks[mask_key(row["path"])])) as handle:
                image = leaf_on_flat_colour(image, handle, index)
        return self.transform(image), int(row["class_idx"])


class PlantDocView(torch.utils.data.Dataset):
    """The frozen field set, whole or cropped to its leaf box."""

    def __init__(self, transform, boxed):
        self.transform, self.boxed = transform, boxed
        by_hash = {}
        for rel, digest in plantdoc.local_files().items():
            by_hash.setdefault(digest, rel)
        root = plantdoc.plantdoc_root()
        source = BOX_FILE if boxed else None
        if boxed:
            with open(source, encoding="utf-8") as fh:
                entries = [{"sha256": r["sha256"], "class_idx": int(r["class_idx"]),
                            "box": tuple(int(r[k]) for k in ("xmin", "ymin", "xmax", "ymax")),
                            "size": (int(r["width"]), int(r["height"]))}
                           for r in csv.DictReader(fh)]
        else:
            entries = [{"sha256": r["sha256"], "class_idx": int(r["class_idx"]),
                        "box": None, "size": None} for r in plantdoc.read_manifest()]
        self.rows = [{**e, "path": os.path.join(root, by_hash[e["sha256"]])} for e in entries]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(paths.long_path(row["path"])) as handle:
            image = handle.convert("RGB")
        if row["box"] is not None:
            x0, y0, x1, y1 = row["box"]
            width, height = row["size"]
            pad_x, pad_y = (x1 - x0) * BOX_MARGIN / 2, (y1 - y0) * BOX_MARGIN / 2
            image = image.crop((max(0, round(x0 - pad_x)), max(0, round(y0 - pad_y)),
                                min(width, round(x1 + pad_x)), min(height, round(y1 + pad_y))))
        return self.transform(image), row["class_idx"]


def backbone():
    """The frozen feature extractor, and the transform it was trained with."""
    model = timm.create_model(BACKBONE, pretrained=True, num_classes=0)
    model.cuda().eval()
    config = timm.data.resolve_model_data_config(model)
    return model, timm.data.create_transform(**config, is_training=False)


@torch.no_grad()
def extract(model, dataset, log, workers=4):
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=False,
                                         num_workers=workers, pin_memory=True)
    chunks, labels, started, seen = [], [], time.time(), 0
    for images, y in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            chunks.append(model(images.cuda(non_blocking=True)).float().cpu())
        labels.append(y)
        seen += len(y)
        if time.time() - started > 20:
            log(f"      {seen}/{len(dataset)}")
            started = time.time()
    return torch.cat(chunks), torch.cat(labels)


def cache(views, limit, log):
    os.makedirs(CACHE_DIR, exist_ok=True)
    model, transform = backbone()
    rows = sources.read_rows()
    if limit:
        rows = rows[::max(1, len(rows) // limit)][:limit]
        log(f"  limited to {len(rows)} PlantVillage rows (a smoke test, not a result)")

    for view in views:
        started = time.time()
        if view in ("colour", "leaf"):
            dataset = PlantVillageView(rows, transform, leaf_only=(view == "leaf"))
            extra = {"source": [r["source"] for r in dataset.rows],
                     "split": [r["split"] for r in dataset.rows],
                     "class_name": [r["class_name"] for r in dataset.rows]}
        else:
            dataset = PlantDocView(transform, boxed=(view == "plantdoc_box"))
            extra = {}
        log(f"    {view}: {len(dataset)} images")
        features, labels = extract(model, dataset, log)
        payload = {"features": features, "labels": labels, "view": view,
                   "backbone": BACKBONE, **extra}
        torch.save(payload, os.path.join(CACHE_DIR, f"{view}.pt"))
        log(f"    {view}: {tuple(features.shape)} in {time.time() - started:.0f}s "
            f"-> results/features/{view}.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--views", nargs="+",
                        default=["colour", "leaf", "plantdoc", "plantdoc_box"])
    parser.add_argument("--limit", type=int, help="use only this many PlantVillage images")
    args = parser.parse_args()
    cache(args.views, args.limit, lambda message: print(message, flush=True))


if __name__ == "__main__":
    main()
