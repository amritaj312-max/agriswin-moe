"""Is it clutter, rather than scale, that a leaf box removes?

The framing run settled one thing and opened another. Scale sensitivity turned
out to be a property of these backbones rather than something fine-tuning
introduced: at a PlantDoc leaf's share of the frame, a frozen ImageNet probe
keeps as much of its accuracy as the fine-tuned models do. Yet a leaf box is
worth about 11 points to every fine-tuned model and 2.5 to the probe. If scale
does not explain that gap, the rest of what a box removes has to: the other
leaves, and the scene around them.

This tests it with scale held still. Each test image is shrunk to a fixed
share of the frame, exactly as in framing.py, and the space around it is
filled two ways:

  background   the image's own border colour -- the framing.py condition,
               already measured, and the control here
  clutter      other leaves of the same class, cut out with PlantVillage's
               masks and scattered around the target

Only the surround differs, so the gap between the two is the cost of clutter
alone. The distractors share the target's class on purpose: a PlantDoc
photograph is one plant, so its other leaves carry the same label. Pasting a
different disease would measure how a model handles an ambiguous image, which
is a different and less interesting question.

If the fine-tuned models lose more here than the probe does, then what a box
buys them is a clean frame, and the fix is a detector rather than a different
backbone.

Usage:
    python ml_pipeline/src/evaluation/clutter.py
    python ml_pipeline/src/evaluation/clutter.py --scales 0.55 --runs swinv2_tiny
"""

import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
for sub in ("data", "models", "debias"):
    sys.path.insert(0, os.path.join(SRC, sub))

import paths  # noqa: E402
from dataset import PlantVillageSplit  # noqa: E402
from features import MASK_FLOOR, mask_index, mask_key  # noqa: E402
from framing import BATCH, RESULTS_DIR, frozen_probe, load_run, zoom_out  # noqa: E402

RESULT_FILE = os.path.join(RESULTS_DIR, "clutter.json")
DISTRACTORS = 5
CHECKPOINT_DIR = os.path.join(paths.REPO_ROOT, "checkpoints")


def cut_leaf(colour_path, mask_path):
    """A leaf on transparency, so it can be pasted without its bench."""
    with Image.open(paths.long_path(colour_path)) as handle:
        colour = handle.convert("RGB")
    with Image.open(paths.long_path(mask_path)) as handle:
        mask = handle.convert("L")
    if mask.size != colour.size:
        mask = mask.resize(colour.size, Image.NEAREST)
    keep = (np.asarray(mask) >= MASK_FLOOR).astype(np.uint8) * 255
    out = colour.convert("RGBA")
    out.putalpha(Image.fromarray(keep))
    return out


def clutter_pad(image, scale, distractors, seed):
    """The image shrunk into its frame, with other leaves scattered around it.

    The distractors go down first and the target last, so the target is never
    occluded: the question is whether a model can still read a leaf it can see
    in full, not whether it can see through one.
    """
    base = zoom_out(image, scale)                 # same background as the control
    width, height = base.size
    target = round(min(width, height) * scale)
    rng = random.Random(seed)
    for leaf in distractors:
        size = max(8, int(target * rng.uniform(0.6, 1.1)))
        patch = leaf.resize((size, size), Image.BICUBIC).rotate(
            rng.uniform(0, 360), expand=True, resample=Image.BICUBIC)
        # keep the centre clear for the target
        while True:
            x = rng.randint(-size // 3, width - size * 2 // 3)
            y = rng.randint(-size // 3, height - size * 2 // 3)
            if abs(x + size / 2 - width / 2) > target * 0.45 or \
               abs(y + size / 2 - height / 2) > target * 0.45:
                break
        base.paste(patch, (x, y), patch)
    small = image.resize((max(1, round(width * scale)), max(1, round(height * scale))),
                         Image.BICUBIC)
    base.paste(small, ((width - small.width) // 2, (height - small.height) // 2))
    return base


class ClutteredTestSplit(torch.utils.data.Dataset):
    """The frozen test split at one scale, surrounded by same-class leaves."""

    def __init__(self, transform, scale, n_distractors=DISTRACTORS):
        self.inner = PlantVillageSplit("test", transform=None)
        self.transform, self.scale, self.n = transform, scale, n_distractors
        root = os.path.join(paths.data_root(), "raw", "plantvillage", "raw")
        self.colour_root = self.inner.image_root
        self.masks = mask_index(root)
        # which other images could stand in as clutter, per class
        self.by_class = {}
        for row in self.inner.rows:
            self.by_class.setdefault(row["class_name"], []).append(row["path"])

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, index):
        row = self.inner.rows[index]
        with Image.open(paths.long_path(os.path.join(self.colour_root, row["path"]))) as handle:
            image = handle.convert("RGB")
        pool = self.by_class[row["class_name"]]
        rng = random.Random(index)
        leaves = []
        for _ in range(self.n):
            other = rng.choice(pool)
            if other == row["path"] and len(pool) > 1:
                continue
            leaves.append(cut_leaf(os.path.join(self.colour_root, other),
                                   self.masks[mask_key(other)]))
        return self.transform(clutter_pad(image, self.scale, leaves, index)), int(row["class_idx"])


@torch.no_grad()
def score(forward, transform, scale, workers=4):
    loader = torch.utils.data.DataLoader(ClutteredTestSplit(transform, scale),
                                         batch_size=BATCH, shuffle=False,
                                         num_workers=workers, pin_memory=True)
    right = seen = 0
    for images, y in loader:
        predicted = forward(images.cuda(non_blocking=True)).float().cpu().argmax(1)
        right += int((predicted == y).sum())
        seen += len(y)
    return right / seen


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--scales", nargs="+", type=float, default=[0.55, 0.4])
    parser.add_argument("--skip-probe", action="store_true")
    args = parser.parse_args()

    runs = args.runs or sorted(
        name for name in os.listdir(CHECKPOINT_DIR)
        if not name.startswith("dev_")
        and os.path.isfile(os.path.join(CHECKPOINT_DIR, name, "best.pt"))
        and os.path.isfile(os.path.join(RESULTS_DIR, f"{name}.json")))

    def log(message):
        print(message, flush=True)

    control_file = os.path.join(RESULTS_DIR, "framing.json")
    control = {}
    if os.path.isfile(control_file):
        with open(control_file, encoding="utf-8") as fh:
            control = json.load(fh)

    log("PlantVillage test at a fixed scale, padded with its own background (control)")
    log(f"or with {DISTRACTORS} same-class leaves (clutter). Only the surround differs.\n")
    results = {}
    arms = [(run, False) for run in runs] + ([] if args.skip_probe else [("frozen probe", True)])
    for name, is_probe in arms:
        log(f"  {name}")
        forward, transform = frozen_probe(log) if is_probe else load_run(name)[:2]
        entry = {}
        for scale in args.scales:
            cluttered = score(forward, transform, scale)
            plain = control.get(name, {}).get("accuracy", {}).get(str(scale))
            entry[str(scale)] = {"clutter": cluttered, "background": plain,
                                 "cost": (plain - cluttered) if plain else None}
            if plain:
                log(f"    scale {scale:<5} background {plain:7.2%}   clutter {cluttered:7.2%}"
                    f"   cost {plain - cluttered:+7.2%}")
            else:
                log(f"    scale {scale:<5} clutter {cluttered:7.2%}   (no control; run framing.py)")
        results[name] = entry
        torch.cuda.empty_cache()

    with open(RESULT_FILE, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    log(f"\nsaved {os.path.relpath(RESULT_FILE, paths.REPO_ROOT)}")


if __name__ == "__main__":
    main()
