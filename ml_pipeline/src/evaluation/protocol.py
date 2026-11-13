"""How much of the gap to published PlantDoc numbers is protocol?

Published lab-to-field results on PlantDoc range from 57.6% (Khalil et al.
2026, BioCLIP 2 frozen features) to 68% (Salman et al. 2025). This project
reports 21-25%. A reviewer's first thought will be that something here is
broken, and that thought has to be answered before anything else in the paper
is read.

It is answered by scoring the same weights four ways. Two choices separate
the protocols, and neither is hidden -- they are simply different questions:

  which images   all 2,551 of PlantDoc that survive de-duplication, or the
                 236 in its own test/ folder. 228 of those 236 survive here;
                 the other 8 are duplicates or carry contradictory labels.
  which classes  all 38 PlantVillage outputs, which is what a deployed model
                 faces, or only the 28 that PlantDoc actually contains, which
                 cannot be known in advance but is what a closed-set
                 evaluation assumes.

The strictest cell is the honest deployment number and the one this paper
leads with. The loosest is comparable with the literature. Reporting both,
and the distance between them, is the point.

Usage:
    python ml_pipeline/src/evaluation/protocol.py
"""

import argparse
import csv
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
for sub in ("data", "models"):
    sys.path.insert(0, os.path.join(SRC, sub))

import paths  # noqa: E402
from classes import PLANTVILLAGE_CLASSES  # noqa: E402
from classifier import Classifier  # noqa: E402
from dataset import eval_transform  # noqa: E402
from plantdoc import MAPPED_CLASS_IDX, PlantDocDataset  # noqa: E402

NUM_CLASSES = len(PLANTVILLAGE_CLASSES)
CHECKPOINT_DIR = os.path.join(paths.REPO_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(paths.REPO_ROOT, "results")
RESULT_FILE = os.path.join(RESULTS_DIR, "protocol.json")
BATCH = 64


def official_test_mask():
    """Which manifest rows come from PlantDoc's own test/ folder."""
    with open(os.path.join(paths.SPLIT_DIR, "plantdoc_eval.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return torch.tensor([r["example_path"].startswith("test/") for r in rows])


@torch.no_grad()
def score(run, log):
    best_path = os.path.join(CHECKPOINT_DIR, run, "best.pt")
    if not os.path.isfile(best_path):
        log(f"  {run}: no checkpoint here -- skipped")
        return None
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    cfg = best["config"]
    model = Classifier(cfg["model"], NUM_CLASSES, pretrained=False,
                       drop_rate=cfg["drop_rate"], drop_path_rate=cfg["drop_path_rate"])
    model.load_state_dict(best["model"])
    model.cuda().eval()
    amp = torch.float16 if cfg["amp_dtype"] == "float16" else torch.bfloat16
    transform = eval_transform(cfg["img_size"], tuple(best["mean"]), tuple(best["std"]))

    loader = torch.utils.data.DataLoader(PlantDocDataset(transform), batch_size=BATCH,
                                         shuffle=False, num_workers=4, pin_memory=True)
    chunks, labels = [], []
    for images, y in loader:
        with torch.autocast("cuda", dtype=amp):
            chunks.append(model(images.cuda(non_blocking=True)).float().cpu())
        labels.append(y)
    logits, labels = torch.cat(chunks), torch.cat(labels)
    del model
    torch.cuda.empty_cache()

    keep = torch.tensor(sorted(set(MAPPED_CLASS_IDX)))
    restricted = torch.full_like(logits, float("-inf"))
    restricted[:, keep] = logits[:, keep]
    official = official_test_mask()

    out = {}
    for images_label, rows in (("all", torch.ones_like(official)), ("official_test", official)):
        for space_label, source in (("38_way", logits), ("28_way", restricted)):
            correct = (source[rows].argmax(1) == labels[rows]).float().mean().item()
            out[f"{images_label}__{space_label}"] = correct
    out["images_all"] = int(len(labels))
    out["images_official"] = int(official.sum())
    log(f"    all {out['all__38_way']:6.1%} / {out['all__28_way']:6.1%}   "
        f"official-test {out['official_test__38_way']:6.1%} / "
        f"{out['official_test__28_way']:6.1%}   (38-way / 28-way)")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", nargs="+")
    args = parser.parse_args()
    runs = args.runs or sorted(
        name for name in os.listdir(CHECKPOINT_DIR)
        if not name.startswith("dev_")
        and os.path.isfile(os.path.join(CHECKPOINT_DIR, name, "best.pt"))
        and os.path.isfile(os.path.join(RESULTS_DIR, f"{name}.json")))

    def log(message):
        print(message, flush=True)

    log("The same weights under four protocols. Published work reports 57.6-68%;")
    log("this project leads with the strictest cell, all images and all 38 outputs.\n")
    results = {}
    for run in runs:
        log(f"  {run}")
        scored = score(run, log)
        if scored:
            results[run] = scored
    if results:
        loosest = max(r["official_test__28_way"] for r in results.values())
        strictest = min(r["all__38_way"] for r in results.values())
        log(f"\n  the two choices move the same weights from {strictest:.1%} to {loosest:.1%}")
    with open(RESULT_FILE, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    log(f"  saved {os.path.relpath(RESULT_FILE, paths.REPO_ROOT)}")


if __name__ == "__main__":
    main()
