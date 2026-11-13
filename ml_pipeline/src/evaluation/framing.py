"""Did fine-tuning teach these models to need a leaf that fills the frame?

The boxed-leaf result left one number unexplained. Handing a model the leaf
is worth about 11 points to every fine-tuned model and only 2.5 to a frozen
ImageNet probe. The obvious reading is that fine-tuning on PlantVillage --
where every image is one leaf filling a plain square -- teaches a model to
expect that framing, and a field photograph, where the largest leaf covers a
median 30% of the frame, does not provide it.

That reading cannot be tested on PlantDoc, because PlantDoc changes framing
and lighting and background and camera and label noise all at once. So this
tests it on PlantVillage instead, where everything else can be held still.

Each test image is zoomed out: the picture is shrunk to a fraction of the
frame and the space around it filled with the image's own border colour, so
the leaf occupies less of the frame and nothing else about the photograph
changes -- same leaf, same lesions, same bench, same exposure. Scale 1.0 is
the image untouched. Scale 0.55 puts the leaf at roughly the share of the
frame a PlantDoc leaf occupies, since 0.55 squared is about 0.30.

A model that reads the leaf should lose little. A model that has learnt
"disease is whatever fills a 256-pixel square" should fall apart. The gap
between the fine-tuned models and the frozen probe is the part fine-tuning
is responsible for.

Usage:
    python ml_pipeline/src/evaluation/framing.py
    python ml_pipeline/src/evaluation/framing.py --scales 1.0 0.55 --runs swinv2_tiny
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
for sub in ("data", "models", "debias"):
    sys.path.insert(0, os.path.join(SRC, sub))

import paths  # noqa: E402
from classes import PLANTVILLAGE_CLASSES  # noqa: E402
from classifier import Classifier  # noqa: E402
from dataset import PlantVillageSplit, eval_transform  # noqa: E402

NUM_CLASSES = len(PLANTVILLAGE_CLASSES)
CHECKPOINT_DIR = os.path.join(paths.REPO_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(paths.REPO_ROOT, "results")
RESULT_FILE = os.path.join(RESULTS_DIR, "framing.json")
BORDER = 8          # pixels of edge used to judge what the background is
BATCH = 64


def zoom_out(image, scale):
    """The image shrunk into its own frame, padded with its border colour.

    Padding with the border's median keeps the acquisition untouched: on
    PlantVillage's uniform benches the seam is invisible, so the only thing
    that changes is how much of the frame the leaf occupies. A flat grey or
    a black border would be a new cue, and the model could read the cue
    rather than the framing.
    """
    if scale >= 1.0:
        return image
    width, height = image.size
    pixels = np.asarray(image.convert("RGB"))
    edge = np.concatenate([pixels[:BORDER].reshape(-1, 3), pixels[-BORDER:].reshape(-1, 3),
                           pixels[:, :BORDER].reshape(-1, 3), pixels[:, -BORDER:].reshape(-1, 3)])
    background = tuple(int(v) for v in np.median(edge, axis=0))
    small = image.resize((max(1, round(width * scale)), max(1, round(height * scale))),
                         Image.BICUBIC)
    canvas = Image.new("RGB", (width, height), background)
    canvas.paste(small, ((width - small.width) // 2, (height - small.height) // 2))
    return canvas


class ZoomedTestSplit(torch.utils.data.Dataset):
    """The frozen PlantVillage test split, zoomed out by a fixed scale."""

    def __init__(self, transform, scale):
        self.inner = PlantVillageSplit("test", transform=None)
        self.transform, self.scale = transform, scale

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, index):
        row = self.inner.rows[index]
        path = os.path.join(self.inner.image_root, row["path"])
        with Image.open(paths.long_path(path)) as handle:
            image = handle.convert("RGB")
        return self.transform(zoom_out(image, self.scale)), int(row["class_idx"])


@torch.no_grad()
def logits_for(forward, transform, scale, log):
    loader = torch.utils.data.DataLoader(ZoomedTestSplit(transform, scale), batch_size=BATCH,
                                         shuffle=False, num_workers=4, pin_memory=True)
    chunks, labels = [], []
    for images, y in loader:
        chunks.append(forward(images.cuda(non_blocking=True)).float().cpu())
        labels.append(y)
    return torch.cat(chunks), torch.cat(labels)


def load_run(run):
    best = torch.load(os.path.join(CHECKPOINT_DIR, run, "best.pt"),
                      map_location="cpu", weights_only=True)
    cfg = best["config"]
    model = Classifier(cfg["model"], NUM_CLASSES, pretrained=False,
                       drop_rate=cfg["drop_rate"], drop_path_rate=cfg["drop_path_rate"])
    model.load_state_dict(best["model"])
    model.cuda().eval()
    amp = torch.float16 if cfg["amp_dtype"] == "float16" else torch.bfloat16

    def forward(images):
        with torch.autocast("cuda", dtype=amp):
            return model(images)
    return forward, eval_transform(cfg["img_size"], tuple(best["mean"]), tuple(best["std"])), model


def frozen_probe(log):
    """A head fitted on frozen ImageNet features, as the no-fine-tuning arm.

    It is the same probe the erasure experiment used, trained on the cached
    colour view, so the only difference from the fine-tuned runs is that the
    backbone never saw PlantVillage.
    """
    import timm
    from erasure import apply_head, fit_head, load

    cached = load("colour")
    rows = torch.tensor([i for i, s in enumerate(cached["split"]) if s == "train"])
    head, mean, std = fit_head(cached["features"][rows], cached["labels"][rows], seed=0)
    log("    frozen probe: head fitted on cached ImageNet features")

    backbone = timm.create_model("swinv2_tiny_window8_256", pretrained=True, num_classes=0)
    backbone.cuda().eval()
    config = timm.data.resolve_model_data_config(backbone)

    def forward(images):
        with torch.autocast("cuda", dtype=torch.float16):
            features = backbone(images).float()
        return apply_head(head, mean, std, features.cpu()).cuda()
    return forward, timm.data.create_transform(**config, is_training=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--scales", nargs="+", type=float,
                        default=[1.0, 0.8, 0.65, 0.55, 0.4, 0.25])
    parser.add_argument("--skip-probe", action="store_true")
    args = parser.parse_args()

    runs = args.runs or sorted(
        name for name in os.listdir(CHECKPOINT_DIR)
        if not name.startswith("dev_")
        and os.path.isfile(os.path.join(CHECKPOINT_DIR, name, "best.pt"))
        and os.path.isfile(os.path.join(RESULTS_DIR, f"{name}.json")))

    def log(message):
        print(message, flush=True)

    log(f"PlantVillage test, zoomed out. Scale 1.0 is untouched; 0.55 puts the leaf")
    log(f"at about the share of the frame a PlantDoc leaf occupies.\n")
    results = {}
    arms = [(run, None) for run in runs] + ([] if args.skip_probe else [("frozen probe", True)])
    for name, is_probe in arms:
        log(f"  {name}")
        if is_probe:
            forward, transform = frozen_probe(log)
            model = None
        else:
            forward, transform, model = load_run(name)
        scores = {}
        for scale in args.scales:
            logits, labels = logits_for(forward, transform, scale, log)
            scores[f"{scale}"] = (logits.argmax(1) == labels).float().mean().item()
            log(f"    scale {scale:<5} acc {scores[f'{scale}']:7.2%}")
        top = scores[f"{args.scales[0]}"]
        results[name] = {"accuracy": scores,
                         "drop_at_field_framing": top - scores.get("0.55", float("nan"))}
        del model, forward
        torch.cuda.empty_cache()

    with open(RESULT_FILE, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    log(f"\nsaved {os.path.relpath(RESULT_FILE, paths.REPO_ROOT)}")


if __name__ == "__main__":
    main()
