"""AGFT: Adaptive Gated Fine-Tuning — one self-contained study.

Written as a module so it can be tested on a laptop, and split on the
`# %% CELL:` markers into the single Colab notebook that actually runs it.

The question
------------
A plant-disease model reaches ~99.8% on PlantVillage and collapses on field
photographs. Kumar et al. (2022, LP-FT) explain why in general terms:
fine-tuning distorts pretrained features and can underperform a plain linear
probe out of distribution. Their remedy, and WiSE-FT's (Wortsman et al.,
2022), is a *fixed* trade-off — probe first then fine-tune, or interpolate the
two weight sets with one global alpha.

A fixed trade-off has to be wrong somewhere, because which branch is better
depends on the image. AGFT therefore keeps both branches and learns a gate
that weighs them **per input**:

    f_frozen = B_frozen(x)          pretrained, never updated
    f_tuned  = B_tuned(x)           fine-tuned
    g        = sigmoid(MLP(f_frozen))        in [0, 1], one value per image
    logits   = g * H_t(f_tuned) + (1 - g) * H_f(f_frozen)

The gate reads the *frozen* features on purpose: they are the branch whose
statistics do not move during training, so the gate sees a stable description
of the input rather than one that drifts under it.

Three ablations make the claim falsifiable, and they are the point of the
experiment rather than decoration:

    g = 1          plain fine-tuning
    g = 0          linear probe on frozen features
    g = alpha      one constant fitted on validation -- a WiSE-FT-style fixed
                   trade-off, in output space
    g = learned    AGFT

If AGFT does not beat the best constant alpha, the gate adds nothing and the
paper says so.

What is measured
----------------
In-distribution accuracy on a held-out PlantVillage test split, and
out-of-distribution accuracy on PlantDoc, which the model never trains on.
PlantVillage alone cannot rank anything: everything reaches ~99.8% and the
differences are smaller than the noise between seeds.

Protocol notes that matter to a reviewer
----------------------------------------
* **Leakage.** PlantVillage holds many near-duplicate photographs of the same
  leaf. Splitting at random puts copies of one leaf on both sides and inflates
  every number. Images are grouped by perceptual hash and whole groups are
  assigned to one split, so a leaf never appears on both sides.
* **Honest tuning.** Hyperparameters come from Optuna on the training folds
  only. The test split and PlantDoc are touched once, at the end.
* **Early stopping** on validation macro-F1, never on test.
* **Resumability.** Every stage writes its result to Drive and is skipped if
  that result exists, so a disconnected Colab session resumes by re-running.
"""

# %% CELL: imports
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# %% CELL: config


@dataclass
class Config:
    """Everything the study depends on, in one place."""

    # data
    pv_dataset: str = "DScomp380/plant_village"
    pd_dataset: str = "agyaatcoder/PlantDoc"
    drop_classes: tuple = ("Background_without_leaves",)
    img_size: int = 224
    val_frac: float = 0.15
    test_frac: float = 0.15

    # models: three published backbones, then the novel one built on the first
    backbones: tuple = (
        "convnextv2_tiny.fcmae_ft_in22k_in1k",
        "mobilenetv4_conv_medium.e500_r256_in1k",
        "swinv2_tiny_window8_256.ms_in1k",
    )
    agft_backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k"

    # training
    batch_size: int = 64
    epochs: int = 12
    patience: int = 3
    lr: float = 3e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.0
    drop_rate: float = 0.2
    warmup_epochs: int = 1
    seed: int = 42

    # search and cross-validation, kept small enough for one L4
    optuna_trials: int = 10
    optuna_epochs: int = 3
    optuna_subset: float = 0.2
    folds: int = 5
    fold_epochs: int = 8

    root: str = "/content/drive/MyDrive/agft"
    # The dataset cache deliberately does NOT live on Drive. Training reads it
    # at random through memory-mapped Arrow files, and doing that over FUSE
    # with several loader workers is both slow and a source of intermittent
    # read errors. Re-downloading two gigabytes after a disconnect costs about
    # two minutes; the checkpoints that cost hours are the ones kept on Drive.
    hf_cache: str = "/content/hf"
    workers: int = 8

    @property
    def paths(self):
        return {k: os.path.join(self.root, k)
                for k in ("cache", "splits", "ckpt", "results", "figures")}

    @property
    def cache_dir(self):
        """Where datasets are downloaded; local disk unless told otherwise."""
        if self.hf_cache and os.path.isdir(os.path.dirname(self.hf_cache)):
            return self.hf_cache
        return os.path.join(self.root, "hf")      # laptops, and anywhere else


CFG = Config()


def prepare_dirs(cfg=CFG):
    for path in cfg.paths.values():
        os.makedirs(path, exist_ok=True)
    return cfg.paths


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# %% CELL: label bridge

# PlantDoc labels each annotated leaf with its own vocabulary; every one maps
# onto a single PlantVillage class. Two readings a reviewer may query: "Corn
# leaf blight" is PlantDoc's name for Northern Leaf Blight, and a bare crop
# name ("Apple leaf") means a healthy leaf of that crop.
PLANTDOC_TO_PV = {
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
    "Potato leaf": "Potato___healthy",
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


# %% CELL: data


def load_plantvillage(cfg=CFG, log=print):
    """PlantVillage from the Hub, minus its background class.

    Returns the dataset and the class names it is labelled with.
    """
    from datasets import load_dataset

    data = load_dataset(cfg.pv_dataset, split="train",
                        cache_dir=cfg.cache_dir)
    names = data.features["label"].names
    drop = {names.index(n) for n in cfg.drop_classes if n in names}
    if drop:
        log(f"  dropping {sorted(cfg.drop_classes)}: "
            f"{sum(1 for l in data['label'] if l in drop)} images")
        data = data.filter(lambda row: row["label"] not in drop,
                           num_proc=min(4, os.cpu_count() or 1))
    keep = [n for i, n in enumerate(names) if i not in drop]
    remap = {names.index(n): i for i, n in enumerate(keep)}
    data = data.map(lambda row: {"label": remap[row["label"]]},
                    num_proc=min(4, os.cpu_count() or 1))
    log(f"  PlantVillage: {len(data)} images, {len(keep)} classes")
    return data, keep


def load_plantdoc(pv_classes, cfg=CFG, log=print):
    """PlantDoc field photographs, relabelled into PlantVillage's class space.

    The published release annotates each *leaf* in a photograph, not the
    photograph, so an image-level label is the category its annotations agree
    on. Twenty images carry two different categories: whatever a model answers
    for those, one annotation calls it wrong, so they are dropped rather than
    scored. They are PlantDoc's ambiguity, not a model's mistake.
    """
    from datasets import concatenate_datasets, load_dataset

    parts = load_dataset(cfg.pd_dataset, cache_dir=cfg.cache_dir)
    data = concatenate_datasets([parts[s] for s in sorted(parts)])
    index = {n: i for i, n in enumerate(pv_classes)}

    def label_of(row):
        categories = set(row["objects"]["category"])
        if len(categories) != 1:
            return {"label": -1}
        name = categories.pop()
        if name not in PLANTDOC_TO_PV:
            raise ValueError(f"PlantDoc category with no PlantVillage mapping: {name!r}")
        return {"label": index[PLANTDOC_TO_PV[name]]}

    data = data.map(label_of, remove_columns=["objects", "image_id", "width", "height"])
    ambiguous = sum(1 for v in data["label"] if v < 0)
    data = data.filter(lambda row: row["label"] >= 0)
    reachable = sorted(set(data["label"]))
    log(f"  PlantDoc: {len(data)} field images ({ambiguous} dropped for carrying two "
        f"labels), covering {len(reachable)} of {len(pv_classes)} PlantVillage classes")
    return data, reachable


def dhash(image, size=8):
    """A 64-bit perceptual hash: near-identical photographs collide.

    Rotations and crops of one leaf are what PlantVillage is full of, and a
    random split would scatter them across train and test. Grouping on this
    keeps every copy of a leaf on one side of the split.
    """
    small = image.convert("L").resize((size + 1, size), 3)
    pixels = np.asarray(small, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2).to_bytes(8, "big").hex()


def group_hashes(data, cfg=CFG, log=print):
    """A perceptual hash per image, cached -- it costs a decode of everything."""
    cache = os.path.join(cfg.paths["splits"], "hashes.json")
    if os.path.isfile(cache):
        with open(cache, encoding="utf-8") as fh:
            hashes = json.load(fh)
        if len(hashes) == len(data):
            log(f"  hashes: reusing {len(hashes)} from cache")
            return hashes
    started, hashes = time.time(), []
    for i, row in enumerate(data):
        hashes.append(dhash(row["image"]))
        if i and i % 10000 == 0:
            log(f"    hashed {i}/{len(data)}")
    with open(cache, "w", encoding="utf-8") as fh:
        json.dump(hashes, fh)
    log(f"  hashes: {len(hashes)} in {time.time() - started:.0f}s "
        f"({len(set(hashes))} distinct -> {len(hashes) - len(set(hashes))} near-duplicates)")
    return hashes


def grouped_split(labels, hashes, cfg=CFG, log=print):
    """Train/val/test by whole hash-group, stratified on the group's class.

    Returns three index arrays. Because a group never straddles a split, a
    near-duplicate cannot leak from train into test.
    """
    cache = os.path.join(cfg.paths["splits"], "split.json")
    if os.path.isfile(cache):
        with open(cache, encoding="utf-8") as fh:
            saved = json.load(fh)
        if saved["n"] == len(labels):
            log("  split: reusing the frozen one")
            return {k: np.array(v) for k, v in saved["idx"].items()}

    from sklearn.model_selection import train_test_split

    groups = {}
    for i, h in enumerate(hashes):
        groups.setdefault(h, []).append(i)
    keys = sorted(groups)
    # a group's class is its first member's: duplicates share a label
    gl = [labels[groups[k][0]] for k in keys]
    rest, test = train_test_split(range(len(keys)), test_size=cfg.test_frac,
                                  stratify=gl, random_state=cfg.seed)
    val_share = cfg.val_frac / (1 - cfg.test_frac)
    train, val = train_test_split(rest, test_size=val_share,
                                  stratify=[gl[i] for i in rest], random_state=cfg.seed)
    out = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        out[name] = np.array(sorted(i for g in part for i in groups[keys[g]]))
    with open(cache, "w", encoding="utf-8") as fh:
        json.dump({"n": len(labels), "idx": {k: v.tolist() for k, v in out.items()}}, fh)
    log(f"  split: train {len(out['train'])}, val {len(out['val'])}, test {len(out['test'])}"
        f"  (by group, so no leaf spans two splits)")
    return out


def check_no_leakage(splits, hashes, log=print):
    """The split's own guarantee, verified rather than assumed."""
    seen = {name: {hashes[i] for i in idx} for name, idx in splits.items()}
    bad = 0
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = seen[a] & seen[b]
        if shared:
            bad += len(shared)
            log(f"  LEAKAGE: {len(shared)} perceptual hashes shared by {a} and {b}")
    total = sum(len(v) for v in splits.values())
    if not bad:
        log(f"  no leakage: {total} images, no perceptual hash in two splits")
    return bad == 0


# %% CELL: torch datasets


class HFImages(torch.utils.data.Dataset):
    """A HuggingFace image split, decoded and transformed on the fly."""

    def __init__(self, data, indices, transform):
        self.data, self.indices, self.transform = data, indices, transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = self.data[int(self.indices[i])]
        return self.transform(row["image"].convert("RGB")), int(row["label"])


def transforms_for(cfg=CFG, train=False):
    from torchvision import transforms

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(cfg.img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.RandomRotation(30)], p=0.5),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])
    return transforms.Compose([
        transforms.Resize(int(cfg.img_size * 1.14)),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def loader_for(data, indices, cfg=CFG, train=False, batch_size=None):
    return torch.utils.data.DataLoader(
        HFImages(data, indices, transforms_for(cfg, train)),
        batch_size=batch_size or cfg.batch_size, shuffle=train,
        num_workers=cfg.workers, pin_memory=True, drop_last=False,
        persistent_workers=cfg.workers > 0)


# %% CELL: models


class Baseline(nn.Module):
    """A timm backbone with one dropout + linear head: the three published arms."""

    def __init__(self, backbone, num_classes, drop_rate=0.2, pretrained=True):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        width = self.backbone.num_features
        self.head = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(width, num_classes))

    def forward(self, x):
        return self.head(self.backbone(x))


class AGFT(nn.Module):
    """Adaptive Gated Fine-Tuning: a frozen branch, a tuned branch, a gate.

    `mode` selects the arm, so every ablation is this one class:
        'gated'   the method -- the gate is learned per image
        'fixed'   one constant alpha, learned but input-independent
                  (a WiSE-FT-style trade-off, in output space)
        'ft'      alpha pinned to 1: plain fine-tuning
        'frozen'  alpha pinned to 0: a linear probe
    """

    MODES = ("gated", "fixed", "ft", "frozen")

    def __init__(self, backbone, num_classes, drop_rate=0.2, mode="gated", pretrained=True):
        super().__init__()
        import timm
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode
        self.tuned = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        width = self.tuned.num_features
        self.head_tuned = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(width, num_classes))

        if mode in ("gated", "fixed", "frozen"):
            self.frozen = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            for p in self.frozen.parameters():
                p.requires_grad = False
            self.frozen.eval()
            self.head_frozen = nn.Sequential(nn.Dropout(drop_rate),
                                             nn.Linear(width, num_classes))
        if mode == "gated":
            # reads the frozen features: the one description of the input that
            # does not drift while the other branch is being trained
            self.gate = nn.Sequential(nn.Linear(width, width // 8), nn.GELU(),
                                      nn.Linear(width // 8, 1))
        if mode == "fixed":
            self.alpha = nn.Parameter(torch.zeros(1))      # sigmoid(0) = 0.5

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "frozen"):
            self.frozen.eval()           # frozen means frozen, batch stats included
        return self

    def gate_value(self, frozen_features):
        if self.mode == "gated":
            return torch.sigmoid(self.gate(frozen_features))
        if self.mode == "fixed":
            return torch.sigmoid(self.alpha).expand(frozen_features.shape[0], 1)
        return torch.full((frozen_features.shape[0], 1), 1.0 if self.mode == "ft" else 0.0,
                          device=frozen_features.device)

    def forward(self, x, return_gate=False):
        if self.mode == "ft":
            logits = self.head_tuned(self.tuned(x))
            return (logits, None) if return_gate else logits
        with torch.no_grad():
            f_frozen = self.frozen(x)
        logits_frozen = self.head_frozen(f_frozen)
        if self.mode == "frozen":
            return (logits_frozen, None) if return_gate else logits_frozen
        g = self.gate_value(f_frozen)
        logits = g * self.head_tuned(self.tuned(x)) + (1 - g) * logits_frozen
        return (logits, g) if return_gate else logits


def build_model(name, num_classes, cfg=CFG, drop_rate=None, pretrained=True):
    """`name` is a timm backbone, or 'agft:<mode>' for one of the gated arms."""
    drop_rate = cfg.drop_rate if drop_rate is None else drop_rate
    if name.startswith("agft"):
        mode = name.split(":", 1)[1] if ":" in name else "gated"
        return AGFT(cfg.agft_backbone, num_classes, drop_rate, mode, pretrained)
    return Baseline(name, num_classes, drop_rate, pretrained)


# %% CELL: train and evaluate


@torch.no_grad()
def evaluate(model, loader, device="cuda", restrict=None):
    """Accuracy, macro-F1 and the mean gate, on one loader."""
    from sklearn.metrics import f1_score

    model.eval()
    preds, trues, gates = [], [], []
    for images, y in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            out = model(images, return_gate=True) if isinstance(model, AGFT) else (model(images), None)
        logits, g = out
        logits = logits.float()
        if restrict is not None:
            masked = torch.full_like(logits, float("-inf"))
            masked[:, restrict] = logits[:, restrict]
            logits = masked
        preds.append(logits.argmax(1).cpu())
        trues.append(y)
        if g is not None:
            gates.append(g.float().mean().item())
    preds, trues = torch.cat(preds), torch.cat(trues)
    return {"accuracy": (preds == trues).float().mean().item(),
            "macro_f1": f1_score(trues, preds, average="macro", zero_division=0),
            "gate": float(np.mean(gates)) if gates else None,
            "n": int(len(trues))}


def train_model(name, data, train_idx, val_idx, num_classes, cfg=CFG, tag=None,
                epochs=None, lr=None, weight_decay=None, drop_rate=None,
                trial=None, log=print):
    """Train one arm with early stopping, resuming from its checkpoint.

    Returns the best validation metrics and the path of the best weights.
    Interrupting and re-running continues from the last finished epoch, which
    is what makes a dropped Colab session cost minutes rather than hours.
    """
    tag = tag or name.replace(":", "_").replace("/", "_").replace(".", "_")
    run_dir = os.path.join(cfg.paths["ckpt"], tag)
    os.makedirs(run_dir, exist_ok=True)
    last_path, best_path = os.path.join(run_dir, "last.pt"), os.path.join(run_dir, "best.pt")
    done_path = os.path.join(run_dir, "done.json")
    if os.path.isfile(done_path):
        with open(done_path, encoding="utf-8") as fh:
            saved = json.load(fh)
        log(f"  {tag}: already finished (val macro-F1 {saved['macro_f1']:.4f}) -- skipped")
        return saved, best_path

    epochs = epochs or cfg.epochs
    lr = lr or cfg.lr
    weight_decay = cfg.weight_decay if weight_decay is None else weight_decay
    seed_everything(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(name, num_classes, cfg, drop_rate).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    train_loader = loader_for(data, train_idx, cfg, train=True)
    val_loader = loader_for(data, val_idx, cfg, train=False)
    steps = max(1, len(train_loader))
    # Warmup then cosine, as a plain function of the global step. A stateful
    # schedule such as OneCycleLR stores its own total_steps in the checkpoint
    # and restores it on resume, so continuing a run with a different epoch
    # budget makes it step past its own end and raise. Deriving the rate from
    # the step count instead means a resumed run cannot desynchronise, and the
    # schedule is rebuilt rather than restored.
    def lr_at(step):
        warm = max(1, int(cfg.warmup_epochs * steps))
        if step < warm:
            return (step + 1) / warm
        progress = (step - warm) / max(1, epochs * steps - warm)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_at)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    start, best, bad, history = 0, {"macro_f1": -1.0}, 0, []
    if os.path.isfile(last_path):
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        scaler.load_state_dict(state["scaler"])
        start, best, bad, history = state["epoch"], state["best"], state["bad"], state["history"]
        for _ in range(start * steps):          # wind the schedule to where it was
            schedule.step()
        log(f"  {tag}: resuming at epoch {start + 1}")

    for epoch in range(start, epochs):
        model.train()
        started, total, seen = time.time(), 0.0, 0
        for images, y in train_loader:
            images, y = images.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = criterion(model(images), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            total += loss.item() * len(y)
            seen += len(y)
        metrics = evaluate(model, val_loader, device)
        metrics.update(epoch=epoch + 1, train_loss=total / max(seen, 1),
                       seconds=round(time.time() - started, 1))
        history.append(metrics)
        improved = metrics["macro_f1"] > best["macro_f1"]
        log(f"    epoch {epoch + 1}/{epochs}  loss {metrics['train_loss']:.4f}  "
            f"val acc {metrics['accuracy']:.4f}  macro-F1 {metrics['macro_f1']:.4f}"
            + (f"  gate {metrics['gate']:.3f}" if metrics["gate"] is not None else "")
            + ("  *" if improved else ""))
        if improved:
            best, bad = dict(metrics), 0
            torch.save({"model": model.state_dict(), "metrics": best}, best_path)
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "optimiser": optimiser.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch + 1, "best": best, "bad": bad, "history": history},
                   last_path)
        if trial is not None:
            trial.report(metrics["macro_f1"], epoch)
            import optuna
            if trial.should_prune():
                raise optuna.TrialPruned()
        if bad >= cfg.patience:
            log(f"    early stop: no improvement for {cfg.patience} epochs")
            break

    best["history"] = history
    with open(done_path, "w", encoding="utf-8") as fh:
        json.dump(best, fh, indent=2)
    return best, best_path
