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
  that result exists, and a run in progress leaves a resume point there, so a
  disconnected Colab session resumes by re-running.
"""

# %% CELL: imports
import hashlib
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

# %% CELL: config


@dataclass
class Config:
    """Everything the study depends on, in one place."""

    # data, pinned to exact commits: the cached perceptual hashes and the
    # frozen split are stored by position, so an upstream edit between two
    # sessions would silently misalign them
    pv_dataset: str = "DScomp380/plant_village"
    pv_revision: str = "5ce680f815ea9fab7b6f8346ae4c71e7099696a5"
    pd_dataset: str = "agyaatcoder/PlantDoc"
    pd_revision: str = "ace645742b6af1b0df9aaefc2c9b0c988e141814"
    drop_classes: tuple = ("Background_without_leaves",)
    # Two photographs are near-duplicates when their 64-bit perceptual hashes
    # differ in at most this many bits. Measured on PlantVillage over all 1.5
    # billion pairs: exact matching (0) links 46 images; 4 links 883 in groups
    # of at most 11; at 6, chains of pairwise matches merge 820 images into
    # one group, which would distort the stratified split. 4 is the widest
    # net before chaining sets in.
    dup_bits: int = 4
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

    # Two places to keep things, chosen per file:
    #
    #   root   Drive. Durable but slow, quota-limited, and deleting there only
    #          moves a file to Trash, which still counts against the quota.
    #          Holds results, the split, and one resume point -- small or
    #          rarely written.
    #   local  The runtime's own disk. Fast and free, gone after a disconnect.
    #          Holds the datasets (read at random by eight workers, which a
    #          FUSE mount does badly), live checkpoints written every epoch,
    #          and Optuna's SQLite file (SQLite on a network mount fails with
    #          "disk I/O error").
    root: str = "/content/drive/MyDrive/agft"
    local: str = "/content/agft_local"
    hf_cache: str = ""             # override the dataset cache; empty = under local
    # How often a run in progress copies its resume point to Drive. Every
    # epoch would rewrite ~350 MB every three minutes for nineteen hours;
    # fifteen minutes bounds what a disconnect can cost while keeping Drive
    # traffic small. Lower it if you have Drive space to spare.
    mirror_minutes: float = 15.0
    workers: int = 8

    @property
    def paths(self):
        return {k: os.path.join(self.root, k)
                for k in ("splits", "runs", "results", "figures", "models", "optuna")}

    @property
    def local_root(self):
        """The runtime's disk on Colab; a folder beside the results elsewhere."""
        if self.local and os.path.isdir(os.path.dirname(self.local)):
            return self.local
        return os.path.join(self.root, "_local")

    @property
    def cache_dir(self):
        return self.hf_cache or os.path.join(self.local_root, "hf")


CFG = Config()


def prepare_dirs(cfg=CFG):
    for path in list(cfg.paths.values()) + [cfg.local_root]:
        os.makedirs(path, exist_ok=True)
    return cfg.paths


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# %% CELL: storage that survives a disconnect
#
# A Colab disconnect can land in the middle of any write. Every file here is
# written so that it is either the old version or the new one, never half of
# each, and every reader treats an unreadable file as absent -- so the stage
# that produced it simply runs again -- rather than stopping the study.


def write_json(obj, path, attempts=3):
    """Write atomically: a temporary file, then a rename over the target."""
    tmp = f"{path}.tmp"
    for attempt in range(attempts):
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(obj, fh, indent=2)
            os.replace(tmp, path)
            return
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))       # Drive's mount has transient errors


def read_json(path, attempts=3):
    """The file's contents; None if it is missing or was cut off mid-write.

    A transient read error is retried rather than treated as missing: calling
    a finished stage missing would quietly re-run hours of work.
    """
    for attempt in range(attempts):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))


def save_atomic(obj, path):
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def copy_atomic(src, dst):
    tmp = f"{dst}.tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def try_load(path):
    """A checkpoint, or None if it is missing or unreadable."""
    if not os.path.isfile(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:                                # noqa: BLE001 -- torn file
        return None


# %% CELL: label bridge

# PlantDoc labels each annotated leaf with its own vocabulary; every one maps
# onto a single PlantVillage class. Two readings a reviewer may query: "Corn
# leaf blight" is PlantDoc's name for Northern Leaf Blight, and a bare crop
# name ("Apple leaf") means a healthy leaf of that crop. Targets are the class
# names of the pinned PlantVillage release, which shortens the original
# folder names ("Corn___Common_rust", not "Corn_(maize)___Common_rust_").
PLANTDOC_TO_PV = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry___healthy",
    "Corn Gray leaf spot": "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn___Common_rust",
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

    data = load_dataset(cfg.pv_dataset, split="train", revision=cfg.pv_revision,
                        cache_dir=cfg.cache_dir)
    names = data.features["label"].names
    drop = {names.index(n) for n in cfg.drop_classes if n in names}
    keep = [n for i, n in enumerate(names) if i not in drop]
    remap = {names.index(n): i for i, n in enumerate(keep)}
    # Both passes read the label column alone. Without input_columns each one
    # would decode all 55,000 photographs just to look at an integer.
    if drop:
        log(f"  dropping {sorted(cfg.drop_classes)}: "
            f"{sum(1 for v in data['label'] if v in drop)} images")
        data = data.filter(lambda labels: [v not in drop for v in labels],
                           input_columns="label", batched=True)
    data = data.map(lambda labels: {"label": [remap[v] for v in labels]},
                    input_columns="label", batched=True)
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

    index = {n: i for i, n in enumerate(pv_classes)}
    missing = sorted({v for v in PLANTDOC_TO_PV.values() if v not in index})
    if missing:
        raise ValueError(f"the PlantDoc bridge names classes PlantVillage does not have: "
                         f"{missing}")
    parts = load_dataset(cfg.pd_dataset, revision=cfg.pd_revision, cache_dir=cfg.cache_dir)
    data = concatenate_datasets([parts[s] for s in sorted(parts)])

    def label_of(objects):
        categories = set(objects["category"])
        if len(categories) != 1:
            return {"label": -1}
        name = categories.pop()
        if name not in PLANTDOC_TO_PV:
            raise ValueError(f"PlantDoc category with no PlantVillage mapping: {name!r}")
        return {"label": index[PLANTDOC_TO_PV[name]]}

    data = data.map(label_of, input_columns="objects",
                    remove_columns=["objects", "image_id", "width", "height"])
    ambiguous = sum(1 for v in data["label"] if v < 0)
    data = data.filter(lambda labels: [v >= 0 for v in labels],
                       input_columns="label", batched=True)
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
    hashes = read_json(cache)
    if hashes is not None and len(hashes) == len(data):
        log(f"  hashes: reusing {len(hashes)} from cache")
        return hashes
    started, hashes = time.time(), []
    for i, row in enumerate(data.select_columns(["image"])):
        hashes.append(dhash(row["image"]))
        if i and i % 10000 == 0:
            log(f"    hashed {i}/{len(data)}")
    write_json(hashes, cache)
    log(f"  hashes: {len(hashes)} in {time.time() - started:.0f}s "
        f"({len(set(hashes))} distinct -> {len(hashes) - len(set(hashes))} near-duplicates)")
    return hashes


_BYTE_BITS = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount_table(x):
    """Set bits per element of a uint64 array, by byte lookup (any numpy)."""
    x = np.ascontiguousarray(x)
    return _BYTE_BITS[x.view(np.uint8)].reshape(*x.shape, 8).sum(-1)


def _popcount(x):
    """Set bits per element of a uint64 array."""
    if hasattr(np, "bitwise_count"):                     # numpy >= 2.0
        return np.bitwise_count(x)
    return _popcount_table(x)


def _as_bits(hashes):
    return np.array([int(h, 16) for h in hashes], dtype=np.uint64)


def near_duplicate_pairs(hashes, max_bits=6, chunk=500, log=print):
    """Every pair of images whose hashes differ in at most `max_bits` bits.

    All n(n-1)/2 pairs are compared exactly -- about 1.5 billion for
    PlantVillage, some twenty seconds -- so no near-duplicate is missed.
    Returns (i, j, bits) triples with i < j.
    """
    started, h, found = time.time(), _as_bits(hashes), []
    for start in range(0, len(h), chunk):
        d = _popcount(h[start:start + chunk, None] ^ h[None, :])
        i, j = np.nonzero(d <= max_bits)
        keep = j > i + start
        i, j = i[keep], j[keep]
        found.extend(zip((i + start).tolist(), j.tolist(), d[i, j].tolist()))
    log(f"  compared {len(h) * (len(h) - 1) // 2:,} pairs in {time.time() - started:.0f}s: "
        f"{len(found)} within {max_bits} bits")
    return found


def group_ids(n, pairs, bits):
    """Images joined by any chain of near-duplicate pairs share one group.

    A group's id is its smallest member's index, so the grouping -- and
    everything drawn from it, down to the CV folds -- is identical in every
    session.
    """
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, d in pairs:
        if d <= bits:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)      # the root stays the minimum
    return [str(find(x)) for x in range(n)]


def grouped_split(labels, groups, cfg=CFG, log=print):
    """Train/val/test by whole near-duplicate group, stratified on class.

    Returns three index arrays. Because a group never straddles a split, a
    near-duplicate cannot leak from train into test. The frozen split is
    reused only while the grouping it was made from is unchanged.
    """
    cache = os.path.join(cfg.paths["splits"], "split.json")
    fingerprint = hashlib.sha1(",".join(groups).encode()).hexdigest()[:16]
    saved = read_json(cache)
    if saved is not None and saved["n"] == len(labels) and saved.get("groups") == fingerprint:
        log("  split: reusing the frozen one")
        return {k: np.array(v) for k, v in saved["idx"].items()}

    from sklearn.model_selection import train_test_split

    members = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    keys = sorted(members)
    # Stratify on each group's first member. A group can mix classes (two
    # diseases photographed alike); that only affects the stratification
    # target, never which split a member lands in -- all of it goes together.
    gl = [labels[members[k][0]] for k in keys]
    rest, test = train_test_split(range(len(keys)), test_size=cfg.test_frac,
                                  stratify=gl, random_state=cfg.seed)
    val_share = cfg.val_frac / (1 - cfg.test_frac)
    train, val = train_test_split(rest, test_size=val_share,
                                  stratify=[gl[i] for i in rest], random_state=cfg.seed)
    out = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        out[name] = np.array(sorted(i for g in part for i in members[keys[g]]))
    write_json({"n": len(labels), "groups": fingerprint,
                "idx": {k: v.tolist() for k, v in out.items()}}, cache)
    log(f"  split: train {len(out['train'])}, val {len(out['val'])}, test {len(out['test'])}"
        f"  (by group, so no leaf spans two splits)")
    return out


def check_no_leakage(splits, hashes, bits=0, chunk=500, log=print):
    """The split's guarantee, verified pair by pair rather than assumed.

    Independent of how the groups were built: every image in one split is
    compared with every image in the others, and any two whose hashes are
    within `bits` of each other count as leakage.
    """
    h, bad = _as_bits(hashes), 0
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        ha, hb = h[np.asarray(splits[a])], h[np.asarray(splits[b])]
        crossing = sum(int((_popcount(ha[s:s + chunk, None] ^ hb[None, :]) <= bits).sum())
                       for s in range(0, len(ha), chunk))
        if crossing:
            bad += crossing
            log(f"  LEAKAGE: {crossing} near-duplicate pairs span {a} and {b}")
    if not bad:
        total = sum(len(v) for v in splits.values())
        log(f"  no leakage: {total} images, and no two in different splits are "
            f"within {bits} bits of each other")
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


def create_backbone(name, img_size=224, pretrained=True):
    """A timm backbone that returns pooled features at `img_size`.

    Swin-V2 fixes its input size and attention window when it is built. Its
    published weights expect 256 px with an 8x8 window, so at 224 px it raises
    on the very first batch -- and an 8x8 window cannot tile the 28x28 stage
    either. Swin-V2's continuous position bias was designed to transfer across
    window sizes, so it is rebuilt for the study's resolution with a window
    that tiles every stage: img_size / 32, which is 7 at 224 px.
    """
    import timm

    kwargs = {}
    if name.startswith("swinv2"):
        if img_size % 32:
            raise ValueError(f"Swin-V2 needs an image size divisible by 32, got {img_size}")
        kwargs = {"img_size": img_size, "window_size": img_size // 32}
    return timm.create_model(name, pretrained=pretrained, num_classes=0, **kwargs)


def feature_width(net, img_size):
    """The width of what the backbone actually returns, measured.

    Not always `num_features`: MobileNetV4 reports 960 there, then pools
    through a 1x1 convolutional head to 1280, so a head sized from the
    attribute fails on the first batch. Measured in eval mode so the probe
    leaves BatchNorm's running statistics untouched.
    """
    was_training = net.training
    net.eval()
    with torch.no_grad():
        width = net(torch.zeros(1, 3, img_size, img_size)).shape[-1]
    net.train(was_training)
    return int(width)


class Baseline(nn.Module):
    """A timm backbone with one dropout + linear head: the three published arms."""

    def __init__(self, backbone, num_classes, drop_rate=0.2, pretrained=True, img_size=224):
        super().__init__()
        self.backbone = create_backbone(backbone, img_size, pretrained)
        width = feature_width(self.backbone, img_size)
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

    def __init__(self, backbone, num_classes, drop_rate=0.2, mode="gated", pretrained=True,
                 img_size=224):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode
        self.tuned = create_backbone(backbone, img_size, pretrained)
        width = feature_width(self.tuned, img_size)
        self.head_tuned = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(width, num_classes))

        if mode in ("gated", "fixed", "frozen"):
            self.frozen = create_backbone(backbone, img_size, pretrained)
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
        return AGFT(cfg.agft_backbone, num_classes, drop_rate, mode, pretrained, cfg.img_size)
    return Baseline(name, num_classes, drop_rate, pretrained, cfg.img_size)


def model_state(model):
    """Everything training changes. The frozen branch is the pretrained weights
    by definition, so storing it would double every AGFT checkpoint for
    nothing; building the model rebuilds it from timm's cache."""
    return {k: v for k, v in model.state_dict().items() if not k.startswith("frozen.")}


def load_model_state(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith("frozen.")]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint does not fit this model: missing {missing[:4]}, "
                           f"unexpected {unexpected[:4]}")


def amp_dtype(device):
    """bfloat16 where the GPU has it (L4, A100): fp16's speed without its
    narrow range, which is what overflows in attention. fp16 elsewhere."""
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


# %% CELL: train and evaluate


@torch.no_grad()
def evaluate(model, loader, device="cuda", restrict=None):
    """Accuracy, macro-F1 and the mean gate, on one loader."""
    from sklearn.metrics import f1_score

    model.eval()
    dtype = amp_dtype(device)
    preds, trues, gate_sum, gate_n = [], [], 0.0, 0
    for images, y in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=dtype, enabled=device == "cuda"):
            out = model(images, return_gate=True) if isinstance(model, AGFT) else (model(images), None)
        logits, g = out
        logits = logits.float()
        if restrict is not None:
            idx = restrict.to(logits.device)
            masked = torch.full_like(logits, float("-inf"))
            masked[:, idx] = logits[:, idx]
            logits = masked
        preds.append(logits.argmax(1).cpu())
        trues.append(y)
        if g is not None:
            gate_sum += g.float().sum().item()
            gate_n += g.numel()
    preds, trues = torch.cat(preds), torch.cat(trues)
    return {"accuracy": (preds == trues).float().mean().item(),
            "macro_f1": float(f1_score(trues, preds, average="macro", zero_division=0)),
            "gate": gate_sum / gate_n if gate_n else None,
            "n": int(len(trues))}


class Diverged(RuntimeError):
    """A run whose loss went non-finite before any epoch finished."""


def _resume_index(cfg):
    return os.path.join(cfg.paths["runs"], "resume.json")


def _slot_path(cfg, slot):
    return os.path.join(cfg.paths["runs"], f"resume_{slot}.pt")


def _mirror(tag, payload, cfg):
    """Write this run's resume point to Drive, into one of two rolling slots.

    Two slots, because a disconnect during the write tears only the file being
    written and the other is still whole. Overwritten in place and never
    deleted, because a file deleted on Drive moves to Trash and goes on
    counting against the quota. Runs are sequential, so a slot that does not
    hold this run belongs to one that has finished and is free to reuse.

    The Drive copy is lean: weights and progress, without the optimiser's
    state, which is twice the size of the weights. It is only read after a
    runtime is lost; then AdamW restarts its moment estimates while the
    learning-rate schedule continues exactly. A run interrupted on a live
    runtime resumes from local disk with everything.
    """
    index = read_json(_resume_index(cfg)) or {}
    mine = {int(k): v["epoch"] for k, v in index.items() if v.get("tag") == tag}
    slot = (1 - max(mine, key=mine.get)) if mine else 0
    lean = {k: v for k, v in payload.items() if k not in ("optimiser", "scaler")}
    torch.save(lean, _slot_path(cfg, slot))
    index[str(slot)] = {"tag": tag, "epoch": payload["epoch"]}
    write_json(index, _resume_index(cfg))


def _restore(tag, last_path, best_path, cfg, log):
    """This run's newest readable resume point: local disk, then Drive."""
    state = try_load(last_path)
    if state is not None and state.get("tag") == tag:
        return state, "local disk"
    if os.path.isfile(last_path):
        log(f"  {tag}: local checkpoint unreadable -- trying Drive")
    index = read_json(_resume_index(cfg)) or {}
    for _, slot in sorted(((v["epoch"], int(k)) for k, v in index.items()
                           if v.get("tag") == tag), reverse=True):
        state = try_load(_slot_path(cfg, slot))
        if state is not None and state.get("tag") == tag:
            # the best weights travel inside the resume point; None means the
            # checkpointed epoch was itself the best one
            best_weights = state["model"] if state["best_model"] is None else state["best_model"]
            save_atomic({"model": best_weights, "metrics": state["best"]}, best_path)
            return state, "Drive"
        log(f"  {tag}: Drive copy {slot} unreadable -- trying the other")
    return None, None


def train_model(name, data, train_idx, val_idx, num_classes, cfg=CFG, tag=None,
                epochs=None, lr=None, weight_decay=None, drop_rate=None,
                trial=None, keep_best=False, resumable=True, log=print):
    """Train one run with early stopping; a disconnect costs minutes.

    Returns (best validation metrics, path of the best weights on Drive if
    keep_best, else None).

    Where each thing is written, and why:
      * every epoch, the full resume point to local disk -- fast, atomic, and
        free of Drive quota;
      * every cfg.mirror_minutes, a copy of it to Drive (see _mirror), which
        is what survives a disconnect;
      * at the end, the result as a small marker on Drive, and with keep_best
        the best weights, copied to Drive once.

    resumable=False is for Optuna trials. They are short and never resumed,
    and a trial number can be handed out again after a disconnect, so a
    marker keyed on it could return an old trial's score for new
    hyperparameters.
    """
    tag = tag or name.replace(":", "_").replace("/", "_").replace(".", "_")
    marker = os.path.join(cfg.paths["runs"], f"{tag}.json")
    weights = os.path.join(cfg.paths["models"], f"{tag}.pt")
    if resumable:
        saved = read_json(marker)
        if saved is not None and (not keep_best or os.path.isfile(weights)):
            log(f"  {tag}: already finished (val macro-F1 {saved['macro_f1']:.4f}) -- skipped")
            return saved, (weights if keep_best else None)

    local_dir = os.path.join(cfg.local_root, "runs", tag)
    if not resumable:
        shutil.rmtree(local_dir, ignore_errors=True)
    os.makedirs(local_dir, exist_ok=True)
    last_path = os.path.join(local_dir, "last.pt")
    best_path = os.path.join(local_dir, "best.pt")

    epochs = epochs or cfg.epochs
    lr = lr or cfg.lr
    weight_decay = cfg.weight_decay if weight_decay is None else weight_decay
    seed_everything(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = amp_dtype(device)
    model = build_model(name, num_classes, cfg, drop_rate).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    train_loader = loader_for(data, train_idx, cfg, train=True)
    val_loader = loader_for(data, val_idx, cfg, train=False)
    try:
        steps = max(1, len(train_loader))

        scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and dtype == torch.float16)
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

        start, best, bad, history, best_state = 0, {"macro_f1": -1.0}, 0, [], None
        state, source = _restore(tag, last_path, best_path, cfg, log) if resumable else (None, None)
        if state is not None:
            load_model_state(model, state["model"])
            if "optimiser" in state:
                optimiser.load_state_dict(state["optimiser"])
            if state.get("scaler") and scaler.is_enabled():
                scaler.load_state_dict(state["scaler"])
            start, best, bad, history = state["epoch"], state["best"], state["bad"], state["history"]
            best_state = state["model"] if state["best_model"] is None else state["best_model"]
            log(f"  {tag}: resuming at epoch {start + 1} from {source}"
                + ("" if "optimiser" in state else " (optimiser moments restart)"))

        # Warmup then cosine, as a plain function of the global step. A
        # stateful schedule such as OneCycleLR stores its own total_steps in
        # the checkpoint and restores it, so resuming with a different epoch
        # budget steps past its end and raises. Derived from the step count,
        # a resumed run cannot desynchronise: the schedule is rebuilt at the
        # step the run had reached, not restored.
        def lr_at(step):
            warm = max(1, int(cfg.warmup_epochs * steps))
            if step < warm:
                return (step + 1) / warm
            progress = (step - warm) / max(1, epochs * steps - warm)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        for group in optimiser.param_groups:
            group.setdefault("initial_lr", lr)
        schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_at,
                                                     last_epoch=start * steps - 1)

        last_mirror = time.time()
        for epoch in range(start, epochs):
            model.train()
            started, total, seen, skipped = time.time(), 0.0, 0, 0
            for images, y in train_loader:
                images, y = images.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimiser.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=dtype, enabled=device == "cuda"):
                    loss = criterion(model(images), y)
                if not torch.isfinite(loss):
                    # one bad batch must not write NaN into every weight
                    skipped += 1
                    schedule.step()
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                norm = torch.nn.utils.clip_grad_norm_(trainable, 5.0)
                if scaler.is_enabled() or torch.isfinite(norm):
                    scaler.step(optimiser)          # a scaler skips bad steps itself
                else:
                    skipped += 1
                scaler.update()
                schedule.step()
                total += loss.item() * len(y)
                seen += len(y)
            if skipped > steps // 2:
                log(f"    epoch {epoch + 1}: {skipped}/{steps} batches non-finite -- the run "
                    f"has diverged; stopping it and keeping its best epoch")
                break
            metrics = evaluate(model, val_loader, device)
            metrics.update(epoch=epoch + 1, train_loss=total / max(seen, 1),
                           seconds=round(time.time() - started, 1))
            history.append(metrics)
            improved = metrics["macro_f1"] > best["macro_f1"]
            log(f"    epoch {epoch + 1}/{epochs}  lr {optimiser.param_groups[0]['lr']:.3e}  "
                f"loss {metrics['train_loss']:.4f}  "
                f"val acc {metrics['accuracy']:.4f}  macro-F1 {metrics['macro_f1']:.4f}"
                + (f"  gate {metrics['gate']:.3f}" if metrics["gate"] is not None else "")
                + ("  *" if improved else ""))
            if improved:
                best, bad = dict(metrics), 0
                best_state = {k: v.detach().to("cpu", copy=True)
                              for k, v in model_state(model).items()}
                save_atomic({"model": best_state, "metrics": best}, best_path)
            else:
                bad += 1
            payload = {"tag": tag, "epoch": epoch + 1, "model": model_state(model),
                       "optimiser": optimiser.state_dict(), "scaler": scaler.state_dict(),
                       "best": best, "bad": bad, "history": history,
                       "best_model": None if improved else best_state}
            save_atomic(payload, last_path)
            if resumable and time.time() - last_mirror >= cfg.mirror_minutes * 60:
                _mirror(tag, payload, cfg)
                last_mirror = time.time()
            if trial is not None:
                trial.report(metrics["macro_f1"], epoch)
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()
            if bad >= cfg.patience:
                log(f"    early stop: no improvement for {cfg.patience} epochs")
                break
    except BaseException as exc:
        if type(exc).__name__ == "TrialPruned":
            shutil.rmtree(local_dir, ignore_errors=True)   # a pruned trial never resumes
        raise
    finally:
        del train_loader, val_loader                       # stop the loader workers now

    if best_state is None:
        shutil.rmtree(local_dir, ignore_errors=True)
        raise Diverged(f"{tag}: diverged before finishing an epoch")
    best = dict(best, history=history)
    if keep_best:
        copy_atomic(best_path, weights)        # before the marker: the marker means done
    if resumable:
        write_json(best, marker)
    shutil.rmtree(local_dir, ignore_errors=True)
    return best, (weights if keep_best else None)
