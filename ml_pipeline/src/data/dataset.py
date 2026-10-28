"""
PyTorch Dataset over the frozen split CSVs.

Every experiment -- the three baselines, each ablation, the distilled student --
loads its data through here, so they all see exactly the same images. If the
comparison table in the paper is to mean anything, nothing may bypass this.

Usage:
    from dataset import build_loaders
    train_loader, val_loader, test_loader = build_loaders(batch_size=16, img_size=224)
"""

import csv
import os
import sys

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classes import NUM_CLASSES  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SPLIT_DIR = os.path.join(REPO_ROOT, "data", "splits")

# ImageNet statistics -- all four backbones are ImageNet-pretrained
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class PlantVillageSplit(Dataset):
    """Reads one plantvillage_{split}.csv and serves (image, label) pairs.

    Also carries expert_idx per sample, which the MoE load-balancing loss and
    the routing-heatmap figure need in Phase 3.
    """

    def __init__(self, split, transform=None, return_expert=False, dataset="plantvillage"):
        csv_path = os.path.join(SPLIT_DIR, f"{dataset}_{split}.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"{csv_path} not found. Run:\n"
                f"  python ml_pipeline/src/data/build_splits.py"
            )

        with open(csv_path, newline="", encoding="utf-8") as fh:
            self.rows = list(csv.DictReader(fh))

        self.split = split
        self.transform = transform
        self.return_expert = return_expert

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        path = os.path.join(REPO_ROOT, row["path"])
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(row["class_idx"])
        if self.return_expert:
            return image, label, int(row["expert_idx"])
        return image, label

    def class_counts(self):
        """Per-class sample counts -- used for class-weighted loss and for the
        dataset table in the paper."""
        counts = [0] * NUM_CLASSES
        for row in self.rows:
            counts[int(row["class_idx"])] += 1
        return counts


def train_transform(img_size=224):
    """Augmentation for training.

    Deliberately strong: PlantVillage is lab-captured with uniform backgrounds,
    so without heavy augmentation the model latches onto background cues and
    collapses on field images (the PlantDoc failure mode). Mixup/CutMix are
    applied in the training loop, not here, because they need the labels.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomRotation(30)], p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def eval_transform(img_size=224):
    """Deterministic transform for val/test. Resize short side then centre-crop,
    the standard ImageNet evaluation protocol."""
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def build_loaders(batch_size=16, img_size=224, num_workers=4, return_expert=False,
                  pin_memory=True):
    """Build train/val/test loaders with the standard transforms.

    Defaults target a 6 GB card: batch 16 at 224px with AMP fits Swin-V2-T.
    """
    train_ds = PlantVillageSplit("train", train_transform(img_size), return_expert)
    val_ds = PlantVillageSplit("val", eval_transform(img_size), return_expert)
    test_ds = PlantVillageSplit("test", eval_transform(img_size), return_expert)

    common = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=pin_memory, persistent_workers=num_workers > 0)

    return (
        DataLoader(train_ds, shuffle=True, drop_last=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
        DataLoader(test_ds, shuffle=False, **common),
    )


if __name__ == "__main__":
    print("Loading splits...")
    for split in ("train", "val", "test"):
        try:
            ds = PlantVillageSplit(split, eval_transform())
        except FileNotFoundError as exc:
            print(exc)
            raise SystemExit(1)

        counts = ds.class_counts()
        empty = sum(1 for c in counts if c == 0)
        print(f"  {split:<6} {len(ds):>7} images   "
              f"min/class={min(counts):<5} max/class={max(counts):<6} "
              f"empty classes={empty}")

    print("\nFetching one batch...")
    loader, _, _ = build_loaders(batch_size=4, num_workers=0)
    images, labels = next(iter(loader))
    print(f"  images {tuple(images.shape)}  {images.dtype}")
    print(f"  labels {labels.tolist()}")
    print(f"  value range [{images.min():.2f}, {images.max():.2f}] (normalised)")
    assert images.shape[1:] == (3, 224, 224)
    assert labels.max() < NUM_CLASSES
    print("\nOK: dataset and loaders work.")
