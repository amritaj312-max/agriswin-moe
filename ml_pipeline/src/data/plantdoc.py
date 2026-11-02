"""
PlantDoc, mapped onto the PlantVillage label space.

PlantDoc is 2,589 photographs of leaves in real fields -- varied backgrounds,
daylight, several leaves per frame -- against PlantVillage's single leaf on a
uniform background. It is never trained on. It is the cross-dataset test that
says what a model trained on lab images does when it meets a field image, and
the evidence for the paper's honest claim about domain shift (ACTION_PLAN 3.3):
not "we fix it", but "under domain shift the model knows that it does not know".

Its 28 class directories carry their own names, so they are mapped here onto
the 38-class PlantVillage label space. Only this mapping decides what counts as
correct, so it is the single source of truth for every cross-dataset number,
exactly as classes.py is for PlantVillage.

Two judgement calls a reviewer may ask about:

  - "Corn leaf blight" is PlantDoc's name for Northern Leaf Blight. The
    ambiguity matters: Northern Leaf Blight and Gray Leaf Spot are the pair
    every baseline confuses on PlantVillage too, and PlantDoc's 196 blight
    images are the largest class here.
  - A bare crop name ("Apple leaf", "Tomato leaf") means a healthy leaf of that
    crop, and maps to that crop's ___healthy class.

Ten PlantVillage classes have no PlantDoc equivalent, so a model is scored
against 28 of its 38 outputs. Accuracy is reported twice: over all 38 logits,
which is what a deployed model faces, and restricted to the 28 reachable
classes, an upper bound that separates "wrong disease" from "class that cannot
be right here".

Usage:
    python ml_pipeline/src/data/plantdoc.py        # map, count and self-check
"""

import os
import sys

from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
from classes import CLASS_TO_IDX, NUM_CLASSES  # noqa: E402

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


class PlantDocDataset(Dataset):
    """Every PlantDoc image, labelled in PlantVillage's 38-class space.

    PlantDoc's own train/test split is ignored and both halves are pooled:
    nothing here is ever trained on or selected on, so the split carries no
    meaning for this project, and pooling gives the cross-dataset estimate its
    full 2,589 images.
    """

    def __init__(self, transform=None, splits=("train", "test")):
        root = plantdoc_root()
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"{root} not found. Run:\n"
                f"  python ml_pipeline/src/data/download.py --dataset plantdoc")
        self.transform = transform
        self.rows = []
        unmapped = set()
        for split in splits:
            split_dir = os.path.join(root, split)
            for folder in sorted(os.listdir(split_dir)):
                target = PLANTDOC_TO_PLANTVILLAGE.get(folder)
                if target is None:
                    unmapped.add(folder)
                    continue
                class_idx = CLASS_TO_IDX[target]
                for name in sorted(os.listdir(os.path.join(split_dir, folder))):
                    self.rows.append({"path": os.path.join(split_dir, folder, name),
                                      "class_idx": class_idx, "plantdoc_class": folder})
        if unmapped:
            raise KeyError(f"PlantDoc folders with no entry in PLANTDOC_TO_PLANTVILLAGE: "
                           f"{sorted(unmapped)}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        # long_path: three images sit in paths over Windows' 260-character
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


if __name__ == "__main__":
    from classes import PLANTVILLAGE_CLASSES

    targets = list(PLANTDOC_TO_PLANTVILLAGE.values())
    assert len(set(targets)) == len(targets), "two PlantDoc folders map to one PlantVillage class"
    assert set(targets) <= set(PLANTVILLAGE_CLASSES), "a mapping target is not a PlantVillage class"

    ds = PlantDocDataset()
    counts = ds.class_counts()
    print(f"PlantDoc: {len(ds)} images over {len(PLANTDOC_TO_PLANTVILLAGE)} folders, "
          f"mapped onto {len(MAPPED_CLASS_IDX)} of {NUM_CLASSES} PlantVillage classes\n")
    print(f"  {'PlantVillage class':<50}{'images':>7}")
    for idx in MAPPED_CLASS_IDX:
        print(f"  {PLANTVILLAGE_CLASSES[idx]:<50}{counts[idx]:>7}")
    missing = [PLANTVILLAGE_CLASSES[i] for i in range(NUM_CLASSES) if i not in set(MAPPED_CLASS_IDX)]
    print(f"\n  {len(missing)} PlantVillage classes have no PlantDoc images:")
    for name in missing:
        print(f"    {name}")

    print("\nOpening every image (this is also the long-path check)...")
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
    assert not unreadable, "some PlantDoc images could not be opened"
    print("\nOK: every PlantDoc image is readable and mapped.")
