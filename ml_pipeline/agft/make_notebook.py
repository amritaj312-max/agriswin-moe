"""Build notebooks/agft_study.ipynb — the single notebook that runs the study.

The notebook imports the tested modules rather than restating their code, so
what runs on Colab is exactly what the unit tests covered on a laptop.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
DEST = os.path.join(REPO_ROOT, "notebooks", "agft_study.ipynb")


def _src(text):
    lines = text.strip("\n").splitlines(keepends=True)
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return lines


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _src(text)}


CELLS = [
    md("""
# AGFT — Adaptive Gated Fine-Tuning for plant disease recognition

One notebook, start to finish: data from the Hub, cleaning, EDA, four models,
Optuna tuning, 5-fold cross-validation, early stopping, and two tests.

**The question.** A plant-disease model reaches ~99.8% on PlantVillage and
collapses on field photographs. Kumar et al. (2022) showed why in general:
fine-tuning distorts pretrained features and can underperform a plain linear
probe out of distribution. LP-FT and WiSE-FT answer with a *fixed* trade-off.
A fixed trade-off must be wrong somewhere, because which branch is better
depends on the image.

**AGFT** keeps both branches and learns a gate that weighs them per input:

> `logits = g(x) * head_tuned(B_tuned(x)) + (1 - g(x)) * head_frozen(B_frozen(x))`

The gate reads the **frozen** features: the one description of the input whose
statistics do not drift while the other branch is being trained.

**Arms compared** — three published backbones, then four settings of the same
gated class, so the ablations *are* the experiment:

| Arm | What it is |
|---|---|
| ConvNeXt-V2-T | published CNN (2023) |
| MobileNetV4-M | published, efficient (2024) |
| Swin-V2-T | published transformer |
| `agft:ft` | gate pinned to 1 — plain fine-tuning |
| `agft:frozen` | gate pinned to 0 — linear probe |
| `agft:fixed` | one learned constant — a WiSE-FT-style fixed trade-off |
| **`agft:gated`** | **the learned per-image gate — the proposed method** |

If `agft:gated` cannot beat `agft:fixed`, the gate adds nothing, and the paper
should say so.

**Before you run:** *Runtime → Change runtime type → **L4 GPU** → Save*, then
*Runtime → Run all*. Every stage writes to Drive and is skipped once done, so
if Colab disconnects, just **Run all again** — it continues from the last
finished epoch.
"""),
    code("""
# 1. Which GPU did we get?
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"""),
    code("""
# 2. Exact package versions. About a minute.
!pip install -q "timm==1.0.30" "optuna>=4.0" "datasets>=3.0" scikit-learn seaborn
"""),
    code("""
# 3. Drive holds every checkpoint, so a disconnect costs minutes, not hours.
#    The first mount can fail with "mount failed" when a previous session left
#    a stale handle behind; a forced remount clears it. If Drive stays
#    unreachable the study still runs, but only until this runtime ends, so
#    the fallback says so loudly rather than failing silently.
from google.colab import drive

STATE = "/content/drive/MyDrive/agft"
try:
    drive.mount("/content/drive")
except Exception as first:
    print(f"first mount failed ({first}); forcing a remount")
    try:
        drive.mount("/content/drive", force_remount=True)
    except Exception as second:
        STATE = "/content/agft"
        print(f"\\nDRIVE UNAVAILABLE ({second}).")
        print(f"Falling back to {STATE}, which disappears when this runtime ends.")
        print("Nothing will resume after a disconnect. Fix Drive before a long run.")
print("state directory:", STATE)
"""),
    code("""
# 4. The tested pipeline. These modules carry 31 unit tests that run on a
#    laptop; what runs here is exactly what passed there.
import os, sys
REPO = "/content/agriswin-moe"
if os.path.isdir(f"{REPO}/.git"):
    !git -C {REPO} pull --rebase
else:
    !git clone --quiet https://github.com/amritaj312-max/agriswin-moe.git {REPO}
sys.path.insert(0, f"{REPO}/ml_pipeline/agft")
!git -C {REPO} log --oneline -1
"""),
    md("""
## Configuration

Everything the study depends on, in one object. The defaults are sized for a
single L4: about **10–12 GPU-hours** for all seven arms, roughly 55 compute
units. Lower `folds` or `optuna_trials` to spend less.
"""),
    code("""
# 5. Configuration
import importlib
import pipeline as P, stages, eda
for m in (P, stages, eda):
    importlib.reload(m)

CFG = P.Config(
    root=STATE,                           # everything resumable lives here
    img_size=224,
    batch_size=64,
    epochs=12, patience=3,                # early stopping on val macro-F1
    optuna_trials=10, optuna_epochs=3, optuna_subset=0.2,
    folds=5, fold_epochs=8,
    workers=8, seed=42,
)
P.prepare_dirs(CFG)
P.seed_everything(CFG.seed)
print("state directory:", CFG.root)
for key, value in CFG.paths.items():
    print(f"  {key:9s} {value}")
"""),
    md("""
## Data

PlantVillage (lab, ~54k images, 38 classes) to train on; PlantDoc (field,
~2.5k photographs) purely as an out-of-distribution test. PlantDoc is never
trained on and is touched once, at the very end.

Two cleaning steps a reviewer will look for:

* PlantVillage's `Background_without_leaves` class is dropped — it is not a
  disease, and keeping it inflates accuracy.
* PlantDoc annotates *leaves*, not photographs, so an image's label is the
  category its annotations agree on. The 20 images carrying two different
  categories are dropped: whatever a model answers, one annotation calls it
  wrong, so they measure PlantDoc's ambiguity rather than a model's skill.
"""),
    code("""
# 6. Fetch both datasets. The first run downloads ~2 GB; later runs reuse Drive.
pv, CLASSES = P.load_plantvillage(CFG)
pd_data, REACHABLE = P.load_plantdoc(CLASSES, CFG)
LABELS = pv["label"]
print(f"\\n{len(CLASSES)} classes | {len(pv)} lab images | {len(pd_data)} field images")
"""),
    md("""
## Splitting without leakage

PlantVillage holds many near-identical photographs of one leaf. A random split
puts copies on both sides and inflates every number reported against it. Images
are grouped by a 64-bit perceptual hash and **whole groups** are assigned to a
single split, so a leaf can never appear in both train and test.

The assertion below *verifies* that rather than assuming it.
"""),
    code("""
# 7. Perceptual hashes (one decode of everything, cached to Drive), a grouped
#    stratified 70/15/15 split, then the leakage check itself.
HASHES = P.group_hashes(pv, CFG)
SPLITS = P.grouped_split(LABELS, HASHES, CFG)
assert P.check_no_leakage(SPLITS, HASHES), "leakage detected -- stop and investigate"
"""),
    md("""
## Exploratory data analysis

Three questions asked of any dataset paper: how imbalanced are the classes,
how much does the training set repeat itself, and how differently are the two
datasets composed.
"""),
    code("""
# 8. EDA: numbers, and one figure saved to Drive.
eda.run(LABELS, CLASSES, HASHES, pd_data["label"], CFG)
from IPython.display import Image as ShowImage
ShowImage(filename=os.path.join(CFG.paths["figures"], "eda.png"))
"""),
    md("""
## Train, tune, cross-validate, test

For each arm, in order: Optuna over learning rate, weight decay and dropout on
a subset with pruning; 5-fold cross-validation on the **training split only**;
one final fit with early stopping; then the two tests.

This is the long cell — several hours. **It is fully resumable:** every stage
writes its result to Drive and is skipped on a re-run, so if Colab drops, run
this cell again and it continues.
"""),
    code("""
# 9. The whole study. Safe to re-run after a disconnect.
ARMS = list(CFG.backbones) + ["agft:ft", "agft:frozen", "agft:fixed", "agft:gated"]
for arm in ARMS:
    stages.run_arm(arm, pv, pd_data, SPLITS, LABELS, CLASSES, REACHABLE, CFG)
"""),
    md("""
## Results

`PV test` is in-distribution and will be saturated — every arm lands within a
point of the others, which is why no architecture claim should rest on it.
**`PlantDoc` is the column that separates them.** `gate` shows how much weight
AGFT gave the fine-tuned branch on field images.
"""),
    code("""
# 10. The table the paper reports.
rows = stages.summarise(CFG)
"""),
    code("""
# 11. The headline figure.
import matplotlib.pyplot as plt

names = [r["name"].replace("agft:", "AGFT ").split(".")[0] for r in rows]
positions = range(len(rows))
plt.figure(figsize=(9.5, 4))
plt.bar([i - 0.2 for i in positions],
        [r["plantvillage_test"]["accuracy"] for r in rows], 0.4, label="PlantVillage (lab)")
plt.bar([i + 0.2 for i in positions],
        [r["plantdoc"]["accuracy"] for r in rows], 0.4, label="PlantDoc (field)")
plt.xticks(list(positions), names, rotation=30, ha="right")
plt.ylabel("accuracy")
plt.title("Lab accuracy hides the difference; field accuracy shows it")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(CFG.paths["figures"], "headline.png"), dpi=150)
plt.show()
"""),
]


def main():
    notebook = {
        "cells": CELLS,
        "metadata": {"accelerator": "GPU",
                     "colab": {"provenance": [], "toc_visible": True},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 0,
    }
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    with open(DEST, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(notebook, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    with open(DEST, encoding="utf-8") as fh:            # it must be valid JSON
        json.load(fh)
    n_code = sum(1 for c in CELLS if c["cell_type"] == "code")
    print(f"wrote {DEST}: {len(CELLS)} cells ({n_code} code), no outputs")


if __name__ == "__main__":
    main()
