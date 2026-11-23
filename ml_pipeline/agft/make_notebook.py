"""Build notebooks/agft_study.ipynb — the single notebook that runs the study.

The notebook imports the tested modules rather than restating their code, so
what runs on Colab is exactly what the unit tests covered on a laptop.

Before writing, the build simulates running the cells top to bottom and fails
if any cell uses a name that no earlier cell (or the cell itself) defines --
the class of bug where a fix in one cell leans on an import in a later one.
"""

import ast
import builtins
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
DEST = os.path.join(REPO_ROOT, "notebooks", "agft_study.ipynb")

# the exact versions the tests ran against
PINS = '"timm==1.0.30" "datasets==5.2.0" "optuna==5.1.0"'


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
*Runtime → Run all*. If Colab disconnects, open this notebook again and
**Run all** — finished work is skipped and the run in progress resumes.
"""),
    code("""
# 1. Which GPU did we get? This study is sized for an L4.
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"""),
    code(f"""
# 2. The exact package versions the pipeline was tested with. About a minute.
#    A "pip's dependency resolver" note, if one appears, concerns other
#    packages Colab ships with; it is a warning, not a failure.
!pip install -q {PINS} scikit-learn seaborn
import importlib.metadata as _md
print("installed:", ", ".join(f"{{p}} {{_md.version(p)}}" for p in ("timm", "datasets", "optuna", "torch")))
"""),
    code("""
# 3. Drive holds the results and the resume points.
#
#    Mounting opens Google's sign-in in a SEPARATE POPUP WINDOW: choose your
#    account and click Allow (or Continue) within about two minutes. A bare
#    "ValueError: mount failed" means that popup was not completed in time.
#    Each of the three attempts below opens it again, and first clears any
#    half-mounted folder a failed attempt left behind. Without Drive the
#    preparatory cells still run from local disk, but the long cell refuses
#    to start: nothing there would survive a disconnect.
import os
import shutil
import subprocess

from google.colab import drive

STATE = "/content/drive/MyDrive/agft"


def mount_drive(attempts=3):
    for attempt in range(attempts):
        try:
            drive.mount("/content/drive", force_remount=attempt > 0)
            return True
        except Exception as problem:
            print(f"  mount attempt {attempt + 1} failed: {problem}")
            subprocess.run(["fusermount", "-u", "/content/drive"],
                           capture_output=True)
            shutil.rmtree("/content/drive", ignore_errors=True)
    return False


if not mount_drive():
    STATE = "/content/agft"
    print("\\nDRIVE NOT MOUNTED after three attempts.")
    print("Google's sign-in popup (a separate window) was not completed. Run this")
    print("cell again and, in the popup, choose your account and click Allow.")
    print(f"Until then the preparatory cells use {STATE}, which vanishes with this")
    print("runtime, and the long training cell will not start.")
else:
    os.makedirs(STATE, exist_ok=True)
print("state directory:", STATE)
"""),
    code("""
# 4. The tested pipeline: these modules carry the unit tests that ran on a
#    laptop, so what runs here is exactly what passed there.
import sys
REPO = "/content/agriswin-moe"
if os.path.isdir(f"{REPO}/.git"):
    !git -C {REPO} pull --quiet --rebase
else:
    !git clone --quiet https://github.com/amritaj312-max/agriswin-moe.git {REPO}
sys.path.insert(0, f"{REPO}/ml_pipeline/agft")
!git -C {REPO} log --oneline -1
"""),
    md("""
## Configuration

Everything the study depends on, in one object. Measured on an L4, the seven
arms take **at most 24.4 GPU-hours (about 120 compute units)** if no run ever
stops early; early stopping and Optuna's pruning bring the real figure well
below that. The preflight below re-measures on your GPU before the long cell
starts. Lower `folds` or `optuna_trials` to spend less.

**Where things are kept.** Results, the split and the resume points go to
Drive. The datasets, the per-epoch checkpoints and Optuna's database stay on
the runtime's own disk, which is faster and costs no Drive quota; a run in
progress copies its resume point to Drive every `mirror_minutes`. A disconnect
therefore costs at most that long, plus a two-minute re-download.
"""),
    code("""
# 5. Configuration
import importlib
import pipeline as P, stages, eda
for m in (P, stages, eda):
    importlib.reload(m)

CFG = P.Config(
    root=STATE,                           # Drive: results, split, resume points
    img_size=224,
    batch_size=64,
    epochs=12, patience=3,                # early stopping on val macro-F1
    optuna_trials=10, optuna_epochs=3, optuna_subset=0.2,
    folds=5, fold_epochs=8,
    max_folds=3,                          # time limit: an arm not yet cross-validated
                                          # runs 3 of the 5 folds (0 = all five);
                                          # only AGFT-gated is affected in this study
    mirror_minutes=15,                    # resume point to Drive this often
    workers=8, seed=42,
)
P.prepare_dirs(CFG)
P.seed_everything(CFG.seed)
print("on Drive:  ", CFG.root)
print("local disk:", CFG.local_root)
"""),
    md("""
## Data

PlantVillage (lab, ~54k images, 38 classes) to train on; PlantDoc (field,
~2.5k photographs) purely as an out-of-distribution test. PlantDoc is never
trained on and is touched once, at the very end. Both are pinned to exact
dataset commits.

Two cleaning steps a reviewer will look for:

* PlantVillage's `Background_without_leaves` class is dropped — it is not a
  disease, and keeping it inflates accuracy.
* PlantDoc annotates *leaves*, not photographs, so an image's label is the
  category its annotations agree on. The 20 images carrying two different
  categories are dropped: whatever a model answers, one annotation calls it
  wrong, so they measure PlantDoc's ambiguity rather than a model's skill.
"""),
    code("""
# 6. Fetch both datasets: about 2 GB to the runtime's disk, a few minutes.
#    After a disconnect this repeats; everything expensive is on Drive.
pv, CLASSES = P.load_plantvillage(CFG)
pd_data, REACHABLE = P.load_plantdoc(CLASSES, CFG)
LABELS = [int(v) for v in pv["label"]]
PD_LABELS = [int(v) for v in pd_data["label"]]
print(f"\\n{len(CLASSES)} classes | {len(pv)} lab images | {len(pd_data)} field images")
"""),
    md("""
## Splitting without leakage

PlantVillage repeats itself: the same leaf photographed again, re-saved, or
shot alike. A random split puts such copies on both sides and inflates every
number reported against it. Each image gets a 64-bit perceptual hash; two
images whose hashes differ in at most `dup_bits` bits (4) are near-duplicates,
chains of them form groups, and **whole groups** go to one split. Exact
matching alone would catch only a twentieth of them. The cross-validation
folds are grouped the same way.

The assertion below *verifies* the result pair by pair: no two images in
different splits are within 4 bits of each other.
"""),
    code("""
# 7. Perceptual hashes (one decode of every image, cached to Drive); every pair
#    of images compared exactly (~1.5 billion, under a minute); near-duplicate
#    groups; a grouped stratified 70/15/15 split; the leakage check.
HASHES = P.group_hashes(pv, CFG)
PAIRS = P.near_duplicate_pairs(HASHES, max_bits=6)
GROUPS = P.group_ids(len(HASHES), PAIRS, CFG.dup_bits)
SPLITS = P.grouped_split(LABELS, GROUPS, CFG)
assert P.check_no_leakage(SPLITS, HASHES, CFG.dup_bits), "leakage detected -- stop and investigate"
"""),
    md("""
## Exploratory data analysis

Three questions asked of any dataset paper: how imbalanced are the classes,
how much does the training set repeat itself, and how differently are the two
datasets composed.
"""),
    code("""
# 8. EDA: numbers, and one figure saved to Drive.
from IPython.display import Image as ShowImage, display
eda.run(LABELS, CLASSES, GROUPS, PAIRS, PD_LABELS, CFG)
display(ShowImage(filename=os.path.join(CFG.paths["figures"], "eda.png")))
"""),
    md("""
## Preflight

Every arm, on this GPU, for a few optimiser steps at the real batch size —
before the long cell commits hours to it. Anything that only shows on the
hardware (running out of memory, a backbone that rejects the input size, mixed
precision) shows here in minutes rather than hours in. It also downloads every
pretrained model now, and turns the measured speeds into a time estimate.
"""),
    code("""
# 9. Preflight: a few minutes. Skipped on re-runs once it has passed.
ARMS = list(CFG.backbones) + ["agft:ft", "agft:frozen", "agft:fixed", "agft:gated"]
PREFLIGHT = stages.preflight(ARMS, pv, SPLITS, len(CLASSES), len(pd_data), CFG)
"""),
    md("""
## Train, tune, cross-validate, test

For each arm, in order: Optuna over learning rate, weight decay and dropout on
a subset with pruning; 5-fold grouped cross-validation on the **training split
only**; one final fit with early stopping; then the two tests.

This is the long cell. **It is fully resumable:** every stage writes its result
to Drive and is skipped on a re-run, and the run in progress resumes from its
last resume point. If Colab drops, open the notebook again and **Run all**.
"""),
    code("""
# 10. The whole study. Safe to re-run after a disconnect.
if not STATE.startswith("/content/drive"):
    raise SystemExit("Drive is not mounted, so none of this would survive a disconnect. "
                     "Run cell 3 again, complete Google's sign-in popup, then Run all.")
for arm in ARMS:
    stages.run_arm(arm, pv, pd_data, SPLITS, labels=LABELS, groups=GROUPS,
                   classes=CLASSES, reachable=REACHABLE, cfg=CFG)
"""),
    md("""
## Results

`PV test` is in-distribution and will be saturated — every arm lands within a
point of the others, which is why no architecture claim should rest on it.
**`PlantDoc` is the column that separates them.** `gate` shows how much weight
AGFT gave the fine-tuned branch on field images.
"""),
    code("""
# 11. The table the paper reports.
rows = stages.summarise(CFG)
"""),
    code("""
# 12. The headline figure, saved to Drive. Needs at least one finished arm.
import matplotlib.pyplot as plt

if not rows:
    raise SystemExit("No arm has finished yet -- run cell 10 to completion first.")

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
headline = os.path.join(CFG.paths["figures"], "headline.png")
plt.savefig(headline, dpi=150)
plt.close()
display(ShowImage(filename=headline))
"""),
    md("""
## What the gate does

The study's claim is about the gate, so it is examined directly. One inference
pass of the trained AGFT-gated model -- no training, about a minute -- records
both branches' predictions and the gate for every image. That answers the three
questions a reviewer will ask:

* **Does each branch work alone?** The fine-tuned branch, the frozen branch
  and the learned mixture, on both datasets.
* **Does a per-image gate beat every fixed trade-off?** All 21 fixed values of
  alpha, including the best one picked with hindsight on PlantDoc -- an oracle
  no fixed method could choose.
* **Does the gate route?** Its values on lab versus field images, and on the
  field images that only the frozen branch, or only the fine-tuned one, gets
  right.
"""),
    code("""
# 13. The gate, examined. Inference only, about a minute; needs the gated arm.
GATE = stages.gate_analysis(pv, pd_data, SPLITS, CLASSES, REACHABLE, CFG)
if GATE is not None:
    display(ShowImage(filename=os.path.join(CFG.paths["figures"], "gate_analysis.png")))
"""),
]


def check_names(cells):
    """Run the cells' name resolution top to bottom; return what is undefined."""
    defined = set(dir(builtins)) | {"get_ipython", "display"}
    problems = []
    for number, cell in enumerate(c for c in cells if c["cell_type"] == "code"):
        lines = []
        for line in "".join(cell["source"]).splitlines():
            body = line.lstrip()
            # shell and magic lines are not Python; keep the indentation
            lines.append(line[:len(line) - len(body)] + "pass"
                         if body.startswith(("!", "%")) else line)
        tree = ast.parse("\n".join(lines))
        assigned = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                assigned.add(node.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                assigned |= {(a.asname or a.name).split(".")[0] for a in node.names}
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                assigned.add(node.name)
            elif isinstance(node, ast.arg):
                assigned.add(node.arg)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                assigned.add(node.name)
        used = {n.id for n in ast.walk(tree)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        missing = used - assigned - defined
        if missing:
            problems.append((number + 1, sorted(missing)))
        defined |= assigned
    return problems


def main():
    problems = check_names(CELLS)
    if problems:
        for number, names in problems:
            print(f"code cell {number} uses names nothing before it defines: {names}")
        sys.exit(1)
    notebook = {
        "cells": CELLS,
        "metadata": {"accelerator": "GPU",
                     "colab": {"provenance": [], "toc_visible": True, "gpuType": "L4"},
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
    print(f"wrote {DEST}: {len(CELLS)} cells ({n_code} code), no outputs, "
          f"every name defined before use")


if __name__ == "__main__":
    main()
