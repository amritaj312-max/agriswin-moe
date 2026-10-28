# AgriSwin-MoE

A Hybrid Vision Transformer with Shape-Texture Decoupled Attention, Sparse
Mixture-of-Experts, and On-Device SLM Advisory for Uncertainty-Aware Offline
Plant Disease Diagnosis.

**Status:** Phase 1 complete. Environment verified, datasets downloaded, splits
frozen. **No models trained yet** — every number in the docs is a target, not a
measurement.

- [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md) — what to do next, in order, with commands.
- [docs/ACTION_PLAN.md](docs/ACTION_PLAN.md) — scope, the paper's table design, and why.

> **The images live in `data/raw/`, but are not committed.** They sit inside the
> project next to the code; `.gitignore` excludes `data/raw/`, so the 1.7 GB of
> JPEGs is never staged, committed or pushed. What *is* committed is
> `data/splits/*.csv`: the list of which image belongs to train, val or test.
> Run `python ml_pipeline/src/data/dataset.py` to confirm the two are connected.

---

## Setup

Requires Python 3.14 (PyTorch 2.14.1 ships `cp314` wheels) and an NVIDIA GPU.

```bash
py -3.14 -m venv .venv
.venv\Scripts\activate

pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.14.1 torchvision
pip install -r ml_pipeline/requirements.txt
```

Verify CUDA, packages and all four backbones:

```bash
python ml_pipeline/src/data/check_env.py
```

**Before training, set this** (see `ml_pipeline/configs/hardware.md` for why it
is not optional on a 6 GB card):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Data

```bash
python ml_pipeline/src/data/download.py --dataset all
python ml_pipeline/src/data/build_splits.py
```

`download.py` fetches PlantVillage (54,305 images, 38 classes) and PlantDoc
(2,589 field images, used **only** for the cross-dataset study).

Two Windows-specific problems are handled automatically. PlantVillage ships
colour, greyscale *and* segmented copies of every image, so a plain clone costs
~6 GB for the 811 MB that is wanted; a blobless sparse checkout fetches only
`raw/color`. PlantDoc has web-scraped filenames containing `?` and names long
enough to exceed `MAX_PATH`, which makes git abort the whole checkout, so the
tree is written manually with sanitised names (109 of 2,581 files are renamed).

This writes to `data/raw/` inside the project, which `.gitignore` excludes. If
the system drive is short on space, or the data should be shared between
checkouts, point it elsewhere:

```bash
python ml_pipeline/src/data/download.py --dataset all --data-root D:/agriswin-data
```

The choice is saved to `ml_pipeline/configs/data_root.txt` (itself gitignored);
delete that file to go back to the in-project default. Split CSVs store paths
*relative to the image root* either way, so the same frozen split works here, on
Colab and on Kaggle unchanged.

`build_splits.py` writes frozen 70/15/15 splits to `data/splits/*.csv`
(37,992 / 8,158 / 8,155 images, no empty classes, zero group leakage). These
CSVs are committed: every baseline, ablation and the student model must be
evaluated on exactly the same images, or the comparison table is meaningless.

The splits are **group-aware**. PlantVillage contains many near-identical images
of the same leaf; a naive random split leaks copies between train and test and
inflates accuracy by several points. Duplicates are grouped by a 64-bit dHash
(identical-hash pass, then a 16-bit-band LSH pass merging anything within 6
bits) and each group stays inside one split.

On this dataset it groups 1,237 near-duplicates (2.3%). To measure how much
that matters, build both and compare — the gap is worth a line in the paper:

```bash
python ml_pipeline/src/data/build_splits.py --no-dedup
```

Note `--no-dedup` and `--limit` overwrite the same CSVs, so re-run without
flags afterwards to restore the real splits.

## Hardware

Images are natively 256x256, and 256 is also the native resolution for
Swin-V2-T, MobileViT-S and FastViT-SA12, so **everything trains at 256** — only
EfficientNet-B0 is natively 224. Measured on the RTX 3050 (6 GB): 288 ms/step at
batch 16, **11 min/epoch**, ~7.5 h for a 40-epoch teacher run. Throughput is flat
from batch 16 to 48, so the GPU is compute-bound; use batch 32 and keep the
headroom. Full numbers in `ml_pipeline/configs/hardware.md`.

## Layout

```
ml_pipeline/
  src/
    data/        classes.py, download.py, build_splits.py, check_env.py
    models/      backbones, ST-DAM, MoE head  (Phase 3)
    training/    trainer, losses, distillation  (Phase 3+)
    evaluation/  metrics, calibration, Grad-CAM  (Phase 4)
    export/      ONNX / TFLite  (Phase 5)
  configs/       per-experiment YAML
data/
  raw/           downloaded datasets (git-ignored)
  splits/        frozen split CSVs (committed)
results/         metrics JSON per run (committed)
paper/           LaTeX source
docs/            design docs + action plan
```

## Class and expert definitions

`ml_pipeline/src/data/classes.py` is the single source of truth for the 38-class
label space and the 6-expert MoE routing. Import from it rather than
re-deriving lists.

The expert split is **10/4/4/4/3/13** (Tomato / Corn / Grape / Apple / Potato /
Other). Note this corrects the original design doc, which listed Potato=5,
Apple=8, Other=7 — counts that do not match the dataset.

```bash
python ml_pipeline/src/data/classes.py   # prints and self-checks the assignment
```
