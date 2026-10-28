# AgriSwin-MoE

A Hybrid Vision Transformer with Shape-Texture Decoupled Attention, Sparse
Mixture-of-Experts, and On-Device SLM Advisory for Uncertainty-Aware Offline
Plant Disease Diagnosis.

**Status:** Phase 1 (foundation). Data pipeline in place; no models trained yet.

See [docs/ACTION_PLAN.md](docs/ACTION_PLAN.md) for scope, the paper's table
design, and the phase-by-phase plan.

---

## Setup

Requires Python 3.14 (PyTorch 2.14.1 ships `cp314` wheels) and an NVIDIA GPU.

```bash
py -3.14 -m venv .venv
.venv\Scripts\activate

pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.14.1 torchvision
pip install -r ml_pipeline/requirements.txt
```

Verify CUDA:

```bash
python ml_pipeline/src/data/check_env.py
```

## Data

```bash
python ml_pipeline/src/data/download.py --dataset all
python ml_pipeline/src/data/build_splits.py
```

`download.py` clones PlantVillage (~2 GB, 54k images, 38 classes) and PlantDoc
(~700 MB, field images, used **only** for the cross-dataset study) into
`data/raw/`.

`build_splits.py` writes frozen 70/15/15 splits to `data/splits/*.csv`. These
CSVs are committed: every baseline, ablation and the student model must be
evaluated on exactly the same images, or the comparison table is meaningless.

The splits are **group-aware**. PlantVillage contains many near-identical images
of the same leaf; a naive random split leaks copies between train and test and
inflates accuracy by several points. Duplicates are grouped by a 64-bit dHash
(identical-hash pass, then a 16-bit-band LSH pass merging anything within 6
bits) and each group stays inside one split.

To measure how much that matters, build both and compare:

```bash
python ml_pipeline/src/data/build_splits.py --no-dedup
```

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
