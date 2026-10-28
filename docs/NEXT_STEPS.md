# What to do next, in order

Plain-language companion to `ACTION_PLAN.md`. That document explains *why*;
this one is just *what to type next*.

---

## First: where everything actually lives

This is the thing that looks broken but is not.

| Thing | Where it is | In git? | Size |
|---|---|---|---|
| The 54,305 PlantVillage images | `A:\agriswin-data\raw\plantvillage\raw\color\` | **No** | ~2 GB |
| The 2,589 PlantDoc images | `A:\agriswin-data\raw\plantdoc\` | **No** | ~0.7 GB |
| The *list* of which image is train/val/test | `data/splits/*.csv` | **Yes** | 6 MB |
| All the code | `ml_pipeline/src/` | **Yes** | small |
| Trained models | nowhere yet | no | - |

**Images are deliberately outside the repo.** GitHub rejects pushes over 2 GB,
and a repo you cannot clone is worse than no repo. What gets committed instead
is the *decision* about which image goes where. That is the part that must never
change, because every model in the paper has to be scored on identical images.

Look inside `data/splits/plantvillage_train.csv` and you will see rows like:

```
Apple___Apple_scab/00075aa8-...___FREC_Scab 3335.JPG,Apple___Apple_scab,0,Apple,3,0
```

That is a *pointer*: filename, class name, class number, crop, which MoE expert
owns it, and which duplicate-group it belongs to. The code joins the pointer to
`A:\agriswin-data\...` at load time. Move the images to another drive, or to
Kaggle, and only `ml_pipeline/configs/data_root.txt` changes - the splits stay
byte-for-byte identical.

Prove the data is present any time:

```bash
python ml_pipeline/src/data/dataset.py
```

Expected output: three split sizes (37,992 / 8,158 / 8,155), then one real batch
of shape `(4, 3, 256, 256)`. If that prints, the whole data layer works.

## Second: nothing is trained yet

To be unambiguous, because this is where the confusion started:

- **Zero models exist.** No weights file, no accuracy number, nothing.
- `ACTION_PLAN.md` names five models (EfficientNet-B0, MobileViT-S, Swin-V2-T,
  AgriSwin-MoE teacher, AgriSwin-MoE student). Those are the five rows of the
  paper's results table. All five cells are empty.
- "Backbones build" in the Phase 1 report means only that `timm` can construct
  the untrained network and run one forward pass on this GPU. That is a plumbing
  check, not training.

Phase 1 built the **road**. No car has driven on it.

---

## Step 1 - Write the training loop (half a day, no GPU time)

One file, `ml_pipeline/src/training/train.py`, used by every run in the paper:
baselines, ablations, teacher, student. One loop means a bug fixed once is fixed
everywhere, and no reviewer can ask whether two rows were trained differently.

It needs: AMP (fp16), cosine LR schedule with warmup, label smoothing,
per-epoch val, best-checkpoint-by-macro-F1, early stopping, a resumable
checkpoint, and a `results/<run>.json` with every metric plus the exact config.

Two non-negotiables, both from `configs/hardware.md`:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

and batch 32 at 256px. Without the allocator setting, Swin-V2-T dies above batch
16 and the error says `CUBLAS_STATUS_EXECUTION_FAILED`, which does not look like
an out-of-memory error at all.

**Checkpoint:** run it for 2 epochs on EfficientNet-B0 and confirm loss falls and
a JSON file appears. Do not start a long run until this works.

## Step 2 - Train the three baselines (~12 h GPU, mostly unattended)

Order matters. Cheapest first, so bugs surface early.

| # | Model | Epochs | Time | Why this one |
|---|---|---|---|---|
| 2a | EfficientNet-B0 | 30 | ~4 h | Cheapest. Proves the loop end to end. |
| 2b | MobileViT-S | 30 | ~5 h | The mobile competitor your student must beat. |
| 2c | Swin-V2-T vanilla | 40 | ~7.5 h | Baseline **and** ablation row 1. Free. |

**2c is the real gate.** Vanilla Swin-V2-T should reach roughly 99% on
PlantVillage - it is a strong ImageNet-pretrained model on an easy lab dataset.
If it lands at 99%, the data pipeline is trustworthy. If it lands at 80%, or at
100.0%, stop: 80% means a bug, 100% means leakage. Either way, nothing built on
top of it would mean anything.

These can run overnight. After Step 2 the paper has three real numbers in it.

## Step 3 - Build ST-DAM (Novelty 1) (~1 week)

Sobel branch (shape) + Gabor branch (texture) + cross-attention fusion, inserted
into Swin-V2-T. Then the ablation that justifies it:

| Run | What it tests |
|---|---|
| + ST-DAM full | does it help at all |
| shape branch only | is shape doing the work |
| texture branch only | is texture doing the work |
| both, concat instead of cross-attention | is the *fusion* the contribution |

That fourth run is the one a reviewer asks for. Without it, "we added attention
and it got better" is not a finding.

Grad-CAM figures come from here too, and they are the figures people remember.

## Step 4 - Build the MoE head (Novelty 2) (~1 week)

6 experts, top-2 routing, load-balancing loss. Use `classes.py` for the split -
**10/4/4/4/3/13** - not the numbers in the old design docs, which are wrong
(see `ACTION_PLAN.md` section 3.1).

Read section 3.2 of `ACTION_PLAN.md` before writing this. The short version: if
you hard-assign experts by crop, the router only has to tell an apple from a
tomato, which is trivial, and the MoE collapses into a per-crop classifier. The
defence is to let routing be **learned** rather than supervised by crop label,
then show which experts actually emerge. If they do not line up with crop
families, that is a genuinely more interesting result.

Then train the combined model (ST-DAM + MoE). That is the teacher.

## Step 5 - Uncertainty and cross-dataset (~3 days, almost no GPU)

The best value in the whole project, and it needs **no training at all** - it
runs inference on models you already have.

MC-dropout, ECE, reliability diagram, risk-coverage curve, then PlantDoc.

The honest framing (section 3.3): ST-DAM and MoE will **not** fix domain shift.
Published work shows 98.9% on PlantVillage collapsing to 13.1% on PlantDoc, and
that gap comes from dataset bias - uniform backgrounds, one centred leaf, studio
lighting - not from attention design. Claim "we solve domain shift" and report
35%, and the paper is rejected. Claim instead:

> Under domain shift, the model knows that it does not know.

Show entropy spiking on PlantDoc, show the model abstaining, and report *at 50%
coverage, accuracy is X%*. Deliverable, cheap, and true.

One real task hiding here: PlantDoc's class names do not match PlantVillage's
(`grape leaf black rot` vs `Grape___Black_rot`), the two sets do not fully
overlap, and one PlantDoc class has 2 training images. That mapping needs to be
written and checked by hand before any cross-dataset number is meaningful.

## Step 6 - Student model and mobile export (~2 weeks)

Distil the teacher into FastViT-SA12 (logit + attention KD), export TFLite INT8,
and benchmark on a real Android phone - latency, model size, peak RAM. Measured
on hardware, not estimated.

## Step 7 - App and SLM advisory (~2 weeks)

React Native app, Gemma 3n on-device, SQLite treatment fallback. This is the demo
and the systems contribution, not an ML claim.

## Step 8 - Write the paper (~2 weeks)

Fill the tables. Results first, then Discussion, then Introduction last.

---

## Do this one thing out of order, today

**Create the LaTeX skeleton with Tables I, II and III empty, before Step 1.**

It takes an hour and it is the single best discipline available here. Once every
table cell exists, each experiment has a specific hole to fill - and anything
that does not fill a hole is work nobody will read. It is also how you find out
in Week 1, rather than Week 11, that a table needs a number you never planned to
measure.

## The shortest honest summary

```
Phase 1 (done)   road built     - data, splits, loaders, measured GPU limits
Step 1           write the loop - half a day
Step 2           3 baselines    - 12 h GPU, overnight, FIRST REAL NUMBERS
Steps 3-4        the novelties  - 2 weeks
Step 5           uncertainty    - 3 days, cheap, highest value
Steps 6-8        mobile + paper - 6 weeks
```

Next command to type is Step 1, and the first thing it produces is a number for
the EfficientNet-B0 row.
