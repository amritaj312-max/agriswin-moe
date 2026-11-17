# AGFT study — what to do, in order

One notebook, one L4, at most about 24 GPU-hours. Everything below is a single
step you finish before starting the next.

**Notebook:** [agft_study.ipynb](https://colab.research.google.com/github/amritaj312-max/agriswin-moe/blob/main/notebooks/agft_study.ipynb)

---

## What the paper claims, in one paragraph

A plant-disease model reaches ~99.8% on PlantVillage and collapses on field
photographs. Kumar et al. (2022) showed why in general: fine-tuning distorts
pretrained features and can underperform a plain linear probe out of
distribution. LP-FT and WiSE-FT answer with a **fixed** trade-off — probe then
fine-tune, or interpolate two weight sets with one global α. A fixed trade-off
must be wrong somewhere, because which branch is better depends on the image.
**AGFT keeps both branches and learns a gate that weighs them per input.**

Novel: the per-input gate. Not novel, and cited as such: the two-branch idea
(side-tuning), the fixed interpolation (WiSE-FT), the staged version (LP-FT),
and the diagnosis that fine-tuning hurts OOD (Kumar et al.).

---

## Before you start

| | |
|---|---|
| GPU | L4. A100 also works but spends units faster; T4 is too slow |
| Compute units | at most ~120 of your ~180 (measured upper bound); early stopping usually saves a good share |
| Time | at most 24.4 GPU-hours, measured on an L4; unattended and resumable |
| Drive space | ~2 GB at the peak. Check you have at least 5 GB free |

**Cheaper if you need it.** In cell 5, `folds=3` cuts roughly a quarter of the
time, and `optuna_trials=5` a little more. Three folds is still defensible;
fewer is not.

---

## Step 1 — Open the notebook and set the GPU

**Close any older copy of this notebook first.** Colab keeps a tab's state,
and an old tab will keep running the old cells.

Open the link above. Then **Runtime → Change runtime type → L4 GPU → Save**.
Confirm the top right says `L4` before going on.

## Step 2 — Run all

**Runtime → Run all.** Two dialogs appear:

1. *"This notebook was not authored by Google"* → **Run anyway**. It is your
   own repository.
2. *"Permit this notebook to access your Google Drive files?"* → **Connect to
   Google Drive**. Then **look for a second, separate popup window** from
   Google: choose your account and click **Allow** (or **Continue**). It may
   open behind the Colab window. You have about two minutes; if it is not
   completed, the mount fails with `ValueError: mount failed` — that is what
   the error you saw meant. The cell tries three times, re-opening it each
   time.

## Step 3 — Check the first four cells (about three minutes)

```
NVIDIA L4, 23034 MiB                                        <- cell 1
installed: timm 1.0.30, datasets 5.2.0, optuna 5.1.0, ...   <- cell 2
Mounted at /content/drive                                   <- cell 3
state directory: /content/drive/MyDrive/agft
<a commit hash>                                             <- cell 4
```

* Cell 2 prints a note from *pip's dependency resolver* about `cudf` and
  `pyarrow`. `cudf` is a library Colab ships with that this study never
  imports; it is a warning, not a failure.
* If cell 3 prints `mount attempt 1 failed`, the sign-in popup was not
  completed — find it and finish it before the next attempt. If all three
  fail it prints `DRIVE NOT MOUNTED`: the data, EDA and preflight cells still
  run, but the long training cell refuses to start. Run cell 3 again,
  complete the popup, then **Run all**.

## Step 4 — The data cells (about five minutes the first time)

Cell 6 downloads ~2 GB to the runtime's own disk. Cell 7 hashes every image
once (cached on Drive afterwards), compares all 1.5 billion pairs of images,
and makes the split. You should see:

```
dropping ['Background_without_leaves']: 1143 images
PlantVillage: 54304 images, 38 classes
PlantDoc: 2558 field images (20 dropped for carrying two labels), covering 27 of 38 PlantVillage classes
hashes: 54304 in ...s
compared 1,474,435,056 pairs in ...s: 4192 within 6 bits
split: train 37990, val 8145, test 8169
no leakage: 54304 images, and no two in different splits are within 4 bits of each other
```

That last line is the leakage guarantee, checked pair by pair. If it fails,
the notebook stops on an assertion — do not work around it.

## Step 5 — The EDA (cell 8)

Prints the class balance and a **threshold table**: how many images count as
near-duplicates if 0, 1, … 6 bits of their hashes may differ. At 4 bits, 883
images fall into 401 groups of at most 11; at 6 bits, chains of matches merge
820 images into one group. That table justifies the 4-bit threshold in the
paper. The three-panel figure is the paper's Figure 1.

## Step 6 — The preflight (cell 9, a few minutes)

Every arm runs a few training steps on the GPU before the long cell starts,
and one line per arm appears:

```
  OK  convnextv2_tiny.fcmae_ft_in22k_in1k        130 img/s   peak memory   7.1 GiB
  OK  mobilenetv4_conv_medium.e500_r256_in1k     767 img/s   peak memory   1.7 GiB
  ...
  OK  agft:gated                                 104 img/s   peak memory   7.1 GiB
```

then an **upper bound** on each arm's GPU time — 24.4 hours in total on the
L4 where this was measured. If an arm fails here, the
message says what to change — usually `batch_size=32` in cell 5 — and you have
lost minutes, not hours. On a re-run the preflight is skipped once it passed.

## Step 7 — The long cell (cell 10, up to ~24 hours)

Seven arms, each: Optuna (with a line per trial) → 5-fold CV → final fit →
two tests. **Leave the tab open**, the laptop awake and on the charger.

**If Colab disconnects:** open the notebook again and **Run all**. Finished
stages are skipped. The run in progress resumes from its last resume point on
Drive, saved every 15 minutes, so a disconnect costs at most ~15 minutes plus
a few minutes of re-download. You will see `resuming at epoch N from Drive`.

## Step 8 — Read the results (cells 11 and 12)

The table has one row per arm. **Read the PlantDoc column, not PlantVillage.**
Every arm will land within a point or so on PlantVillage, because that
benchmark is saturated; no architecture claim survives there.

The comparison that decides the paper:

| If… | Then |
|---|---|
| `agft:gated` beats `agft:fixed` on PlantDoc | The gate earns its place. That is the paper |
| they tie | The *ensemble* helps, the gate does not. Report it honestly — a two-branch result with a negative gate finding is still publishable |
| `agft:gated` beats `agft:ft` but loses to `agft:frozen` | Fine-tuning is the problem, not the gate. That reproduces Kumar et al. in a new domain and is worth reporting |

The `gate` column is the mean weight AGFT gave the fine-tuned branch on field
images. Below 0.5 means it learned to lean on frozen features out of
distribution — direct evidence for the paper's thesis.

## Step 9 — Write

Everything the paper needs is on Drive under `agft/`: `results/summary.json`
for the tables, `results/eda.json` for the dataset numbers and the threshold
table, `figures/` for the plots, `models/` for the seven final models.

1. **Introduction** — lab accuracy is saturated; field accuracy is not.
2. **Related work** — LP-FT, WiSE-FT, side-tuning; plant-disease baselines.
3. **Method** — the gate, and why it reads the frozen branch.
4. **Experiments** — the protocol: near-duplicate-grouped splits and folds,
   Optuna, 5-fold CV, early stopping, one touch of the test set.
5. **Results** — the table, and the gate's behaviour under shift.
6. **Limitations** — one dataset pair, one backbone for AGFT, a single seed
   per arm. Say so; reviewers find these anyway.

**Venue.** An IEEE conference such as ICASSP, ICIP or an agricultural-AI
workshop. The study's size suits a 4–6 page conference paper.

**Afterwards:** empty your Drive Trash. Small replaced files collect there,
and Trash counts against your quota for 30 days.

---

## If something goes wrong

| Symptom | Fix |
|---|---|
| `mount failed` three times | Runtime → Disconnect and delete runtime, then Run all |
| The preflight says out of memory | In cell 5 set `batch_size=32`, then Run all |
| Drive is full | Free space at drive.google.com and empty Trash; raise `mirror_minutes` in cell 5 to write resume points less often |
| You want to redo one arm | Delete its `results/final_*.json` and `results/cv_*.json`, and its files under `runs/` |

## What this study does **not** claim

- No new backbone. The three published models are compared as they are.
- No claim that AGFT is better in distribution; PlantVillage cannot show that.
- No claim the gate must help. The `fixed` arm exists so the gate can fail
  visibly, and if it does, the honest result is the one to publish.
