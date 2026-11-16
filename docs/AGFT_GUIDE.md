# AGFT study — what to do, in order

One notebook, one L4, about 19 GPU-hours. Everything below is a single step
you finish before starting the next.

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
| GPU | L4 only. A100 is fine but wastes units; T4 is too slow for the 19 hours |
| Compute units | ~93 of your ~180. Leaves room for one re-run |
| Time | ~19 GPU-hours, unattended, resumable |
| Drive space | ~4 GB (datasets + checkpoints) |

**Cheaper if you need it.** In cell 5, `folds=3` saves ~7 hours, and
`optuna_trials=5` saves ~1 hour. Three folds is still defensible; fewer is not.

---

## Step 1 — Open the notebook and set the GPU

Open the link above. Then **Runtime → Change runtime type → L4 GPU → Save**.

Confirm the top right says `L4` before going on. On a T4 this study takes
three times as long.

## Step 2 — Run all

**Runtime → Run all.** Two dialogs appear:

1. *"This notebook was not authored by Google"* → **Run anyway**. It is your
   own repository.
2. *"Permit this notebook to access your Google Drive files?"* → **Connect to
   Google Drive** → choose your account → **Allow**. Click this **promptly**;
   the mount times out if it waits too long.

## Step 3 — Check the first four cells, then walk away

Within about three minutes you should see:

```
NVIDIA L4, 23034 MiB            <- cell 1
Mounted at /content/drive       <- cell 3
state directory: /content/drive/MyDrive/agft
<a commit hash>                 <- cell 4
```

**If cell 3 says `mount failed`,** the notebook retries with a forced remount
by itself. If it then prints `DRIVE UNAVAILABLE`, stop: the study will run but
nothing will resume after a disconnect. Fix Drive first — usually
*Runtime → Disconnect and delete runtime*, then Run all again.

## Step 4 — Let the data cells finish (~20 minutes, first run only)

Cell 6 downloads ~2 GB. Cell 7 decodes all 54,000 images once to hash them.
Both cache to Drive, so later sessions skip them.

**What you must see before training starts:**

```
PlantVillage: 54304 images, 38 classes
PlantDoc: 2558 field images (20 dropped for carrying two labels)
split: train ... val ... test ...  (by group, so no leaf spans two splits)
no leakage: 54304 images, no perceptual hash in two splits
```

That last line is the leakage guarantee, verified rather than assumed. If it
fails the notebook stops on an assertion — do not work around it.

## Step 5 — Look at the EDA figure (cell 8)

Three panels: class balance, how much PlantVillage repeats itself, and how
differently the two datasets are composed. This figure goes in the paper as
Figure 1 and justifies two decisions: macro-F1 rather than accuracy (the
classes are imbalanced), and grouped splitting (the dataset repeats itself).

## Step 6 — The long cell (~19 hours)

Cell 9 runs seven arms, each: Optuna → 5-fold CV → final fit → two tests.
Expect roughly 2 hours per published arm and 3 per AGFT arm.

**Leave the tab open**, the laptop awake and on the charger. You can close the
laptop lid only if you have already run the `powercfg` lines in
`docs/NEXT_STEPS.md` Part A3.

**If Colab disconnects:** reopen the notebook and **Run all** again. Finished
stages are skipped and an interrupted arm resumes from its last finished
epoch. You lose minutes, not hours. This is the single most important property
of the notebook, and it is why every stage writes to Drive.

## Step 7 — Read the results (cells 10 and 11)

The table has one row per arm. **Read the PlantDoc column, not PlantVillage.**
Every arm will land within a point or so on PlantVillage, because that
benchmark is saturated; no architecture claim survives there.

The comparison that decides the paper:

| If… | Then |
|---|---|
| `agft:gated` beats `agft:fixed` on PlantDoc | The gate earns its place. That is the paper |
| they tie | The *ensemble* helps, the gate does not. Report it honestly — a two-branch result with a negative gate finding is still publishable |
| `agft:gated` beats `agft:ft` but loses to `agft:frozen` | Fine-tuning is the problem, not the gate. That reproduces Kumar et al. in a new domain and is worth reporting |

Also look at the `gate` column: it is the mean weight AGFT gave the fine-tuned
branch on field images. Below 0.5 means the model learned to lean on frozen
features out of distribution — direct evidence for the paper's thesis.

## Step 8 — Write

Everything the paper needs is on Drive: `results/summary.json` for the tables,
`figures/` for the plots. Suggested structure:

1. **Introduction** — lab accuracy is saturated; field accuracy is not.
2. **Related work** — LP-FT, WiSE-FT, side-tuning; plant-disease baselines.
3. **Method** — the gate, and why it reads the frozen branch.
4. **Experiments** — the protocol: grouped splits, Optuna, 5-fold CV, early
   stopping, one touch of the test set.
5. **Results** — the table, and the gate's behaviour under shift.
6. **Limitations** — one dataset pair, one backbone for AGFT, a single seed
   per arm. Say so; reviewers find these anyway.

**Venue.** For IEEE: a conference such as ICASSP, ICIP, or an agricultural-AI
workshop. The study's size suits a 4–6 page conference paper, not a journal.

---

## If something goes wrong

| Symptom | Fix |
|---|---|
| `mount failed` twice | Runtime → Disconnect and delete runtime, then Run all |
| Out of memory | In cell 5 set `batch_size=32`. AGFT holds two backbones |
| A stage repeats work you have already done | A result file was lost. Check `results/` on Drive |
| Optuna seems stuck | Trials are pruned silently; the study resumes from `optuna.db` |
| You want to start one arm over | Delete its folder under `ckpt/` and its `results/final_*.json` |

## What this study does **not** claim

- No new backbone. The three published models are compared as they are.
- No claim that AGFT is better in distribution; PlantVillage cannot show that.
- No claim the gate must help. The `fixed` arm exists so the gate can fail
  visibly, and if it does, the honest result is the one to publish.
