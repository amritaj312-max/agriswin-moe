# What to do next, step by step

The hands-on guide: what to open, what to type, what you should see, and what to
do when something looks wrong. For *why* each step exists, see
[ACTION_PLAN.md](ACTION_PLAN.md).

**Where you are (2 Nov 2026):** the baselines (A–D) are trained and checked,
scored on identical images with one recipe. These are Table II's baseline rows
(Part E) — copy them in once the paper skeleton exists:

| Model | Params | Test acc | Macro-F1 | ECE | Mistakes (of 8,155) |
|---|---|---|---|---|---|
| EfficientNet-B0 (seeds 42, 1, 2: mean ± sd) | 4.06M | 99.77 ± 0.02% | 99.66 ± 0.03% | 0.14 ± 0.02% | 17–20 |
| MobileViT-S | 4.96M | 99.79% | 99.65% | 0.17% | 17 |
| Swin-V2-T | 27.6M | 99.82% | 99.78% | 0.13% | 15 |

MobileViT-S's first run hit a stochastic-depth bug, since fixed, that wiped
about 1 in 5 of its training images on every step; the row above is the re-run.

**Every log from those four days has been audited.** Nothing was lost and no
number is tainted: no run ever crashed, was restarted or resumed, and not one
step in any run produced a non-finite loss. A closed lid suspends the training
process and a flat battery slows the GPU down, but neither changes the
arithmetic a step performs, and nothing in `train.py` depends on the clock —
the learning rate follows the step count and early stopping counts epochs. The
interruptions cost **2 h 45 min of waiting across the four days, and nothing
else.** The one real fault in that window was the MobileViT-S bug above, which
training speed had nothing to do with. What the stalls *did* spoil is the
`train_time_s` each run recorded, which is wall clock and so counts time spent
asleep; `summarize_results.py` now reports median epoch × epochs instead, and
that is the GPU column to quote in the paper.

All five runs are within five test images of each other, and a 95%
confidence interval on 8,155 images is about ±0.1 points, so **PlantVillage
accuracy alone cannot rank models or show what a new module adds.** Phase 3's
ablations must also be judged where differences can show up: the corn
leaf-spot pair (Swin-V2-T makes 3 mistakes there, MobileViT-S and
EfficientNet-B0 7–9 — Swin-V2-T's whole lead comes from this pair), PlantDoc,
and calibration.

**The evaluation harness (Part E2) shows PlantDoc accuracy cannot rank them
either.** On 2,551 field images the models never saw, accuracy collapses to
21–25% — the domain-shift collapse the literature reports (paper [02]: 98.88%
→ 13.10%), and no module will repair it. In `compare.py`'s terms:

- **No accuracy gap clears the seed floor.** Changing only the seed moves
  EfficientNet-B0 by 4.2 points (21.3%, 22.9%, 25.5%), more than any gap
  between models, so a single-seed PlantDoc accuracy ranks seeds, not
  architectures. Phase 3 needs several seeds per configuration.
- **Calibration does separate them.** Under shift Swin-V2-T is far more
  overconfident: ECE 64.6% against EfficientNet-B0's 53.8 ± 3.1%, a gap well
  beyond the seed floor. Being confidently wrong is exactly what Novelty 3's
  abstention is for.
- **So does the corn pair.** Swin-V2-T makes 40 Gray Leaf Spot / Leaf Blight
  mistakes against EfficientNet-B0's 63–74: its PlantVillage advantage on that
  pair survives on field images.
- **Head-only MC-dropout gives almost no epistemic uncertainty** (about 0.02
  nats): the abstention signal comes from the softmax, not from the passes
  disagreeing. That bears on how Novelty 3 is built, in Phase 4.

PlantDoc itself ships 9 images under two contradictory labels and 20 twice
under one label. `plantdoc.py` leaves those out and freezes the other 2,551 by
content hash in `data/splits/plantdoc_eval.csv`, so the laptop and Colab score
exactly the same images whatever their files are called.

---

## Do these next, in this order

Training now moves to **Google Colab** (an L4 or A100), where a run that took
this laptop seven hours takes one or two. The laptop stays the place where code
is written and checked. **You** marks the steps that need you; for every other
one, ask Claude.

**1. ~~Back up to GitHub.~~ Done.** Code and results are in
[github.com/amritaj312-max/agriswin-moe](https://github.com/amritaj312-max/agriswin-moe),
and the baselines' weights are the release
[weights-phase2](https://github.com/amritaj312-max/agriswin-moe/releases/tag/weights-phase2).
Weights trained on Colab are kept on your Google Drive.

**2. ~~Evaluation harness.~~ Done** — Part E2, findings above.

**3. You, once (about 15 min): set up Colab.** Part H1 and H2: Colab Pro or
pay-as-you-go compute units (the L4 and A100 need one of them), and a GitHub
token saved as a Colab secret, so each result pushes itself when it finishes.

**4. You: the first Colab queue (a few hours of GPU, unattended).** Part H3:
open `train_on_colab.ipynb`, choose the L4, *Run all*. It retrains
EfficientNet-B0 seed 42 as an end-to-end check, then Swin-V2-T seeds 1 and 2,
which give the ablation control a seed floor of its own. **Running (2 Nov):**
EfficientNet-B0 99.67% and Swin-V2-T seed 1 99.80% are done; seed 2 is
training. Their pushes failed on the token's permissions (H2 says how to fix
it) but the results are safe on Drive.

**5. You: evaluate them (minutes).** Part H6: `evaluate_on_colab.ipynb`,
*Run all*.

**6. Claude: the ST-DAM module (no GPU).** Code, configs, checks and an
ablation queue file. Expect three design questions first.

**7. You: the ST-DAM ablation on Colab.** The same training notebook with the
ablation queue, several seeds per configuration, as the seed floor demands.

**8. Then the MoE head and the full teacher.** Part F has the rest.

---

## The whole route

| | Stage | Who does it | Your time | GPU time |
|---|---|---|---|---|
| **A** | One-time laptop setup | you | 15 min | — |
| **B** | Baseline 1: EfficientNet-B0 | you, one command | 10 min | up to ~2 h |
| **C** | Baseline 2: MobileViT-S | you, one command | 10 min | up to ~4 h |
| **D** | Baseline 3: Swin-V2-T — the gate | you, one command | 10 min | up to ~8 h (overnight) |
| **E** | Read the results, fill Table II | you | 30 min | — |
| ∥ | While the GPU works: venue, paper skeleton, related work | you | hours | — |
| **F** | Phases 3–8: the novelties, uncertainty, mobile, app, writing | Claude writes code, you run it | weeks | days |

**A–E can all be done today and tomorrow.** F needs code that does not exist yet;
you start each part of it by asking Claude.

A realistic schedule:

| When | Do |
|---|---|
| Day 1, morning | A (setup), then start B |
| Day 1, ~2 h later | check B, commit it, start C |
| Day 1, evening | check C, commit it, start D and leave it overnight |
| Day 2, morning | check D (the gate), commit it, then E |

While B, C and D train, work on the parallel track (below).

**Training on a bigger machine instead of this laptop?** Read Part H before the
first paper run.

---

## Part A — One-time setup (15 minutes)

### A1. Open PowerShell in the project folder

1. Press the **Windows key**, type `PowerShell`, press **Enter**.
2. Type this and press **Enter**:

```powershell
cd "C:/Users/AMRITA JENA/Plant Disease Detection"
```

The quotes matter, because the path contains spaces. The prompt should now end
in `Plant Disease Detection>`.

Every command in this guide is typed into this window, from this folder. A new
window needs the `cd` again.

> **Use a standalone PowerShell window for training**, not a terminal inside
> another app. Closing the window stops the run. (Nothing is lost but the current
> epoch — see Part G — but it costs time.)

### A2. Check that Python, the GPU and the data all work

```powershell
.venv/Scripts/python.exe ml_pipeline/src/data/check_env.py
```

You should see a column of `[ok]` lines and, at the end, **`Environment is
ready.`** then `Data is in place.` Any `[FAIL]` or `[warn]` line: stop and tell
Claude.

```powershell
.venv/Scripts/python.exe ml_pipeline/src/data/dataset.py
```

You should see `train 37992 images`, `val 8158`, `test 8155`, and at the end
**`OK: dataset and loaders work.`**

> **Why `.venv/Scripts/python.exe` and not just `python`?** Plain `python` is
> the system Python, which has no PyTorch. The `.venv` one has everything this
> project needs. (Shortcut if you prefer: run `.venv/Scripts/Activate.ps1` once
> per window — the prompt then starts with `(.venv)` and plain `python` works.)

### A3. Make the laptop safe for multi-hour runs

A run takes 2–8 hours. `train.py` stops Windows going to sleep by itself while
it runs — its header says `power   Windows will not sleep while this runs` —
and the screen turning off is fine. The rest is up to you. Do these once:

1. **Plug in the charger** and leave it in. On battery the GPU slows right down.
2. **Power mode:** Settings → System → Power & battery → *Power mode* →
   **Best performance**.
3. **Pause Windows Update during long runs:** Settings → Windows Update →
   **Pause updates** for 1 week. An overnight update restart kills a run, and
   no program can prevent it.
4. **Keep the lid open**, on a hard flat surface — not a bed or a lap. Closing
   the lid puts the laptop to sleep by default, which suspends the run until
   someone opens it again; blocked vents also make the GPU run hotter than the
   75–78 °C it measured under load, which slows it down. Over four days of
   baselines, a closed lid and a pulled charger cost 2 h 45 min of waiting.
   These two lines stop a closed lid sleeping the laptop **while it is plugged
   in** (on battery it still sleeps, which protects the battery):

   ```powershell
   powercfg /setacvalueindex SCHEME_CURRENT 4f971e89-eebd-4455-a8de-9e59040e7347 5ca83367-6e45-459f-a27b-476b1d01c936 0
   powercfg /setactive SCHEME_CURRENT
   ```

   Keep the lid open anyway: this is insurance against knocking it shut, not
   permission to close it. To undo it, run the same lines with `1` in place of
   the final `0`.

If a run gets interrupted anyway, see Part G: you lose one epoch at most.

### A4. Check that the code is committed

```powershell
git status
```

You should see **`nothing to commit, working tree clean`**. Every result records
the exact code version that produced it, so `train.py` refuses to start a real
run while the code, configs or splits have uncommitted changes (Part G shows the
message). If you see changes you did not make, ask Claude before training.

---

## Part B — Baseline 1: EfficientNet-B0 (up to ~2 h)

### B1. Start it

```powershell
.venv/Scripts/python.exe ml_pipeline/src/training/train.py --config ml_pipeline/configs/baselines/efficientnet_b0.yaml
```

### B2. What you will see

The examples below are real output from the 2-epoch test run.

**A header.** Check two things in it: the `data` line reads **37,992 / 8,158 /
8,155**, and there is **no `SUBSET` note**. If either is wrong, press Ctrl+C and
tell Claude. (If the `power` line ever says `WARNING`, set sleep to *Never*
yourself: Settings → System → Power & battery.)

```
==============================================================================
run     efficientnet_b0  (fresh start)
model   efficientnet_b0  4.06 M params  weights timm/efficientnet_b0.ra_in1k
data    train 37,992 / val 8,158 / test 8,155
input   256px  mean (0.485, 0.456, 0.406)  std (0.229, 0.224, 0.225)
recipe  AdamW lr 0.0001 wd 0.05  warmup 2 ep, cosine to 1e-06  40 ep x 1187 steps  batch 32  float16
power   Windows will not sleep while this runs (closing the lid still will)
==============================================================================
  data workers took 23s to start
```

**A progress line every 100 steps** (about every 15 s for this model):

```
  epoch 1 [ 300/1187]  loss 2.7664  lr 8.43e-05  209 img/s  eta 2m15s
```

`loss` should fall. `img/s` should be ~200. `eta` is the time left *in this
epoch*, not in the whole run.

**One summary line per epoch:**

```
epoch  2/40  train loss 0.1335 acc 96.25%  |  val loss 0.0413 acc 98.84% F1 98.51% ECE 1.09%  |  3m12s  * best
```

- **val acc and F1 are the numbers that matter.** Expect ~98% after the first
  epoch, then a slow climb.
- **train acc is lower than val acc. That is normal here, not a bug:** training
  images are deliberately distorted (rotations, colour shifts, erased patches)
  and dropout is on, so training is harder than validation.
- **`* best`** means this epoch beat every earlier one on val macro-F1, so its
  weights were saved.
- **The first epoch is slower** (~6 min) than the rest (~3 min): workers start
  up and every file is read for the first time. Normal.

### B3. How it ends

It stops by itself, in one of two ways:

- after epoch 40, or
- earlier, with **`early stop: no val macro-F1 improvement for 10 epochs`**.
  This is the normal, expected ending: the model stopped improving, and its best
  weights are already saved.

Then it scores the **test set, once**, and prints (numbers from the test run):

```
best epoch 2: val macro-F1 98.51%
test   acc 99.03%  macro-F1 98.64%  balanced acc 98.70%  ECE 1.18%  NLL 0.0355
time   9m05s training, peak VRAM 1.83 GB
saved  results\efficientnet_b0.json
```

and returns to the prompt. That is the run finished.

### B4. Watching it (optional — it is safe to walk away)

- **Is the GPU busy?** Open a *second* PowerShell window and run `nvidia-smi`.
  **GPU-Util** should be 95–99%.
- **Is it stuck?** A new progress line appears every 15 s or so for this model
  (about a minute for Swin-V2-T). Nothing new for 5 minutes means stuck: press
  Ctrl+C and resume (Part G).
- **History so far:** open `checkpoints\efficientnet_b0\log.csv` in **Notepad**
  or VS Code — one row per epoch. Everything printed is also in
  `checkpoints\efficientnet_b0\train.log`.

> **Do not open `log.csv` in Excel while training.** Excel locks the file.
> Training survives it — it prints a warning and carries on — but the CSV misses
> rows. Copy the file first if you want it in Excel.

**One run at a time.** Two at once share the 6 GB of GPU memory, can run out of
it, and both go slower.

### B5. Check the result

```powershell
.venv/Scripts/python.exe ml_pipeline/src/evaluation/summarize_results.py
```

One row per finished run:

```
run                         params   epochs  test acc  macro-F1     ECE     NLL   time   notes
----------------------------------------------------------------------------------------------
efficientnet_b0              4.06M   23/33     99.xx%    99.xx%   x.xx%  0.0xxx   1.8h   early stop

epochs = best epoch / epochs run.  ECE and NLL: lower is better.
Splits: all 1 full-data run(s) were scored on identical images.
```

What to check:

- **test acc at least 99%** for EfficientNet-B0. The 2-epoch test already
  reached 99.03%, so a full run below that is suspicious — tell Claude.
- **notes does not say `uncommitted code`.** That means A4 was skipped: the
  number is real, but cannot be traced to a code version. (`early stop` in
  notes is normal — see B3.)
- The last line says **`scored on identical images`**. That is the fairness
  guarantee for Table II. A `WARNING` there means the runs are not comparable —
  stop and tell Claude.

### B6. Save the result in git

Only the small results file goes into git. The model weights stay in
`checkpoints\`, which git ignores because they are too big.

```powershell
git add results/efficientnet_b0.json
```

```powershell
git commit -m "Baseline result: EfficientNet-B0"
```

---

## Part C — Baseline 2: MobileViT-S (up to ~4 h)

Exactly as Part B, with a different config:

```powershell
.venv/Scripts/python.exe ml_pipeline/src/training/train.py --config ml_pipeline/configs/baselines/mobilevit_s.yaml
```

Differences to expect:

- The `input` line reads **`mean (0.000, 0.000, 0.000)  std (1.000, 1.000,
  1.000)`**. That is correct: MobileViT-S was pretrained on raw pixels and is
  fed them that way.
- About 6 minutes per epoch. The first progress line of a session shows only
  ~60 img/s while the GPU tunes itself to this model; after that, ~109 img/s.
- **Its training accuracy should track the other models'** and pass ~95% within
  a few epochs (the re-run did at epoch 4, and finished at 99.7%). The first
  run, before a fix in `classifier.py`, sat near 85% for all 40 epochs while
  val accuracy reached 99.8%: drop-path was wiping about 1 in 5 training
  images. Training accuracy that stays far below val accuracy again means
  something is wrong — tell Claude.

Then check it (B5) and commit `results/mobilevit_s.json` (B6). **If MobileViT-S
finishes clearly below EfficientNet-B0, tell Claude** before carrying on: all
three models share one learning rate, and a reviewer may ask whether that
short-changed one of them.

---

## Part D — Baseline 3: Swin-V2-T, the gate (up to ~8 h — run it overnight)

```powershell
.venv/Scripts/python.exe ml_pipeline/src/training/train.py --config ml_pipeline/configs/baselines/swinv2_tiny.yaml
```

About 11–12 minutes per epoch, a progress line roughly every minute, ~4 GB of
GPU memory.

This is the most important of the three. It is your teacher's backbone with
nothing added, so it is both a baseline and the first row of the ablation
table — and it is the check on the whole pipeline:

| Swin-V2-T test accuracy | Meaning | What to do |
|---|---|---|
| about **99–99.9%** | the pipeline is healthy | carry on |
| **below ~98%** | something is wrong | stop — tell Claude before training anything else |
| exactly **100.00%** | unusual on these splits; possible leakage | stop — tell Claude, so it can be checked before you trust it |

Start it in the evening and check it in the morning. If it was interrupted, see
Part G. Then check (B5) and commit `results/swinv2_tiny.json` (B6).

---

## Part E2 — Cross-dataset and uncertainty evaluation (~10 min GPU, no training)

Phase 4 of [ACTION_PLAN.md](ACTION_PLAN.md), and it runs **before** the ST-DAM
ablations rather than after. The reason is in the status section at the top:
the baselines land within five test images of each other on PlantVillage, so
that number cannot show what a new module adds. These can.

Three commands, in order. The first is the only one that needs the GPU:

```powershell
.venv/Scripts/python.exe ml_pipeline/src/evaluation/evaluate.py
.venv/Scripts/python.exe ml_pipeline/src/evaluation/compare.py
.venv/Scripts/python.exe ml_pipeline/src/evaluation/figures.py
```

**`evaluate.py`** scores every finished run on the PlantVillage test split and
on all 2,589 PlantDoc field images, with MC-dropout uncertainty, and writes
`results/eval/<run>.json`. It takes about two minutes per run. Before trusting
anything it produces, it re-scores the PlantVillage test split from the
checkpoint and **stops** unless that reproduces the accuracy and macro-F1 that
`results/<run>.json` recorded at training time — if a checkpoint and its
published result ever drift apart, this is what catches it.

**`compare.py`** is the one to read carefully. It prints a **seed floor**: how
far apart two runs of the *same* model land when only the seed changes. Any gap
between two different models smaller than that floor is not a result. It then
runs McNemar's paired test on every pair and says, for each, whether the
difference clears the floor. Read the two together: McNemar is sensitive enough
to call a mere seed change significant, which is exactly the mistake the floor
prevents.

**`figures.py`** writes `paper/figures/reliability`, `risk_coverage` and
`uncertainty` as PNG and PDF. Include the PDFs in LaTeX.

Re-run all three after every future phase, so each new model is judged the same
way. `evaluate.py --runs <name>` scores one run instead of all of them.

---

## Part E — Fill Table II's baseline rows

```powershell
.venv/Scripts/python.exe ml_pipeline/src/evaluation/summarize_results.py
```

Copy **params, test acc, macro-F1 and ECE** of the three baselines into Table II.
Leave PlantDoc accuracy, TFLite size and latency empty; they come in Phases 5–6.

Two things that table prints are worth understanding before quoting them:

- **`GPU`** is the median epoch times the number of epochs, not the clock time
  the run took. Quote this as training cost. A run that slept or ran on battery
  also says how many epochs stalled and what the clock said, and those two
  numbers differ by up to 56%.
- **`Splits`** and **`Recipe`** are the lines that make the table legitimate:
  they confirm every row saw the same images and the same hyperparameters. If
  either ever prints a WARNING, the rows cannot share a table — stop and tell
  Claude.

Then tell Claude **"baselines done"** and paste the table. Claude checks the
numbers and starts Phase 3.

---

## The parallel track — while the GPU is busy

Training needs no attention. Use those hours for work only you can do:

1. **Choose the venue and its deadline.** The page limit decides how many tables
   and experiments the paper can hold; everything later depends on it
   (ACTION_PLAN.md, Phase 0).
2. **Create the paper skeleton with Tables I–III empty.** Easiest: open
   [overleaf.com](https://www.overleaf.com) → New Project → the venue's template
   (most engineering conferences use the IEEE conference template). Or ask
   Claude to set it up in `paper/`. Once every table cell exists, each
   experiment has a specific hole to fill — and anything that fills no hole is
   work nobody will read.
3. **Read the three Table I papers** — STA-Net [19], MobilePlantViT [20],
   AgriSentinel [09] — and draft the Related Work section. Notes are in
   `research_papers/paper_analysis.txt`.

---

## Part F — After the baselines

These phases need code that does not exist yet. Each one starts with you asking
Claude, and ends the way Part E does: a summary table and a commit.

| Phase | What Claude builds | What you run | GPU |
|---|---|---|---|
| **3. ST-DAM** (Novelty 1) | shape (Sobel) + texture (Gabor) branches with cross-attention fusion, inside Swin-V2-T | 4 runs: full, shape-only, texture-only, no fusion | ~4 × 8 h |
| **4. MoE head** (Novelty 2) | 6 experts, top-2 routing, load-balancing loss | MoE alone, then the full teacher | ~2 × 8 h |
| **5. Uncertainty + PlantDoc** | MC-dropout, reliability diagram, risk-coverage curve, PlantDoc class mapping | evaluation scripts on models you already have — no retraining | ~1 h |
| **6. Student + mobile** | distillation into FastViT-SA12, TFLite export | one distillation run, then benchmarks on a phone | ~6 h |
| **7. App + advisory** | the app, with Gemma 3n on the device | builds and on-phone tests | — |
| **8. Writing** | — | fill the tables, write the paper | — |

---

## Part G — When something goes wrong

| You see | It means | Do this |
|---|---|---|
| The laptop slept or restarted, or the window closed | the run stopped part-way | run the **same command with `--resume` added** at the end. It continues from the last finished epoch |
| A run stalled for a while, then crawled at about a third of its speed | the lid was closed (the laptop sleeps and the run waits), and the charger was out — on battery the GPU runs at a third of its speed, and drained a full battery to 8–9% in under an hour | plug the charger back in: the run speeds up by itself, and **the result is unaffected** — a suspended or throttled run does the same arithmetic, just later. **To move the laptop mid-run**, press Ctrl+C first, move it, plug it in, then `--resume` |
| `is not named dev_..., so it has to be able to stand as a paper result` | the code, configs or splits have uncommitted changes, or `--set` changed the recipe | commit first (`git add`, then `git commit`), or drop the `--set`. To experiment, name the run `dev_...` |
| `already holds a run: pass --resume to continue it` | that run already exists | add `--resume` to continue it — or, to start from zero, delete the folder `checkpoints\<run name>` first |
| `config differs from the checkpoint in [...]` | the recipe or a config changed since the run began | undo the change. Never edit `recipe.yaml` part-way through a set of runs |
| `already finished` | that run is complete | nothing to do; its result is in `results\` |
| `out of memory` or `CUBLAS_STATUS_EXECUTION_FAILED` | GPU memory ran out | close games, videos, anything else using the GPU, then `--resume`. If it happens again, tell Claude |
| `non-finite loss: the run has diverged` | the training numbers blew up | tell Claude. Do not change the recipe yourself: every run would need redoing |
| `could not write ...log.csv (open in Excel?)` | Excel has the log open | close Excel; training is unaffected |
| `CUDA is required` | the GPU is not visible | restart the laptop, then run A2 again |
| `No module named torch` | plain `python` was used | use `.venv/Scripts/python.exe` |
| A long error message after you press Ctrl+C | that is just how a stopped run exits | nothing; `--resume` continues it |

**To stop a run on purpose,** press **Ctrl+C**. Only the epoch in progress is
lost. To continue later, add `--resume`:

```powershell
.venv/Scripts/python.exe ml_pipeline/src/training/train.py --config ml_pipeline/configs/baselines/swinv2_tiny.yaml --resume
```

**To split a long run on purpose** — say 20 epochs tonight, the rest tomorrow —
add `--stop-after 20`, and later run the same command with `--resume` instead.

**To repeat a run with another random seed** (for error bars), override only the
seed and the name — a paper run is allowed to change those two:

```powershell
.venv/Scripts/python.exe ml_pipeline/src/training/train.py --config ml_pipeline/configs/baselines/efficientnet_b0.yaml --set seed=1 run_name=efficientnet_b0_seed1
```

**To run several trainings unattended, one after another,** list them in a queue
file — `ml_pipeline/configs/queues/baselines.txt` holds the baselines — and
start:

```powershell
.venv/Scripts/python.exe ml_pipeline/src/training/run_queue.py ml_pipeline/configs/queues/baselines.txt
```

The queue waits for the charger before each run, commits each result as it
lands and pushes it to GitHub, stops without committing if a result looks
wrong, and resumes a run that crashed. After a restart, start the same command
again: finished runs are skipped, interrupted ones resumed. Its log is
`checkpoints\queue.log`, where each result should be followed by a `pushed`
line. A `push failed` line leaves the result safely committed on the laptop;
send it later with `git push`.

**To try something that is not a paper result,** give it a name starting with
`dev_` and override what you like. Git ignores these results:

```powershell
.venv/Scripts/python.exe ml_pipeline/src/training/train.py --config ml_pipeline/configs/baselines/efficientnet_b0.yaml --set run_name=dev_test epochs=2
```

---

## Part H — Training on Colab (L4 or A100)

Colab lends you a GPU several times faster than this laptop's, a few hours at a
time. The two notebooks in `notebooks/` do everything except what has to be
yours: choosing the GPU, letting Colab use your Google Drive, and making the
GitHub token. A Colab run goes through the same `train.py`, recipe and frozen
splits as a laptop run, so its result can share a table with them — and the
first queue proves it.

### H1. What you need (once)

- **Colab Pro, or pay-as-you-go compute units**
  ([colab.research.google.com/signup](https://colab.research.google.com/signup)).
  The free tier offers at most a T4; the L4 and A100 need paid compute, charged
  in compute units per hour while connected. The *Resources* panel shows the
  rate.
- **About 2 GB free on Google Drive** for the image cache, plus room for
  checkpoints: about 60 MB per EfficientNet-B0 run and 420 MB per Swin-V2-T run.
  A finished run only needs its `best.pt`; deleting a finished run's `last.pt`
  (kept only for resuming) frees about three quarters of its space.
- **A GitHub token saved as a Colab secret** — H2. Optional, but without it
  results stay on Drive and reach the repository only by hand.

### H2. Make the GitHub token (once, about 5 minutes)

The token lets Colab push results to this one repository and do nothing else.
You make it and store it; it never appears in a notebook, a file or any output.

1. On GitHub: **Settings → Developer settings → Personal access tokens →
   Fine-grained tokens → Generate new token**
   ([direct link](https://github.com/settings/personal-access-tokens/new)).
2. **Token name** `colab-agriswin`, **Expiration** 90 days.
3. **Repository access:** *Only select repositories* → `agriswin-moe`. **Do
   not leave the default, *Public repositories*:** it is read-only, and every
   push is then refused with `403 ... Permission ... denied`.
4. **Permissions → Repository permissions → Contents:** *Read and write*.
   Leave everything else as it is.
5. **Generate token** and copy it — GitHub shows it only once.
6. In Colab, with any notebook open, click the **key icon** in the left sidebar
   (*Secrets*) → **Add new secret** → Name `GITHUB_TOKEN`, Value: paste the
   token → switch on **Notebook access**.

Secrets belong to your Google account, so this covers every notebook. When the
token expires, make a new one and replace the secret's value.

**If cell 4 says the secret "cannot push" (a 403):** the token is fine but may
not write here — almost always step 3 or 4. On GitHub open **Settings →
Developer settings → Fine-grained tokens → colab-agriswin**, fix *Repository
access* and *Contents*, and save. Editing keeps the same token, so the Colab
secret needs no change. Results trained meanwhile are safe on Drive; the next
time cell 4 runs it pushes them.

### H3. Train

1. Open the training notebook straight from GitHub:
   **[train_on_colab.ipynb](https://colab.research.google.com/github/amritaj312-max/agriswin-moe/blob/main/notebooks/train_on_colab.ipynb)**.
2. **Runtime → Change runtime type → L4 GPU** (or A100 GPU) → **Save**.
3. **Runtime → Run all.** The first time, Colab asks to connect to Google
   Drive: choose your account and **Allow**.
4. **Keep the tab open** until the queue finishes. Colab may disconnect a
   runtime whose tab is closed or idle; nothing is lost if it does (H5), but
   the GPU waits until you come back.

What you should see, cell by cell:

| Cell | Expect |
|---|---|
| 1 | one line naming the GPU, such as `NVIDIA L4, 23034 MiB` |
| 3 | the latest commit on GitHub |
| 4 | the first time, a download of a few minutes. Then `identical to the reference` for train, val and test, `all 2551 images present by content`, and a summary box whose `results` line says *pushed to GitHub* |
| 6 | the familiar `train.py` header and progress lines, with several times the laptop's img/s |
| 7 | the results table, ending in `identical images` and `same recipe` |

**Why each table is trained on one machine.** EfficientNet-B0 seed 42, already
trained on the laptop (99.75–99.79% over three seeds), was retrained first as
an end-to-end check, and scored **99.67%** on the L4 — about 27 test mistakes
against 17–20. That does *not* show the GPU changes results: "seed 42" on
another machine is effectively a new random draw (a different GPU, PyTorch
build and number of loader workers change the augmentation stream), its
checkpoint was chosen at a one-epoch spike in validation before early stopping
cut the low-learning-rate phase short, and a fourth run falls outside the range
of three **half the time** even when nothing differs. But one run cannot prove
the machines equivalent either. So comparisons stay within a machine: Table II
is the laptop baselines, and every row of the Phase 3 ablation table, its
Swin-V2-T control included, is trained on Colab with the same seeds. Swin-V2-T
seed 1 on Colab, 99.80%, sits within one image of the laptop's 99.82%.

### H4. Which GPU

Start on the **L4**. It costs much less per hour than the A100 and has ample
memory for batch 32. Batch size stays at 32 on every GPU — it is part of the
recipe, so raising it would make every new row incomparable with the baselines
(`train.py` refuses to) — and at batch 32 the A100's extra power is partly
idle. The `img/s` in the progress lines is the real speed: to compare, run a
queue on each and divide by each GPU's compute-unit rate.

### H5. If the runtime disconnects

Open the notebook and **Run all** again. Checkpoints, the image cache and a copy
of every finished result live on your Drive, so finished runs are skipped, an
interrupted run resumes from its last finished epoch, and any result the last
session finished but did not push is pushed now.

### H6. Evaluate Colab runs

**[evaluate_on_colab.ipynb](https://colab.research.google.com/github/amritaj312-max/agriswin-moe/blob/main/notebooks/evaluate_on_colab.ipynb)**
→ any GPU → **Run all**. It scores every run whose weights are on your Drive
and has no evaluation yet, compares all runs, laptop and Colab together,
redraws the figures, and pushes the lot. Then, on the laptop:

```powershell
git pull
```

### H7. Another machine instead (a lab server, Kaggle)

The same steps by hand. Clone the repository, and build a Python environment as
in the README's *Setup* (on Kaggle, keep its PyTorch and install only
`timm==1.0.30`). Fetch the images — `download.py` checks out the same pinned
commits as everywhere else:

```bash
python ml_pipeline/src/data/download.py --dataset all
```

Then check them. This must end with `OK: every image is byte-for-byte the one
the results were made on` before anything trains there:

```bash
python ml_pipeline/src/data/verify_data.py
```

Train with the queue runner, adding `--set-all num_workers=8` (about one per CPU
core) so a fast GPU is not kept waiting for images, and never change
`batch_size`. Commit and push each `results/<run>.json`, then `git pull` on the
laptop. To use the baselines' weights there, download `SHA256SUMS.txt` and the
parts you need from the release
[weights-phase2](https://github.com/amritaj312-max/agriswin-moe/releases/tag/weights-phase2)
into one folder, and run:

```bash
python ml_pipeline/src/training/restore_weights.py <that folder>
```

---

## Five rules that keep the results publishable

The code now enforces or checks four of them; only rule 3 rests on you alone.

1. **Never edit `ml_pipeline/configs/recipe.yaml` between runs that share a
   table.** If it has to change, every affected run is redone — ask Claude
   first. *Checked:* `summarize_results.py` warns when runs used different
   recipes.
2. **Commit before every real run.** Results record the code version.
   *Enforced:* `train.py` refuses a real run with uncommitted code, configs or
   splits.
3. **Never use test numbers to choose anything** — not the number of epochs,
   not a setting. The loop scores the test set once, at the end, by itself;
   leave it that way.
4. **Never edit `data/splits/*.csv`.** They define what "the test set" is.
   *Enforced and checked:* an edited split counts as an uncommitted change, and
   `summarize_results.py` confirms every run used the same images.
5. **Name experiments `dev_...`.** Only real runs belong in `results/` and in
   git. *Enforced:* a run that changes the recipe with `--set`, or trains on a
   subset, must be named `dev_...`.

---

## Where everything is

All paths are inside `C:\Users\AMRITA JENA\Plant Disease Detection\`.

| What | Where | In git? |
|---|---|---|
| The 54,305 PlantVillage images | `data\raw\plantvillage\raw\color\` | no — too big |
| The 2,589 PlantDoc images | `data\raw\plantdoc\train\`, `...\test\` | no — too big |
| Which image is train / val / test | `data\splits\*.csv` | **yes** |
| The shared training recipe | `ml_pipeline\configs\recipe.yaml` | **yes** |
| One config per baseline | `ml_pipeline\configs\baselines\*.yaml` | **yes** |
| Model weights and logs, per run | `checkpoints\<run>\` | no — too big |
| Results, per run | `results\<run>.json` | **yes** (except `dev_*`) |
| The code | `ml_pipeline\src\` | **yes** |

A finished run's folder holds `best.pt` (weights at the best epoch — what every
later phase uses), `last.pt` (full state, for `--resume`), `log.csv`,
`train.log`, and `FINISHED`. For a real example, look in
`checkpoints\dev_efficientnet_b0\`.
