# What to do next, step by step

The hands-on guide: what to open, what to type, what you should see, and what to
do when something looks wrong. For *why* each step exists, see
[ACTION_PLAN.md](ACTION_PLAN.md).

**Where you are (30 Oct 2026):** four baseline runs are in. Swin-V2-T scored
**99.82% test accuracy, 99.78% macro-F1, ECE 0.13%** (15 mistakes in 8,155 test
images). EfficientNet-B0 scored **99.77 ± 0.02%, 99.66 ± 0.03%, ECE 0.14 ±
0.02%** over seeds 42, 1 and 2 (17–20 mistakes). MobileViT-S is being re-trained:
its first run hit a stochastic-depth bug, now fixed, that wiped about 1 in 5 of
its training images on every step. The re-run starts by itself once the charger
is in, and takes ~4.2 h. Then Part E, then Phase 3.

All five runs are within seven test images of each other, and a 95%
confidence interval on 8,155 images is about ±0.1 points, so **PlantVillage
accuracy alone cannot rank models or show what a new module adds.** Phase 3's
ablations must also be judged where differences can show up: the corn
leaf-spot pair (Swin-V2-T makes 3 mistakes there, EfficientNet-B0 7–9),
PlantDoc, and calibration.

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
   the lid still puts the laptop to sleep, and blocked vents make the GPU run
   hotter than the 75–78 °C it measured under load, which slows it down.

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
  a few epochs. The first run, before a fix in `classifier.py`, sat near 85%
  for all 40 epochs while val accuracy reached 99.8%: drop-path was wiping
  about 1 in 5 training images. Training accuracy that stays far below val
  accuracy again means something is wrong — tell Claude.

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

## Part E — Fill Table II's baseline rows

```powershell
.venv/Scripts/python.exe ml_pipeline/src/evaluation/summarize_results.py
```

Copy **params, test acc, macro-F1 and ECE** of the three baselines into Table II.
Leave PlantDoc accuracy, TFLite size and latency empty; they come in Phases 5–6.

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
| A run stalled for a while, then crawled at about a third of its speed | the lid was closed (the laptop sleeps and the run waits), and the charger was out — on battery the GPU runs at a third of its speed, and drained a full battery to 29% in 41 minutes | plug the charger back in: the run speeds up by itself. **To move the laptop mid-run**, press Ctrl+C first, move it, plug it in, then `--resume` |
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
lands (and pushes it, once GitHub is set up), stops without committing if a
result looks wrong, and resumes a run that crashed. After a restart, start the
same command again: finished runs are skipped, interrupted ones resumed. Its
log is `checkpoints\queue.log`.

**To try something that is not a paper result,** give it a name starting with
`dev_` and override what you like. Git ignores these results:

```powershell
.venv/Scripts/python.exe ml_pipeline/src/training/train.py --config ml_pipeline/configs/baselines/efficientnet_b0.yaml --set run_name=dev_test epochs=2
```

---

## Part H — Training on another, faster machine

**Copying the whole folder is not enough, and mostly wasted.** Of its 6.5 GB,
4.75 GB is `.venv`, which works only on this laptop: it points at
`C:\Python314`. The new machine needs its own Python environment and, often, a
different PyTorch build for its GPU. Everything else travels.

| What | Size | On the new machine |
|---|---|---|
| Code, configs, splits, git history | under 10 MB | **move it** — H1 |
| `data\raw\` (the images) | 1.7 GB | **move it**, as one archive — H2 |
| `.venv\` | 4.75 GB | **never copy it** — rebuild it, H3 |
| `checkpoints\` | grows per run | only to finish a run that started here |
| Downloaded pretrained weights | 0.15 GB | only if the machine has no internet — H3 |

**Decide where to train before the first paper run.** Runs from two machines are
still comparable — same recipe, same images, and every result records its GPU —
but one machine for every run in a table is cleanest. No paper run exists yet,
so now is the time to choose.

### H1. Get the code there

**Best: through GitHub.** It also backs your work up: today the code and its
history exist only on this laptop's disk.

1. On [github.com](https://github.com), create a new **private** repository —
   say `agriswin-moe` — with no README, licence or .gitignore.
2. Give Claude its address, and Claude connects the laptop to it and pushes.
   (By hand, in PowerShell in the project folder: `git remote add origin
   <address>`, then `git push -u origin main`.)
3. On the new machine: `git clone <address>`, then `cd agriswin-moe`.

**Or copy the folder** — leaving out `.venv` and `checkpoints`.

### H2. Move the images as one file

57,000 small files copy slowly; one archive copies quickly. In PowerShell, in the
project folder on this laptop:

```powershell
tar -cf agriswin-data.tar -C data raw
```

That writes `agriswin-data.tar` — 1.77 GB, about a minute, and git ignores it.
Copy it into the project folder on the new machine (USB drive, Google Drive,
`scp`), and unpack it there:

```bash
tar -xf agriswin-data.tar -C data
```

Then delete the `.tar` on both machines. Downloading the data afresh with
`download.py` also works, but on Linux PlantDoc's filenames then come out
unsanitised — different from this laptop's. Copying keeps every machine
byte-identical.

The images can also live outside the project, for example on Kaggle or Colab:
set the environment variable `AGRISWIN_DATA_ROOT` to the folder that contains
`raw/`.

### H3. Build the Python environment there

On **Linux**, in the project folder:

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

```bash
.venv/bin/pip install -r ml_pipeline/requirements.txt
```

On **Windows**, follow Setup in the README.

- **The `cu128` part must suit that GPU** — check
  [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).
  RTX 50-series cards need cu128 or newer, and very old cards (Kaggle's P100,
  for one) may be missing from the newest builds. A wrong choice cannot slip
  through: `check_env.py` in H4 runs a real calculation on the GPU, and fails
  at once if the build does not support it.
- **On Kaggle or Colab**, skip the PyTorch line and use the PyTorch already
  installed, which matches their drivers. Run only the `requirements.txt` line.
- **No internet on the new machine** (some university clusters)? Copy the folder
  `C:\Users\AMRITA JENA\.cache\huggingface\hub` to `~/.cache/huggingface/hub`
  there, or the first run cannot download its starting weights.

### H4. Check it before training

```bash
.venv/bin/python ml_pipeline/src/data/check_env.py
```

```bash
.venv/bin/python ml_pipeline/src/data/dataset.py
```

```bash
git status
```

Expect what A2 and A4 show here: `Environment is ready.`, `OK: dataset and
loaders work.` and a clean tree. Once training starts, its header must name the
same starting weights as on this laptop — for EfficientNet-B0,
`weights timm/efficientnet_b0.ra_in1k`.

### H5. Train — the same commands, with three differences

1. On Linux the Python is `.venv/bin/python`, not `.venv/Scripts/python.exe`.
2. **Use more data workers** if the machine has more CPU cores, or a fast GPU
   will sit waiting for images: add `--set num_workers=8` (roughly the number of
   CPU cores minus two). Paper runs may change this; it does not change results.
3. **Keep `batch_size` at 32.** A bigger GPU tempts you to raise it, but batch
   size is part of the recipe: change it, and every run is redone with the new
   value — and 32 is the most this laptop can hold. A faster GPU is faster at
   batch 32 too. (`train.py` refuses `--set batch_size=...` on a paper run.)

Kaggle and Colab sessions end after roughly 9–12 hours: split Swin-V2-T with
`--stop-after` and `--resume`, and keep `checkpoints/` on storage that outlives
the session.

### H6. Bring the results back

Commit `results/<run>.json` on the new machine and push it; on the laptop, run
`git pull`. If later phases will run on the laptop, also copy back
`checkpoints/<run>/best.pt` — the trained weights every later phase starts from.

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
