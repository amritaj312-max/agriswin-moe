# Finish the AGFT project — exact steps, in order

Six of seven arms are finished and saved on Drive. One thing remains: the
final fit of **AGFT-gated** (about 55 minutes on the L4), then the numbers go
into the paper. Everything else is done.

## Where things stand

| Arm | CV macro-F1 | PlantVillage test | PlantDoc open | PlantDoc closed |
|---|---|---|---|---|
| ConvNeXt-V2-T | 99.73 ± 0.06 | 99.77 | 32.25 | 36.00 |
| MobileNetV4-M | 99.60 ± 0.07 | 99.83 | 21.62 | 25.53 |
| Swin-V2-T | 99.66 ± 0.09 | 99.80 | 25.45 | 32.45 |
| AGFT g=1 (fine-tune) | 99.74 ± 0.08 | 99.88 | 29.24 | 34.79 |
| **AGFT g=0 (linear probe)** | 96.67 ± 0.11 | 97.75 | **47.69** | **50.74** |
| AGFT fixed α (0.507) | 99.69 ± 0.05 | 99.82 | 29.83 | 34.52 |
| AGFT learned gate | 99.70 ± 0.07 (3 folds) | *pending* | *pending* | *pending* |

To finish in time, the gated arm's cross-validation stops at the three folds
already done, and its final fit uses 8 epochs instead of 12. The paper states
both.

---

## Step 1 — Mount Drive (2 minutes, needs you)

1. Go to the Chrome tab **agft_study.ipynb – Colab**.
2. **Runtime → Run all.** If asked, click **Run anyway**.
3. When *"Permit this notebook to access your Google Drive files?"* appears,
   **click Connect to Google Drive yourself**, straight away.
4. A **separate Google window** opens; it may sit behind Chrome or in the
   taskbar. Choose **amritaj312@gmail.com** and click **Allow** or **Continue**.
5. Cell 3 must print `Mounted at /content/drive`.

If it prints `mount attempt 1 failed`, the Google window was not finished in
time. Wait for the next dialog and do steps 3–4 again; there are three tries.
If all three fail, click the ▶ on cell 3 to run it alone, complete the window,
then **Runtime → Run all**.

## Step 2 — Check the final training started (5 minutes after Step 1)

Cell 10 should show, in this order:

```
... six lines ending "already complete"
agft:gated: reusing tuned {...}
cv_agft_gated_f0 / f1 / f2: already finished ... skipped
agft:gated: cross-validation stopped after 3 of 5 folds (max_folds=3)
agft:gated: final training on all 37990 training images
    epoch 1/8  lr ...  val acc ...  macro-F1 ...  gate ...
```

Each epoch takes about 6–7 minutes, so eight take about 55. **Keep the laptop
plugged in and awake, and leave the tab open.**

## Step 3 — While it trains: put the paper on Overleaf (15 minutes)

1. Open overleaf.com → **New Project → Upload Project**.
2. Upload the folder `paper/agft` from this repository, zipped: it holds
   `agft_ieee.tex`, `refs.bib` and `figures/`.
3. Menu → Compiler: **pdfLaTeX**. Click **Recompile**. It compiles now; red
   `[...]` markers show the values still to come.
4. In the author block, replace *Department*, *Institution*, *City, Country*
   and *email* with your details, and add co-authors or a supervisor if you
   have them.

## Step 4 — When training ends: collect the numbers (5 minutes)

The notebook then runs cells 11–13 by itself.

- **Cell 10**, last line: `agft:gated: PlantVillage X | PlantDoc Y open, Z closed  mean gate g`.
- **Cell 11** prints the results table and, under it, **LaTeX rows**. Copy the
  row starting `AGFT, learned gate (ours)`.
- **Cell 13** prints the gate analysis and its **LaTeX rows**. Copy all five.
- On Drive, open `MyDrive/agft/figures/` and download **gate_analysis.png**.

## Step 5 — Fill the paper (15 minutes)

In Overleaf:

1. **Table 2 (main results):** replace the line beginning
   `AGFT, learned gate (ours)` with the row from cell 11. Keep the dagger
   `$^{\dagger}$` after the CV value.
2. **Table 3 (gate decomposition):** replace the line containing
   `\pending{rows from results/table\_gate.tex}` with the five rows from
   cell 13.
3. Upload **gate_analysis.png** into Overleaf's `figures/` folder; Figure 4
   appears by itself.
4. Search the paper for `\pending` and replace each one: the two numbers in
   the abstract (X, Y), the paragraph **"The learned gate."**, the paragraph
   under **"What the Gate Does"**, and the sentence in the **Conclusion**.
   Use the rules below to decide what to write.
5. Recompile. When no red text is left, **Download PDF**.

### How to write the gated result honestly

Compare the learned gate's PlantDoc number (Y) with the others:

| If Y is… | Write |
|---|---|
| above the linear probe (47.69) | AGFT is the best arm in the field and near the top in the lab: the gate gets the best of both. This is the strongest outcome. |
| between fixed α (29.83) and 47.69 | The per-image gate recovers part of the frozen branch's robustness that a fixed trade-off cannot, at little laboratory cost. |
| about equal to fixed α | The gate adds little in this setting. Point to the logit-scale numbers from cell 13: the fine-tuned branch's larger logits dominate any mixture. |

Then use cell 13's numbers for the "What the Gate Does" paragraph. Report:
the mean gate on lab versus field images and the Mann–Whitney *p*; whether the
gate is lower where only the frozen branch is right; and where the learned
gate sits against the best fixed α (the oracle).

---

## If something goes wrong

| Problem | Do this |
|---|---|
| Drive will not mount | Step 1 again; click Connect **yourself** and look for Google's window in the taskbar |
| Colab says *Sign in* or Reconnect does nothing | Sign in to Google in Chrome, reload the notebook, Run all |
| It disconnects during training | Reload the notebook, Run all, mount Drive: it resumes from Drive |
| Out of time before training ends | Submit with the gated row reported as cross-validation only, and say the final evaluation is pending; every other number is final |
