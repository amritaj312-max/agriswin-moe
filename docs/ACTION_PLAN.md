# AgriSwin-MoE — Action Plan & Paper Strategy

> Status as of 2026-10-28. Written after reading all four `project_breakdown*` docs,
> `papers_index.md`, `paper_analysis.txt`, and inspecting the working tree.

---

## 0. Where the project actually stands

**Updated 3 Nov 2026.** The 28 Oct snapshot below it is kept for the record.

| Phase (section 4) | Share of the work | Done | What remains |
|---|---|---|---|
| 0 Decisions | — | partly | **venue and deadline not locked** — they set the page limit, so the tables, so the experiments |
| 1 Foundation | 10% | ~90% | the paper skeleton (LaTeX with empty tables) |
| 2 Baselines | 10% | 100% | — three models, two of them on three seeds |
| 3 Novelty modules | 25% | ~30% | ST-DAM built and verified, its three-seed control complete; the ablation trains next; MoE not started |
| 4 Uncertainty + cross-dataset | 10% | ~70% | deep-ensemble ceiling and temperature scaling |
| 5 Student + mobile | 15% | 0% | |
| 6 App + SLM | 15% | 0% | |
| 7 Writing | 15% | 0% | |
| **Overall** | | **≈ 33%** | |

Phases 2 and 4 were scheduled for weeks 2 and 6; both were essentially done
six days in. The part that is done is the part that decides whether anything
after it can be believed.

<details><summary>28 Oct 2026 snapshot</summary>

| Item | State |
|---|---|
| Research gap analysis (20 papers) | **Done** — thorough, well sourced |
| Architecture design on paper | **Done** — very detailed |
| Model code | **None** |
| Dataset downloaded | **None** |
| Experiments / results | **None** |
| Mobile app | **None** |
| Version control | **Not a git repo** |
| Python env | 3.14.6, no PyTorch (torch wheels are unreliable on 3.14) |
| GPU | RTX 3050 Laptop, 6 GB VRAM |

The whole project is currently **design documents**. Every number in the
breakdowns (97% teacher, 95% student, <200 ms latency) is a *target*, not a
measurement. The plan is excellent; the risk is that it is 6 novelties wide and
0 lines deep.

</details>

### 0.1 The novelties, checked against 2025–26 literature (3 Nov 2026)

The 20-paper gap analysis predates most of this. Checked claim by claim:

| Planned novelty | Closest published work | Verdict |
|---|---|---|
| ST-DAM, shape–texture attention | STA-Net (Qiu et al., 2025), the paper ST-DAM is adapted from; Gabor-plus-attention is common in plant disease work (ATD-Net, LeafTAM) | **Incremental.** New only as "inside a pretrained hierarchical ViT, tested for field robustness" — STA-Net was never tested across domains — and only if it measurably helps |
| Sparse MoE head | Salman et al., *Front. Plant Sci.* 2025: ViT + MoE, PlantVillage→PlantDoc; Hoang & Phan 2026: soft MoE with a Swin-Tiny expert | **Not novel** |
| MC-dropout uncertainty and abstention under field shift | *Front. Plant Sci.* 2026 "reliability gap" (PlantVillage→PlantDoc, calibration, selective prediction); *Sci. Rep.* 2026 (Swin-T variant, MC-dropout, cross-domain ECE) | **Not novel** — and our head-only version measurably yields no epistemic signal |
| Router disagreement as epistemic uncertainty (proposed in NEXT_STEPS on 3 Nov) | CARE (2026: disagreement equals the ensemble's weighted predictive variance), EMoE (2025), E³ "Sparse MoEs meet Efficient Ensembles" (2021) | **Not novel as a method.** Sensible as a design; not a claim |

**What is distinctive, and none of the closest papers have it: the
measurement.** Salman et al. report single runs with no variance, on a PlantDoc
test set whose size and class count after "cleaning" are never stated, and get
68% PlantVillage→PlantDoc. The *Sci. Rep.* paper reports "five independent runs
with fixed seed 42" — one run five times — and 90% controlled→field. Careful
studies find 22–30% (TinyCNN 28.6%; the reliability-gap paper a 67.7-point
collapse; this project 22.8%). This project has per-model seed floors, paired
tests, ablation pairs that differ in the module and nothing else, and a PlantDoc
set frozen by content hash with its contradictory labels removed — and findings
only that rigor produces:

- seed floors differ thirteen-fold between models on the same field images
  (PlantDoc accuracy sd 0.16 points for Swin-V2-T, 2.10 for EfficientNet-B0);
- the same seed on a different GPU is a fresh draw;
- a single-seed comparison manufactured a "Swin-V2-T keeps its advantage on
  field images" result that three seeds erased (corn pair 40 against 60–66);
- three seeds of one model, with identical accuracy, give the same answer on
  only 44% of field images;
- head-only MC-dropout over cached features gives structurally almost no
  epistemic uncertainty (2.9–5.0% of predictive entropy).

**Recommended framing: lead with the measurement.** Roughly: *how much of
cross-domain plant-disease progress is real — a seed-controlled study of
shape–texture attention and uncertainty under lab-to-field shift.* ST-DAM
becomes the controlled test case, positive or null; uncertainty becomes "what
actually helps abstention under shift"; the MoE head is built only if a
deep-ensemble ceiling test shows ensemble-style uncertainty is worth
approximating. This is a decision for the authors, and it does not block the
ST-DAM ablation, which both framings need.

Sources: [STA-Net](https://arxiv.org/abs/2509.03754) ·
[Salman et al. 2025](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1522985/full) ·
[Hoang & Phan 2026](https://arxiv.org/pdf/2606.23441) ·
[Reliability gap, 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13236948/) ·
[Sci. Rep. 2026](https://www.nature.com/articles/s41598-026-55107-6) ·
[TinyCNN](https://arxiv.org/pdf/2609.20290) ·
[CARE](https://arxiv.org/html/2607.26052v1) ·
[EMoE](https://arxiv.org/pdf/2505.13273) ·
[E³](https://arxiv.org/pdf/2110.03360)

---

## 1. The comparison table — answer to "I don't want 8 rows"

You are right to cut it, but the fix is not just "delete rows." The screenshot is
a **qualitative feature matrix**, and a matrix whose non-ours rows are a wall of
"No" reads to reviewers as *"we have more checkboxes"* rather than *"we are
better."* It is also not the table that gets a paper accepted.

Split it into two tables with different jobs.

### Table I — Positioning (qualitative). Keep **3** baseline rows.

Pick the single closest prior work per novelty axis. Each row must be a paper
that is genuinely strong on *some* axis, so the table shows positioning rather
than a scoreboard you wrote yourself.

| Work | Backbone | Shape-Texture Attn. | Crop-Routed MoE | Calibrated Uncertainty | Offline Advisory |
|---|---|---|---|---|---|
| STA-Net [19] | NAS-CNN | Shape + Texture (CNN only) | — | — | — |
| MobilePlantViT [20] | Hybrid ViT | — | — | — | — |
| AgriSentinel [09] | CNN + LLM | — | — | — | Cloud LLM |
| **AgriSwin-MoE (ours)** | Swin-V2 to FastViT | **Cross-attn fusion inside ViT** | **6 experts, top-2** | **MC-dropout + ECE** | **On-device SLM** |

Rationale for these three, and for dropping the rest:

- **[19] STA-Net** — you *must* keep it. Your Novelty 1 is explicitly "STA-Net's
  idea, moved into a transformer." A reviewer who knows [19] will look for it.
- **[20] MobilePlantViT** — the strongest mobile-hybrid-ViT competitor; it is the
  thing your student model has to beat.
- **[09] AgriSentinel** — the only prior work doing LLM advisory; your claim is
  "same capability, but offline."
- **Dropped [01], [02], [03], [07], [16]** — all redundant with [20] on the
  "mobile CNN/ViT, no extras" axis. Cite them in the Related Work prose instead;
  that costs 3 lines, not 5 table rows.

If space is still tight, this table can become one paragraph of Related Work
text and you lose nothing.

### Table II — Quantitative results (the table that matters). **3 trained baselines.**

This is where "minimum 2 models" applies, and 3 is the number that survives
review. Critically: **train all three yourself on your own splits.** Numbers
copied from other papers are not comparable (different splits, resolutions,
augmentation) and reviewers call this out.

| Model | Params | PV Acc. | Macro-F1 | PlantDoc Acc. (cross-dataset) | ECE (lower better) | TFLite (MB) | Latency (ms) |
|---|---|---|---|---|---|---|---|
| EfficientNet-B0 | 5.3 M | | | | | | |
| MobileViT-S | 5.6 M | | | | | | |
| Swin-V2-T (vanilla) | 28 M | | | | | | |
| **AgriSwin-MoE (teacher)** | ~30 M | | | | | | |
| **AgriSwin-MoE-S (student)** | ~12 M | | | | | | |

Why these three are nearly free:

- **Swin-V2-T vanilla** costs you *nothing extra* — it is row 1 of your ablation
  table anyway, and it is the honest "does my novelty help?" control.
- **EfficientNet-B0** trains in ~1-2 h and is the "what people actually deploy"
  reference.
- **MobileViT-S** is the mobile-ViT competitor and makes your student's win meaningful.

So: **2 extra training runs buys you 3 defensible baseline rows.**

### Table III — Ablation (non-negotiable)

For a "novel module" paper this is weighted more heavily than Table II.

| Config | ST-DAM | MoE | Acc. | Macro-F1 | Delta Acc |
|---|---|---|---|---|---|
| Swin-V2-T | — | — | | | — |
| + ST-DAM | yes | — | | | |
| + MoE | — | yes | | | |
| + both (full) | yes | yes | | | |
| ST-DAM shape-only | partial | yes | | | |
| ST-DAM texture-only | partial | yes | | | |
| no cross-attention fusion | partial | yes | | | |

---

## 2. Scope concern — read this before writing any code

**Six claimed novelties is too many for one conference paper**, and the
experiment matrix behind them does not fit a 6 GB laptop plus Colab free tier in
12 weeks. Papers that claim 6 contributions typically get reviewed as
"incremental combination of existing parts." Fewer, better-evidenced claims
score higher.

Recommended split:

**Core ML contributions — keep and ablate rigorously (3):**

1. **ST-DAM** — cheap, ablatable, produces strong Grad-CAM figures. The heart of the paper.
2. **Sparse MoE head** — novel for this domain, and the routing heatmap (which
   expert fires for which crop) is an excellent figure.
3. **Uncertainty + calibration** — *highest value per GPU-hour in the entire
   project*: it needs no retraining, only inference-time evaluation (ECE,
   reliability diagram, risk-coverage curve).

**Demote to a "System" section — still in the title, not claimed as ML novelty:**

4. **Quadruple KD** — expensive; attention distillation is well-trodden. Run plain
   logit + attention KD, report it, don't claim it as novel.
5. **On-device SLM advisory** — a genuine systems contribution and a great demo,
   but hard to evaluate rigorously (how do you score "good treatment advice"
   without agronomist review?). Present as deployment, with a latency/RAM table.
6. **End-to-end mobile pipeline** — same; this is the system wrapper.

You keep your full title. You just stop asking reviewers to evaluate six things.

**Alternative worth considering:** split into two papers — Paper A (ML:
ST-DAM + MoE + uncertainty, with ablations and cross-dataset study), Paper B
(systems/demo: the offline mobile app with SLM advisory). This is a common and
effective strategy. For a conference deadline, do Paper A.

---

## 3. Three technical problems found in the design docs

### 3.1 The MoE expert split uses wrong class counts — fix before coding

`project_breakdown_v3_final.md` assigns: Tomato 10, Corn 4, Grape 4, **Potato 5**,
**Apple 8**, Others 7.

Real PlantVillage (38 classes) is:

```
Apple 4 | Blueberry 1 | Cherry 2 | Corn 4 | Grape 4 | Orange 1 | Peach 2
Pepper_bell 2 | Potato 3 | Raspberry 1 | Soybean 1 | Squash 1 | Strawberry 2
Tomato 10                                                        = 38
```

So Potato is **3** (not 5), Apple is **4** (not 8), and "Others" is **13** (not 7).
Your 6-expert design would currently mis-size four of six experts.

Also note the severe imbalance: Tomato alone is ~18 k of 54 k images, and 12 of
the 38 classes are "healthy" leaves of different crops that look alike. Both
matter for routing and for macro-F1.

### 3.2 The MoE may be trivially solvable — pre-empt the obvious reviewer question

If experts are hard-assigned to crop families, the router only has to recognise
*which crop* — and crops are visually very distinct — so routing accuracy will be
~100% and the MoE collapses into "a per-crop classifier." A reviewer will ask:
*"isn't this just a two-stage crop-then-disease classifier?"*

Two cheap defences:

- Add a **hierarchical crop-then-disease classifier** as a baseline and show MoE beats it.
- Let the router be **learned, not supervised by crop label**, then show the
  emergent routing heatmap. If experts emerge that do *not* align with crop
  families, that is a more interesting result than if they do.

### 3.3 Gap 1 (domain shift) is over-promised — reframe it

Paper [02] reports 98.88% dropping to 13.10% when moving PlantVillage to PlantDoc.
ST-DAM and MoE **will not fix this**; the collapse comes from dataset bias (uniform
background, single centred leaf, lab lighting), not from attention design. If you
promise "we solve domain shift" and report 35% on PlantDoc, the paper is rejected.

Reframe to something you can actually deliver, which is also *more* interesting:

> **"Under domain shift our model knows that it does not know."**

Show that MC-dropout predictive entropy rises sharply on PlantDoc, enabling
abstention, and plot a **risk-coverage curve**: *at 50% coverage on PlantDoc,
accuracy is X%.* This ties Novelty 1 and Novelty 3 together, it is cheap (no
retraining), and it is honest. Optionally also report a fine-tuned-on-PlantDoc
number as an upper bound.

---

## 4. Sequenced plan — do these in order

### Phase 0 — Decisions (today, ~1 hour, no code)

1. **Lock the venue and deadline.** Page limit drives table count, which drives
   how many experiments you actually need. Everything below depends on this.
2. **Lock scope** to the 3 core + 3 system contributions above.
3. **Lock the table design** from section 1.

### Phase 1 — Foundation (Week 1)

4. **`git init`** and commit the docs. Twelve weeks of work with no version
   control is the single largest avoidable risk here.
5. **Create a Python 3.11 or 3.12 venv.** Do not fight PyTorch on 3.14.
   `torch`, `timm`, `albumentations`, `scikit-learn`, `matplotlib`.
6. **Download PlantVillage + PlantDoc.** Build **stratified splits once**, save
   the split file with a fixed seed, and commit it. PlantVillage contains
   near-duplicate images — group-aware splitting avoids inflated accuracy.
7. **Write the paper skeleton now, not in Week 12.** Create the LaTeX file with
   Tables I-III *empty*. Then every experiment you run has a cell to fill. This
   is the strongest single discipline for finishing on time — it stops you
   running experiments nobody will read.

### Phase 2 — Baselines first (Week 2) — the step most people skip

8. Train **vanilla Swin-V2-T**. This validates your data loader, splits,
   augmentation, metrics, and logging on a known-good model. If this does not
   reach ~99% on PlantVillage, nothing downstream matters.
9. Train **EfficientNet-B0**.
10. Train **MobileViT-S**.

After Week 2 you already have Table II's baseline rows and a de-risked pipeline.
Finding a bug here costs hours; finding it in Week 8 costs the deadline.

> **6 GB VRAM note:** Swin-V2-T at 224 px, batch 32, FP32 will OOM. Use AMP +
> batch 16 + gradient accumulation. Develop and debug locally on a 10% subset;
> run full training on **Kaggle** (30 GPU-h/week, more reliable than Colab free).

### Phase 3 — Novelty modules (Weeks 3-5)

11. Implement **ST-DAM**, train, ablate (shape-only / texture-only / no fusion).
12. Implement **MoE head** with corrected class counts, train, ablate
    (top-1 vs top-2; 4 / 6 / 8 experts).
13. Train the **combined model**. Produce the routing heatmap figure.

### Phase 4 — Uncertainty + cross-dataset (Week 6) — best value per GPU-hour

14. MC-dropout, ECE, reliability diagram, risk-coverage curve, PlantDoc transfer.
    Requires **no retraining** — pure inference-time evaluation on models you
    already have. Do not leave this to Week 7 as currently scheduled.

### Phase 5 — Student + mobile (Weeks 7-9)

15. KD to FastViT-SA12, TFLite INT8 export, real Android latency/size/RAM benchmarks.

### Phase 6 — App + SLM (Weeks 9-11)

16. React Native app, Gemma 3n integration, SQLite treatment fallback.

### Phase 7 — Writing (Weeks 11-12)

17. Fill the tables you created in Week 1. Write Results, then Discussion, then
    Introduction last (the intro is easiest to write once you know what you found).

---

## 5. What I'd do in the next 48 hours

```
1. git init + commit the docs                       (10 min)
2. Pick venue + deadline                            (30 min)
3. Python 3.11 venv + torch + timm, verify CUDA     (45 min)
4. Download PlantVillage, build + freeze splits     (2 h)
5. Train EfficientNet-B0 end-to-end on the splits   (2 h)
```

Step 5 is the real milestone: it turns the project from a plan into a codebase
with a number in it.
