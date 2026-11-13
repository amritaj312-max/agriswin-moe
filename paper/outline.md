# Paper outline

The argument, and which measurement carries each step. Numbers live in
[tables.md](tables.md), which is generated from `results/` — never retyped here.

**Working title.** *Right leaf, wrong reason: what a 99.8% plant-disease model
is actually reading, and what removing it does not fix.*

**Status.** Every claim below is measured; every table has its numbers.

---

## The problem with the field

A plant-disease classifier reports 99.8% on PlantVillage and 22% on field
photographs. Published work treats that as one number to improve. It is at
least three failures, and separating them changes what is worth building.

## Claim 1 — Lab accuracy cannot rank anything (Table I)

Five architectures land within five test images of each other, and the seed
alone moves the result by one image. Any paper ranking architectures on
PlantVillage is ranking seeds.

*Method note:* the seed floor is a property of the model and the machine, not
of the benchmark. Swin-V2-T's three seeds spread 0.27 points on PlantDoc where
EfficientNet-B0's spread 4.16 — a thirteen-fold difference on the same images.
Judging one model's ablation by another's floor is how a real effect gets
discarded, and this project did exactly that until it was caught.

## Claim 2 — A third of the collapse is never finding the leaf (Table II)

In a PlantDoc photograph the largest annotated leaf covers a median 30% of the
frame. Hand the model that leaf and accuracy rises 10.6–12.8 points on **every**
architecture, crop accuracy 13–15 points, and the collapse into a single sink
class falls from 16–23% of images to 4–8%. Paired McNemar over the same
photographs, p ≤ 2×10⁻³².

**But two thirds survives it.** The best boxed score is 37.4% against 99.8% in
the lab. Neither "it is just clutter" nor "it is all shortcut" is true.

## Claim 3 — The shortcut is real, recorded, and in the filenames (Table III)

PlantVillage's filenames carry the photo session that produced each image. **31
of 38 classes were shot in a single session**, so "what this disease looks like"
and "what that afternoon looked like" are one signal in training. Background
pixels alone predict the class 45% of the time against 2.6% chance.

This makes PlantVillage a shortcut benchmark whose confounder is *recorded*
rather than synthetic. The parsed table is released with the paper. No prior
work appears to use these codes.

## Claim 4 — Removing it from the representation does not help (Table IV)

Estimate each session's signature in the 7 classes where session and class vary
independently; remove it from all 38. Five head seeds per arm.

**It does nothing: −0.05 points at the best setting, inside a seed spread of
0.13–0.61.** A precise null, bounded within ±0.6 points. The kill rule was
written down before the run and it fires.

Three results survive the null, and they are why this claim belongs in the paper
rather than a drawer:

1. **Class-conditional estimation is necessary.** The marginal version — session
   means over the whole dataset, which is effectively what GroupDRO, DANN or IRM
   would reach for here — costs up to 11.35 points, 37× the seed spread, because
   on data this confounded those directions nearly *are* the class directions.
2. **The obvious guard does not work.** Holding one session fixed and checking
   the class signal survives passed at 97.3% while field accuracy fell 11
   points. It can only see *within*-session class signal and is blind to damage
   done to telling sessions apart. A guard for this has to be cross-session.
3. **The identifiable subspace has rank at most 8 of 768**, set by the data, not
   the estimator.

## Claim 5 — Framing is ~22 of the 79 points, and fine-tuning did not cause it (Table V)

PlantVillage test, each image shrunk into its own frame and padded with its own
border colour, so only the framing changes. Weighting each model's measured
curve by PlantDoc's own leaf-size distribution: **framing accounts for about 22
of the 79 points**, leaving ~57 to everything else — lighting, clutter, camera,
real-world variation, label noise.

**The hypothesis this run was built to test is refuted, and that is the more
useful result.** Fine-tuned models gain ~11 points from a leaf box while the
frozen probe gains 2.5, which suggested fine-tuning on full-frame leaves had
taught a dependence on that framing. It had not. Measured as each model's share
of its own lab accuracy at PlantDoc-median framing, the frozen probe retains
92.6% and the fine-tuned models 92.6–95.6% — the probe is at the *bottom* of the
range — and at extreme zoom-out the probe is the most robust of the six (35.6%
retained against 22.5–29.8%). Averaged over PlantDoc's real leaf sizes the two
arms are indistinguishable: 78.8% for the probe, 73.9–78.8% fine-tuned.

So scale sensitivity is a property of these backbones, not something
fine-tuning introduced, and a leaf detector would help every model about
equally.

**What that leaves unexplained is the boxed-leaf gap itself**, and the
explanation has to be that a leaf box does more than fix scale: it also removes
clutter, distractor leaves and surrounding scene. A model trained on one clean
leaf on a plain bench gains from that cleanup; an ImageNet backbone, which has
seen cluttered scenes throughout pretraining, does not. That is a claim about
clutter rather than scale, and it is testable the same way — pad a PlantVillage
leaf with *other leaves* rather than its own background.

---

## What this paper does not claim

- No new architecture. ST-DAM was built, verified, and cut unrun: every
  architecture collapses and recovers alike, so the backbone's attention is not
  the bottleneck.
- No improvement on field accuracy. The one method tried returned a null, and
  it is reported as one.
- No claim that the shortcut is unremovable — only that removing it linearly
  from frozen features does not recover field accuracy.

## Before submission

- [ ] Clutter, as the successor to Claim 5: pad a PlantVillage leaf with other
      leaves instead of its own background, and see whether fine-tuned models
      lose more than the frozen probe. One afternoon, no Colab.
- [ ] Reproduce one published number (BioCLIP 2 on PlantDoc's standard 236-image
      split) so reviewers can see 22% is a protocol difference, not a broken
      pipeline
- [ ] Redraw the three existing figures from the current results
- [ ] Decide venue: *Smart Agricultural Technology* or *Computers and
      Electronics in Agriculture*
- [ ] Release: frozen splits, the session table, the box manifest, per-image
      predictions

## Related work to position against

STA-Net 2025 (shape–texture attention, the source ST-DAM adapted); Salman et al.
2025 (ViT+MoE, 68% PlantVillage→PlantDoc, single runs, test subset unspecified);
Khalil et al. 2026 (13 foundation encoders, BioCLIP 2 57.6% on 236 images, with
bootstrap CIs and a label audit — the closest in rigour, and it does not split
crop from disease or condition on the crop); Noyan 2022 (PlantVillage background
bias survives background removal); Front. Plant Sci. 2026 (reliability gap,
temperature scaling and selective prediction help little); LeafLiteX 2026 and
LeafyGAN 2024 (segmentation and generative leaf augmentation). For the method:
Nuisance Attribute Projection (Solomonoff et al. 2005; Campbell et al.), INLP
(Ravfogel 2020), LEACE (Belrose 2023), DFR (Kirichenko 2023), CIDDG (Li 2018).
