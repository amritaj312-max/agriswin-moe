# 🌿 Plant Disease Detection — FINAL IEEE-GRADE Architecture (v3)
# Novel Architecture: **AgriSwin-MoE** with On-Device SLM Treatment Advisory

---

## RESEARCH GAP ANALYSIS (from 20 Papers)

```
+===========================================================================+
|        GAPS & LIMITATIONS IDENTIFIED ACROSS 20 RESEARCH PAPERS            |
+===========================================================================+
|                                                                           |
|  GAP 1: DOMAIN SHIFT / GENERALIZATION FAILURE                  [CRITICAL]|
|  +-------------------------------------------------------------------+   |
|  | Papers 01, 02, 05, 07, 08, 17, 18, 20 all report that models     |   |
|  | trained on PlantVillage (lab-controlled) COLLAPSE on real-world   |   |
|  | field images. Paper 02 (TinyCNN) drops from 98.88% to 13.10% on  |   |
|  | PlantDoc. Grad-CAM shows models attend to BACKGROUND, not leaves. |   |
|  | No paper provides a unified solution for lab-to-field transfer.   |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
|  GAP 2: NO UNCERTAINTY ESTIMATION ON MOBILE                    [CRITICAL]|
|  +-------------------------------------------------------------------+   |
|  | Papers 01-20: NONE provide calibrated confidence on edge devices.  |   |
|  | A farmer gets "94% Tomato Blight" even on a misclassified image.  |   |
|  | Paper 13 (TinyBayes) proposes Bayesian inference on edge but NOT  |   |
|  | for plant disease & not combined with transformers.                |   |
|  | No paper says "I'm uncertain, please retake or consult expert."   |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
|  GAP 3: CLASSIFICATION ONLY — NO TREATMENT RECOMMENDATION      [MAJOR]  |
|  +-------------------------------------------------------------------+   |
|  | Papers 01-08, 11-12, 15-20: Only classify disease, do NOT suggest |   |
|  | treatment. Paper 09 (AgriSentinel) uses LLM for treatment but     |   |
|  | requires cloud/API (GPT-2 fine-tuned) — NOT offline.              |   |
|  | Paper 10 (FourCropNet) has basic treatment but no contextual AI.  |   |
|  | NO paper combines on-device SLM with vision model for offline     |   |
|  | contextual treatment advice in local languages.                   |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
|  GAP 4: SINGLE ATTENTION MECHANISM — NO SHAPE-TEXTURE DECOUPLING [MAJOR]|
|  +-------------------------------------------------------------------+   |
|  | Paper 19 (STA-Net) proposes decoupled shape/texture attention     |   |
|  | BUT uses only CNN backbone (no transformer), tested on small CCMT |   |
|  | dataset (88.96% only), and no mobile deployment.                  |   |
|  | Paper 12: single SE attention head — "may limit complex           |   |
|  | inter-class feature capture."                                     |   |
|  | No paper combines shape-texture decoupled attention WITH          |   |
|  | vision transformers AND knowledge distillation for mobile.        |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
|  GAP 5: NO MIXTURE OF EXPERTS FOR CROP-SPECIFIC SPECIALIZATION   [MAJOR]|
|  +-------------------------------------------------------------------+   |
|  | All 20 papers use a single dense classification head.             |   |
|  | Tomato diseases (10 classes) and Apple diseases (4 classes) share |   |
|  | the same weights — causing cross-crop confusion.                  |   |
|  | No paper uses MoE for crop-family-specific expert routing.        |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
|  GAP 6: NO END-TO-END SYSTEM (Detection + Classification + Advice) [GAP]|
|  +-------------------------------------------------------------------+   |
|  | Papers either do classification OR detection OR treatment.        |   |
|  | No paper builds a COMPLETE pipeline: detect leaf region ->        |   |
|  | classify disease -> estimate uncertainty -> suggest treatment     |   |
|  | -> all OFFLINE on a mobile phone.                                 |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
|  GAP 7: ENERGY / LATENCY NOT MEASURED FOR ACTUAL MOBILE DEPLOYMENT [GAP]|
|  +-------------------------------------------------------------------+   |
|  | Paper 03: "energy consumption critical for battery-powered        |   |
|  | drones/solar nodes remains unmeasured."                           |   |
|  | Paper 04: discusses energy but only on desktop/server.            |   |
|  | Most papers benchmark on GPU/CPU, NOT actual Android phones.      |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
|  GAP 8: NO MULTI-MODAL / SENSOR FUSION                           [GAP]  |
|  +-------------------------------------------------------------------+   |
|  | Paper 17 future work: "Multi-modal data, environmental params    |   |
|  | or hyperspectral imagery could improve generalizability."         |   |
|  | No paper fuses image data with weather/humidity/location context. |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
|  GAP 9: NO FEDERATED/PRIVACY-PRESERVING LEARNING                 [GAP]  |
|  +-------------------------------------------------------------------+   |
|  | Paper 03 future work: "federated distillation for privacy-        |   |
|  | preserving, decentralized training across edge nodes."            |   |
|  | Paper 09 adds differential privacy but only for image upload.    |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
|  GAP 10: NO GRAPH-BASED REASONING COMBINED WITH VISION MODELS     [GAP] |
|  +-------------------------------------------------------------------+   |
|  | Paper 15 (GAT-GCN) uses graph networks but NO vision transformer,|   |
|  | NO mobile deployment, NO knowledge distillation.                  |   |
|  | Combining GNN disease-symptom reasoning with ViT features is     |   |
|  | completely unexplored.                                            |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
+===========================================================================+
```

---

## OUR NOVEL CONTRIBUTIONS (IEEE Novelty)

```
+===========================================================================+
|                                                                           |
|         PROPOSED NOVEL ARCHITECTURE: "AgriSwin-MoE"                       |
|                                                                           |
|   A Hybrid Vision Transformer with Shape-Texture Decoupled Attention,     |
|   Sparse Mixture-of-Experts Classification, Bayesian Uncertainty          |
|   Estimation, and On-Device SLM Treatment Advisory for Offline            |
|   Mobile Plant Disease Diagnosis                                          |
|                                                                           |
+===========================================================================+
|                                                                           |
|  NOVELTY 1: SHAPE-TEXTURE DECOUPLED ATTENTION MODULE (ST-DAM)            |
|  ============================================================            |
|  - Inspired by STA-Net (Paper 19) BUT integrated into a                  |
|    Vision Transformer (Swin-V2) backbone — first-of-its-kind             |
|  - Shape Branch: Sobel-filtered edge features capture lesion contours    |
|  - Texture Branch: Gabor-filtered patterns capture disease textures      |
|  - Cross-Attention Fusion: shape and texture features attend to          |
|    each other before classification                                      |
|  - ADDRESSES: Gap 4 (single attention), Gap 1 (background attention)     |
|                                                                           |
|  NOVELTY 2: SPARSE MoE CLASSIFICATION HEAD WITH CROP-ROUTING             |
|  ============================================================            |
|  - 6 lightweight expert sub-networks, each specialized for a             |
|    crop family (Tomato, Corn, Grape, Potato, Apple, Others)              |
|  - Learnable gating router: top-2 sparse activation per inference        |
|  - Load-balancing auxiliary loss prevents expert collapse                 |
|  - FIRST use of MoE in plant disease classification                      |
|  - ADDRESSES: Gap 5 (no crop-specific specialization)                    |
|                                                                           |
|  NOVELTY 3: BAYESIAN UNCERTAINTY VIA MC-DROPOUT ON EDGE                  |
|  ============================================================            |
|  - Monte Carlo Dropout at inference time (5 forward passes)              |
|  - Epistemic uncertainty estimation: variance across predictions         |
|  - Uncertainty-aware decision: if uncertainty > threshold,               |
|    say "Uncertain — retake photo or consult expert"                      |
|  - Lightweight: only 5x inference cost (still <500ms on phone)           |
|  - ADDRESSES: Gap 2 (no uncertainty on mobile)                           |
|                                                                           |
|  NOVELTY 4: HYBRID KNOWLEDGE DISTILLATION (Attn + Logit + ST-DAM)       |
|  ============================================================            |
|  - Teacher: Swin-V2-Tiny + ST-DAM + MoE (high accuracy, ~97%)           |
|  - Student: FastViT-SA12 + ST-DAM-Lite + MoE-Lite (mobile)              |
|  - Triple-loss distillation:                                             |
|    L = a*L_ce + b*L_kd_logit + g*L_attn + d*L_stdam                    |
|  - ST-DAM attention maps are ALSO distilled (shape+texture alignment)    |
|  - FIRST distillation of decoupled attention maps for agriculture        |
|  - ADDRESSES: Gap 4 (attention), Gap 1 (generalization)                  |
|                                                                           |
|  NOVELTY 5: ON-DEVICE SLM FOR CONTEXTUAL TREATMENT ADVISORY             |
|  ============================================================            |
|  - Gemma 3n E2B (~2B params, runs on 2-4GB RAM phones)                   |
|  - Receives: disease_name + confidence + crop + severity                 |
|  - Generates: contextual treatment advice in LOCAL LANGUAGE              |
|  - Fallback: pre-built SQLite DB when SLM unavailable                    |
|  - 100% offline — no API, no cloud, no internet needed                   |
|  - FIRST on-device SLM integration for plant disease treatment           |
|  - ADDRESSES: Gap 3 (no treatment), Gap 6 (no end-to-end system)        |
|                                                                           |
|  NOVELTY 6: COMPLETE END-TO-END OFFLINE MOBILE PIPELINE                  |
|  ============================================================            |
|  - Detect (leaf ROI) -> Classify (disease) -> Uncertainty ->             |
|    Treatment (SLM) -> History -> ALL OFFLINE                             |
|  - Actual Android phone benchmarks (latency, energy, RAM)                |
|  - ADDRESSES: Gap 6 (end-to-end), Gap 7 (real mobile benchmarks)        |
|                                                                           |
+===========================================================================+
|                                                                           |
|  PAPER TITLE (SUGGESTED):                                                |
|  "AgriSwin-MoE: A Hybrid Vision Transformer with Shape-Texture           |
|   Decoupled Attention, Sparse Mixture-of-Experts, and On-Device           |
|   SLM Advisory for Uncertainty-Aware Offline Plant Disease Diagnosis"     |
|                                                                           |
+===========================================================================+
```

---
---

# =====================================================
# PART 1: BUILD THE MODEL (Updated Novel Architecture)
# =====================================================

## 1.1 — Novel Model Architecture: AgriSwin-MoE

```
+===========================================================================+
|                 AgriSwin-MoE ARCHITECTURE (TEACHER MODEL)                 |
+===========================================================================+
|                                                                           |
|  INPUT IMAGE (224 x 224 x 3)                                             |
|       |                                                                   |
|       v                                                                   |
|  +================================================================+      |
|  |            SWIN TRANSFORMER V2-TINY BACKBONE                    |      |
|  |            (Pretrained ImageNet-1K, 28M params)                 |      |
|  |                                                                  |      |
|  |  Stage 1: Patch Embed -> 2x Swin Blocks (56x56, 96-dim)        |      |
|  |       |                                                          |      |
|  |  Stage 2: Patch Merge -> 2x Swin Blocks (28x28, 192-dim)       |      |
|  |       |                                                          |      |
|  |  Stage 3: Patch Merge -> 6x Swin Blocks (14x14, 384-dim)       |      |
|  |       |                                                          |      |
|  |  Stage 4: Patch Merge -> 2x Swin Blocks (7x7, 768-dim)         |      |
|  |                                                                  |      |
|  +============================+=====================================+      |
|                               |                                           |
|                               | 768-dim feature maps (7x7)                |
|                               v                                           |
|  +================================================================+      |
|  |       SHAPE-TEXTURE DECOUPLED ATTENTION MODULE (ST-DAM)         |      |
|  |       [OUR NOVEL CONTRIBUTION #1]                               |      |
|  |                                                                  |      |
|  |       Input Features (7x7x768)                                  |      |
|  |            |                                                     |      |
|  |       +----+----+                                                |      |
|  |       |         |                                                |      |
|  |       v         v                                                |      |
|  |  +---------+ +---------+                                        |      |
|  |  | SHAPE   | | TEXTURE |                                        |      |
|  |  | BRANCH  | | BRANCH  |                                        |      |
|  |  |         | |         |                                        |      |
|  |  | Sobel   | | Gabor   |                                        |      |
|  |  | Filter  | | Filter  |                                        |      |
|  |  | (fixed) | | Bank    |                                        |      |
|  |  |    |    | | (fixed) |                                        |      |
|  |  |    v    | |    |    |                                        |      |
|  |  | 1x1Conv | |    v    |                                        |      |
|  |  | BN+ReLU | | 1x1Conv |                                        |      |
|  |  | Spatial | | BN+ReLU |                                        |      |
|  |  | Attn    | | Channel |                                        |      |
|  |  | (sigmoid| | Attn    |                                        |      |
|  |  |  gate)  | | (SE blk)|                                        |      |
|  |  +----+----+ +----+----+                                        |      |
|  |       |           |                                              |      |
|  |       v           v                                              |      |
|  |  +---------------------------------+                             |      |
|  |  | CROSS-ATTENTION FUSION          |                             |      |
|  |  |                                 |                             |      |
|  |  | Q = Shape features              |                             |      |
|  |  | K, V = Texture features         |                             |      |
|  |  | + Residual from original        |                             |      |
|  |  |                                 |                             |      |
|  |  | Output = LayerNorm(             |                             |      |
|  |  |   original + a*ShapeAttn        |                             |      |
|  |  |            + b*TextureAttn)     |                             |      |
|  |  +---------------------------------+                             |      |
|  |              |                                                   |      |
|  +==============+=================================================+      |
|                 |                                                         |
|                 | ST-DAM enhanced features (7x7x768)                     |
|                 v                                                         |
|  +================================================================+      |
|  |          GLOBAL AVERAGE POOLING -> 768-dim vector               |      |
|  +================================================================+      |
|                 |                                                         |
|                 v                                                         |
|  +================================================================+      |
|  |      SPARSE MIXTURE OF EXPERTS (MoE) CLASSIFICATION HEAD       |      |
|  |      [OUR NOVEL CONTRIBUTION #2]                                |      |
|  |                                                                  |      |
|  |      768-dim input                                              |      |
|  |           |                                                      |      |
|  |      +----+----+                                                |      |
|  |      |  ROUTER |                                                |      |
|  |      | Linear  |                                                |      |
|  |      | (768->6)|                                                |      |
|  |      | Softmax |                                                |      |
|  |      | Top-2   |                                                |      |
|  |      +----+----+                                                |      |
|  |           |  routing weights (w1, w2)                           |      |
|  |   +-------+-------+-------+-------+-------+-------+            |      |
|  |   |       |       |       |       |       |       |            |      |
|  |   v       v       v       v       v       v       v            |      |
|  | +---+   +---+   +---+   +---+   +---+   +---+                 |      |
|  | |E1 |   |E2 |   |E3 |   |E4 |   |E5 |   |E6 |                 |      |
|  | |Tom|   |Crn|   |Grp|   |Pot|   |Apl|   |Oth|                 |      |
|  | |10 |   | 4 |   | 4 |   | 5 |   | 8 |   | 7 |   classes      |      |
|  | +---+   +---+   +---+   +---+   +---+   +---+                 |      |
|  |   |       |       |       |       |       |                    |      |
|  |   +-------+-------+-------+-------+-------+                   |      |
|  |                   |                                             |      |
|  |       Weighted sum of top-2 expert outputs                     |      |
|  |                   |                                             |      |
|  |                   v                                             |      |
|  |          38-class probability distribution                     |      |
|  |                                                                  |      |
|  |  Loss = L_ce + 0.01 * L_load_balance                           |      |
|  |  (load balance prevents all inputs routing to 1 expert)        |      |
|  +================================================================+      |
|                 |                                                         |
|                 v                                                         |
|  +================================================================+      |
|  |      BAYESIAN UNCERTAINTY (MC-Dropout)                          |      |
|  |      [OUR NOVEL CONTRIBUTION #3]                                |      |
|  |                                                                  |      |
|  |      At inference: keep Dropout(0.2) ENABLED                    |      |
|  |      Run 5 forward passes with different dropout masks          |      |
|  |                                                                  |      |
|  |      predictions = [p1, p2, p3, p4, p5]  (5 x 38 probs)       |      |
|  |      mean_pred   = mean(predictions)                            |      |
|  |      uncertainty  = mean(variance(predictions))                 |      |
|  |                                                                  |      |
|  |      if uncertainty > 0.15:                                     |      |
|  |          output = "UNCERTAIN — retake photo / consult expert"   |      |
|  |      elif max(mean_pred) < 0.60:                                |      |
|  |          output = "LOW CONFIDENCE — try clearer image"          |      |
|  |      else:                                                      |      |
|  |          output = top class + confidence + uncertainty score    |      |
|  +================================================================+      |
|                 |                                                         |
|                 v                                                         |
|  OUTPUT: {class, confidence, uncertainty, expert_id, treatment}           |
|                                                                           |
+===========================================================================+
```

---

## 1.2 — Knowledge Distillation Pipeline (Teacher -> Student)

```
+===========================================================================+
|         HYBRID KNOWLEDGE DISTILLATION WITH ST-DAM TRANSFER                |
|         [OUR NOVEL CONTRIBUTION #4]                                       |
+===========================================================================+
|                                                                           |
|    TEACHER (frozen)                    STUDENT (training)                  |
|    +-----------------------+          +-----------------------+           |
|    | Swin-V2-Tiny          |          | FastViT-SA12          |           |
|    | + ST-DAM (full)       |          | + ST-DAM-Lite         |           |
|    | + MoE Head (6 experts)|          | + MoE-Lite (6 experts)|           |
|    | 28M params            |          | 11.6M params          |           |
|    | ~97% accuracy         |          | ImageNet pretrained   |           |
|    +----------+------------+          +----------+------------+           |
|               |                                  |                        |
|               |    Same input image               |                        |
|               |    fed to both                    |                        |
|               v                                  v                        |
|    +----------+------------+          +----------+------------+           |
|    | Teacher outputs:      |          | Student outputs:      |           |
|    | 1. Logits (38-dim)    |          | 1. Logits (38-dim)    |           |
|    | 2. Backbone attn maps |          | 2. Backbone attn maps |           |
|    | 3. ST-DAM shape maps  |          | 3. ST-DAM shape maps  |           |
|    | 4. ST-DAM texture maps|          | 4. ST-DAM texture maps|           |
|    | 5. MoE routing weights|          | 5. MoE routing weights|           |
|    +----------+------------+          +----------+------------+           |
|               |                                  |                        |
|               +----------------+  +--------------+                        |
|                                |  |                                       |
|                                v  v                                       |
|                     +--------------------+                                |
|                     | QUADRUPLE LOSS:    |                                |
|                     |                    |                                |
|                     | L_total =          |                                |
|                     |   a * L_ce         | (hard label cross-entropy)     |
|                     | + b * L_kd         | (soft logit KL-div, T=4)       |
|                     | + g * L_attn       | (backbone attention MSE)       |
|                     | + d * L_stdam      | (shape+texture map MSE)        |
|                     | + e * L_route      | (MoE routing alignment)        |
|                     |                    |                                |
|                     | a=1.0, b=4.0,      |                                |
|                     | g=2.0, d=2.0,      |                                |
|                     | e=0.5, T=4.0       |                                |
|                     +--------+-----------+                                |
|                              |                                            |
|                              | Backprop through STUDENT only              |
|                              v                                            |
|                     +--------------------+                                |
|                     | Distilled Student  |                                |
|                     | Target: >= 95%     |                                |
|                     | + calibrated uncert|                                |
|                     +--------------------+                                |
|                                                                           |
|   KEY INNOVATION:                                                         |
|   - ST-DAM attention maps are distilled so student learns WHERE           |
|     to look (lesion shapes + disease textures) from teacher               |
|   - MoE routing weights are aligned so student learns WHICH              |
|     expert to activate for which crop family                              |
|   - This is a QUADRUPLE distillation (logit + attn + ST-DAM + route)     |
|     — no existing paper does this                                        |
|                                                                           |
+===========================================================================+
```

---

## 1.3 — Model Export & Mobile Deployment Pipeline

```
+===========================================================================+
|              MODEL EXPORT & QUANTIZATION FOR MOBILE                       |
+===========================================================================+
|                                                                           |
|  DISTILLED FastViT-SA12 + ST-DAM-Lite + MoE-Lite (PyTorch)              |
|       |                                                                   |
|       |  Step 1: Structural Reparameterization                            |
|       |  (Fuse RepMixer multi-branch -> single conv)                      |
|       v                                                                   |
|  fastvit_agriswin_fused.pth (~45 MB FP32)                                |
|       |                                                                   |
|       |  Step 2: Quantization-Aware Training (QAT)                        |
|       |  (Retrain 5 epochs with fake quantization nodes)                   |
|       v                                                                   |
|  fastvit_agriswin_qat.pth (~45 MB but quantization-ready)                |
|       |                                                                   |
|       |  Step 3: ONNX Export + Simplify                                   |
|       v                                                                   |
|  agriswin_moe.onnx (~42 MB) -> onnx-simplifier -> ~40 MB                |
|       |                                                                   |
|       |  Step 4: TFLite Conversion + INT8 Quantization                    |
|       v                                                                   |
|  +----------------------------------------------+                        |
|  | FINAL MOBILE ARTIFACTS:                       |                        |
|  |                                               |                        |
|  | agriswin_moe_int8.tflite      ~10 MB          |                        |
|  | class_labels.json                2 KB          |                        |
|  | disease_treatments.db          200 KB          |                        |
|  | gemma_3n_e2b_q4.task          ~1.0 GB          | (optional)             |
|  |                                               |                        |
|  | CORE BUNDLE (without SLM):    ~10.2 MB         | <-- any phone          |
|  | FULL BUNDLE (with SLM):       ~1.1 GB          | <-- 4GB+ RAM phones    |
|  +----------------------------------------------+                        |
|                                                                           |
|  MC-DROPOUT ON MOBILE:                                                    |
|  +----------------------------------------------+                        |
|  | 5 forward passes with dropout enabled         |                        |
|  | Each pass: ~30-40 ms on mid-range phone CPU   |                        |
|  | Total: ~150-200 ms for uncertainty estimate    |                        |
|  | Still under 200ms target for "instant" feel    |                        |
|  +----------------------------------------------+                        |
|                                                                           |
+===========================================================================+
```

---

## 1.4 — Training Configuration (Google Colab)

```
+===========================================================================+
|                  TRAINING HYPERPARAMETERS (FINAL)                         |
+===========================================================================+
|                                                                           |
|  STAGE A: TRAIN TEACHER (Swin-V2-T + ST-DAM + MoE)   [Colab A100/T4]   |
|  +----------------------------------+-----------------------------+      |
|  | Input Resolution                 | 224 x 224 x 3              |      |
|  | Backbone                         | Swin-V2-Tiny (ImageNet)    |      |
|  | Novel Module                     | ST-DAM (Shape-Texture)     |      |
|  | Classification Head              | MoE (6 experts, top-2)     |      |
|  | Number of Classes                | 38                         |      |
|  | Batch Size                       | 32 (Colab T4)              |      |
|  | Phase 1 (head + ST-DAM only)     | lr=1e-3, 5 epochs          |      |
|  | Phase 2 (full fine-tune)         | lr=5e-5, 35 epochs         |      |
|  | Optimizer                        | AdamW (wd=0.05)            |      |
|  | Scheduler                        | CosineAnnealingWarmRestarts|      |
|  | Loss                             | CE + LabelSmooth(0.1)      |      |
|  |                                  | + 0.01*LoadBalance         |      |
|  | Augmentation                     | RandAugment + Mixup(0.8)   |      |
|  |                                  | + CutMix(1.0) + RandomErase|     |
|  | Mixed Precision                  | FP16 (torch.cuda.amp)      |      |
|  | Early Stopping                   | patience=10, monitor=val_F1|      |
|  | MC-Dropout rate                  | 0.2 (kept enabled)         |      |
|  +----------------------------------+-----------------------------+      |
|  | TARGET:  >= 96% accuracy, >= 0.95 macro-F1                     |      |
|  +----------------------------------------------------------------+      |
|                                                                           |
|  STAGE B: KNOWLEDGE DISTILLATION (Teacher -> FastViT Student)            |
|  +----------------------------------+-----------------------------+      |
|  | Student Backbone                 | FastViT-SA12 (ImageNet)    |      |
|  | Student Novel Module             | ST-DAM-Lite (lighter ver.) |      |
|  | Student Head                     | MoE-Lite (6 experts, top-2)|     |
|  | Temperature (T)                  | 4.0                        |      |
|  | Distillation Epochs              | 50                         |      |
|  | Learning Rate                    | 3e-4 (cosine)              |      |
|  | Loss Weights                     | a=1, b=4, g=2, d=2, e=0.5 |      |
|  | Adapter Layers                   | 1x1 Conv (channel align)   |      |
|  +----------------------------------+-----------------------------+      |
|  | TARGET: >= 95% accuracy (within 1-2% of teacher)               |      |
|  +----------------------------------------------------------------+      |
|                                                                           |
+===========================================================================+
```

---
---

# =====================================================
# PART 2: MOBILE APPLICATION (Updated)
# =====================================================

## 2.1 — Complete On-Device Inference Pipeline

```
+===========================================================================+
|            ON-DEVICE INFERENCE PIPELINE (AgriSwin-MoE Mobile)             |
+===========================================================================+
|                                                                           |
|  USER CAPTURES/SELECTS PHOTO                                              |
|       |                                                                   |
|       v                                                                   |
|  +-------------------------------+                                        |
|  | STEP 1: IMAGE PREPROCESSING   |  Time: ~15 ms                         |
|  | - Decode JPEG to RGB bitmap   |                                        |
|  | - Resize to 256x256 (bilinear)|                                        |
|  | - Center crop to 224x224      |                                        |
|  | - Normalize (ImageNet stats)  |                                        |
|  +---------------+---------------+                                        |
|                  |                                                        |
|                  v                                                        |
|  +-------------------------------+                                        |
|  | STEP 2: TFLite INFERENCE      |  Time: ~35 ms (per pass)              |
|  | AgriSwin-MoE INT8 model       |                                        |
|  |                               |                                        |
|  | FastViT-SA12 backbone         |                                        |
|  |     -> ST-DAM-Lite            |                                        |
|  |     -> MoE-Lite (top-2 route) |                                        |
|  |     -> 38-class output        |                                        |
|  +---------------+---------------+                                        |
|                  |                                                        |
|                  v                                                        |
|  +-------------------------------+                                        |
|  | STEP 3: MC-DROPOUT UNCERTAINTY|  Time: ~175 ms (5 passes)             |
|  | [OUR NOVEL CONTRIBUTION #3]   |                                        |
|  |                               |                                        |
|  | Run model 5 times with        |                                        |
|  | different dropout masks        |                                        |
|  |                               |                                        |
|  | preds = [p1, p2, p3, p4, p5]  |                                        |
|  | mean_pred = mean(preds)        |                                        |
|  | epistemic_uncertainty =        |                                        |
|  |   mean(var(preds, axis=0))    |                                        |
|  |                               |                                        |
|  | Decision logic:               |                                        |
|  | +---------------------------+ |                                        |
|  | | uncertainty > 0.15        | |                                        |
|  | |  -> "UNCERTAIN: retake"   | |                                        |
|  | |                           | |                                        |
|  | | max(mean_pred) < 0.60     | |                                        |
|  | |  -> "LOW CONFIDENCE"      | |                                        |
|  | |                           | |                                        |
|  | | else                      | |                                        |
|  | |  -> confident prediction  | |                                        |
|  | +---------------------------+ |                                        |
|  +---------------+---------------+                                        |
|                  |                                                        |
|                  v                                                        |
|  +-------------------------------+                                        |
|  | STEP 4: TREATMENT ADVISORY    |  Time: ~5ms (DB) / ~2s (SLM)          |
|  |                               |                                        |
|  | PATH A (Instant - SQLite):    |                                        |
|  |   Query disease_treatments.db |                                        |
|  |   Return: chemical, organic,  |                                        |
|  |   dosage, prevention, severity|                                        |
|  |                               |                                        |
|  | PATH B (Rich - Gemma 3n SLM): |                                        |
|  |   [OUR NOVEL CONTRIBUTION #5] |                                        |
|  |   Prompt: "Disease: {name},   |                                        |
|  |   Confidence: {conf}%,        |                                        |
|  |   Crop: {crop}, Season: {s}   |                                        |
|  |   Provide treatment in {lang}"|                                        |
|  |   -> Contextual, personalized |                                        |
|  |      advice in Hindi/Telugu/  |                                        |
|  |      Tamil/English            |                                        |
|  +---------------+---------------+                                        |
|                  |                                                        |
|                  v                                                        |
|  +-------------------------------+                                        |
|  | STEP 5: RESULTS DISPLAY       |  Time: ~20 ms                         |
|  |                               |                                        |
|  | +---------------------------+ |                                        |
|  | | Disease: Tomato Late      | |                                        |
|  | |          Blight            | |                                        |
|  | | Confidence: 94.2%         | |                                        |
|  | | Uncertainty: LOW (0.03)   | |                                        |
|  | | Expert: Tomato (E1)       | |                                        |
|  | | Severity: CRITICAL        | |                                        |
|  | |                           | |                                        |
|  | | [Treatment Card]          | |                                        |
|  | | [Prevention Tips]         | |                                        |
|  | | [Uncertainty Indicator]   | |                                        |
|  | | [Save] [Share] [Rescan]   | |                                        |
|  | +---------------------------+ |                                        |
|  +-------------------------------+                                        |
|                                                                           |
|  TOTAL END-TO-END:                                                        |
|  +----------------------------------------------+                        |
|  | Without SLM:  ~210 ms  (5-pass MC-Dropout)    |                        |
|  | With SLM:     ~2.2 sec (includes treatment)    |                        |
|  | Single-pass:  ~50 ms   (no uncertainty)         |                        |
|  +----------------------------------------------+                        |
|                                                                           |
+===========================================================================+
```

---

## 2.2 — Offline-First App Architecture with Protocols

```
+===========================================================================+
|           MOBILE APP ARCHITECTURE (OFFLINE-FIRST + MCP/A2A)               |
+===========================================================================+
|                                                                           |
|  +============================= OFFLINE CORE ==========================+ |
|  |                                                                      | |
|  |  +----------+   +------------------+   +----------+   +----------+  | |
|  |  | Camera / |-->| AgriSwin-MoE     |-->| MC-Drop  |-->| SQLite / |  | |
|  |  | Gallery  |   | TFLite INT8      |   | Uncert.  |   | Gemma 3n |  | |
|  |  | Input    |   | (FastViT+STDAM   |   | (5 pass) |   | SLM Adv. |  | |
|  |  |          |   |  +MoE-Lite)      |   |          |   |          |  | |
|  |  +----------+   +------------------+   +----------+   +----------+  | |
|  |       |                                                     |        | |
|  |       +----------+     SCAN HISTORY (SQLite)     +----------+        | |
|  |                  |                               |                   | |
|  |                  v                               v                   | |
|  |            +----------------------------------------+                | |
|  |            | LOCAL STORAGE                          |                | |
|  |            | - scan_history table                   |                | |
|  |            | - disease_treatments table             |                | |
|  |            | - user_preferences table               |                | |
|  |            | - model_metadata table                 |                | |
|  |            +----------------------------------------+                | |
|  |                                                                      | |
|  +========================== 100% OFFLINE =============================+ |
|                              |                                           |
|                              | When connectivity available               |
|                              v                                           |
|  +======================== ONLINE MODE ================================+ |
|  |                                                                      | |
|  |  MCP (Model Context Protocol) LAYER                                  | |
|  |  +----------------------------------------------------------------+  | |
|  |  | On-device MCP Client exposes local tools:                      |  | |
|  |  |   - tool: "get_scan_history" (read local SQLite)               |  | |
|  |  |   - tool: "get_weather_data" (fetch from OpenWeather)          |  | |
|  |  |   - tool: "check_model_update" (check remote version)         |  | |
|  |  |   - tool: "sync_scans_to_cloud" (upload scan logs)            |  | |
|  |  +----------------------------------------------------------------+  | |
|  |                              |                                        | |
|  |  A2A (Agent-to-Agent) LAYER                                          | |
|  |  +----------------------------------------------------------------+  | |
|  |  | Mobile Agent <-- A2A Protocol --> Cloud Agent Swarm             |  | |
|  |  |                                                                |  | |
|  |  | +------------------+  +-------------------+  +---------------+ |  | |
|  |  | | Weather Agent    |  | Expert Consult    |  | Model Update  | |  | |
|  |  | | (OpenMeteo API)  |  | Agent (community  |  | Agent (check  | |  | |
|  |  | |                  |  | forum relay)      |  | new .tflite)  | |  | |
|  |  | +------------------+  +-------------------+  +---------------+ |  | |
|  |  |                                                                |  | |
|  |  | Agent Cards (JSON) for discovery:                              |  | |
|  |  | /.well-known/agent-card.json on each endpoint                  |  | |
|  |  +----------------------------------------------------------------+  | |
|  |                                                                      | |
|  +======================================================================+ |
|                                                                           |
+===========================================================================+
```

---

## 2.3 — Project File Structure (Complete)

```
PlantDiseaseDetection/
|
|-- ml_pipeline/                           # PART 1: Model Training
|   |-- notebooks/
|   |   |-- 01_data_exploration.ipynb
|   |   |-- 02_preprocessing_augmentation.ipynb
|   |   |-- 03_train_teacher_swinv2.ipynb
|   |   |-- 04_train_stdam_module.ipynb       # [NOVEL] ST-DAM training
|   |   |-- 05_train_moe_head.ipynb           # [NOVEL] MoE head training
|   |   |-- 06_knowledge_distillation.ipynb   # [NOVEL] Quadruple KD
|   |   |-- 07_uncertainty_calibration.ipynb   # [NOVEL] MC-Dropout calib
|   |   |-- 08_evaluation_gradcam.ipynb
|   |   |-- 09_export_tflite.ipynb
|   |   +-- 10_generate_treatment_db.ipynb    # [NOVEL] Llama 3.1 gen
|   |
|   |-- src/
|   |   |-- models/
|   |   |   |-- swin_v2_teacher.py
|   |   |   |-- fastvit_student.py
|   |   |   |-- stdam.py                      # [NOVEL] Shape-Texture module
|   |   |   |-- stdam_lite.py                 # [NOVEL] Lightweight version
|   |   |   |-- moe_head.py                   # [NOVEL] MoE classification
|   |   |   |-- moe_lite.py                   # [NOVEL] Lightweight MoE
|   |   |   +-- agriswin_moe.py               # [NOVEL] Full model assembly
|   |   |
|   |   |-- training/
|   |   |   |-- trainer.py
|   |   |   |-- distiller.py                  # [NOVEL] Quadruple distill
|   |   |   |-- losses.py                     # CE + KD + Attn + STDAM + Route
|   |   |   +-- uncertainty.py                # [NOVEL] MC-Dropout calibration
|   |   |
|   |   |-- evaluation/
|   |   |   |-- metrics.py
|   |   |   |-- gradcam.py
|   |   |   |-- uncertainty_eval.py           # [NOVEL] ECE, reliability diag
|   |   |   +-- domain_shift_test.py          # PlantVillage -> PlantDoc OOD
|   |   |
|   |   +-- export/
|   |       |-- to_onnx.py
|   |       |-- to_tflite.py
|   |       +-- quantize_qat.py
|   |
|   +-- requirements.txt
|
|-- mobile_app/                            # PART 2: React Native App
|   |-- src/
|   |   |-- screens/
|   |   |   |-- HomeScreen.tsx
|   |   |   |-- CameraScreen.tsx
|   |   |   |-- ResultScreen.tsx              # Shows uncertainty indicator
|   |   |   |-- HistoryScreen.tsx
|   |   |   +-- SettingsScreen.tsx
|   |   |
|   |   |-- services/
|   |   |   |-- AgriSwinService.ts            # [NOVEL] TFLite + MC-Dropout
|   |   |   |-- UncertaintyService.ts         # [NOVEL] 5-pass uncertainty
|   |   |   |-- GemmaSLMService.ts            # [NOVEL] On-device SLM
|   |   |   |-- TreatmentService.ts
|   |   |   |-- MCPClientService.ts           # [NOVEL] MCP tool client
|   |   |   +-- A2AAgentService.ts            # [NOVEL] A2A protocol
|   |   |
|   |   |-- components/
|   |   |   |-- UncertaintyGauge.tsx           # [NOVEL] Visual uncertainty
|   |   |   |-- MoEExpertBadge.tsx             # Shows which expert fired
|   |   |   |-- TreatmentCard.tsx
|   |   |   +-- OfflineBadge.tsx
|   |   |
|   |   +-- assets/
|   |       |-- models/
|   |       |   |-- agriswin_moe_int8.tflite   # ~10 MB
|   |       |   +-- class_labels.json
|   |       +-- database/
|   |           +-- treatments.db              # Pre-generated via Llama
|   |
|   +-- package.json
|
|-- research_papers/                       # Downloaded 20 papers
|   +-- papers_index.md
|
|-- data/
|   |-- raw/
|   |-- processed/
|   +-- metadata/
|
+-- README.md
```

---

## 2.4 — Project Timeline (Updated)

```
+===========================================================================+
|                    PROJECT TIMELINE (10-12 WEEKS)                         |
+===========================================================================+
|                                                                           |
|  PART 1: ML MODEL (AgriSwin-MoE)                                        |
|  ================================                                        |
|                                                                           |
|  Week 1-2  ##########  Data collection, cleaning, EDA, splits            |
|  Week 3    ##########  Train Swin-V2-T teacher (baseline)                |
|  Week 4    ##########  Design & train ST-DAM module [NOVEL]              |
|  Week 5    ##########  Design & train MoE head [NOVEL]                   |
|  Week 6    ##########  Knowledge Distillation to FastViT [NOVEL]         |
|  Week 7    ##########  MC-Dropout calibration + eval [NOVEL]             |
|  Week 7    #####.....  TFLite export + quantization + Treatment DB gen   |
|                                                                           |
|  PART 2: MOBILE APP                                                      |
|  ==================                                                      |
|                                                                           |
|  Week 7    .....#####  React Native setup, scaffolding (overlap)         |
|  Week 8    ##########  Camera + TFLite inference + MC-Dropout on device  |
|  Week 9    ##########  Gemma 3n SLM integration + SQLite treatment DB    |
|  Week 10   ##########  MCP/A2A protocols + UI polish + uncertainty gauge |
|  Week 11   ##########  End-to-end testing, real Android benchmarks       |
|  Week 12   ##########  APK build, paper writing, demo preparation        |
|                                                                           |
|  MILESTONES:                                                              |
|  M1 (Week 2):  Dataset ready with stratified splits                      |
|  M2 (Week 5):  Teacher model >= 96% with ST-DAM + MoE                   |
|  M3 (Week 7):  Distilled student >= 95%, TFLite exported                 |
|  M4 (Week 9):  Core app working: scan -> detect -> uncertainty -> treat  |
|  M5 (Week 11): Full offline system tested on real Android phone          |
|  M6 (Week 12): IEEE paper draft + demo APK ready                         |
|                                                                           |
+===========================================================================+
```

---

## IEEE PAPER CONTRIBUTION SUMMARY

```
+===========================================================================+
|              IEEE PAPER — NOVELTY CONTRIBUTION CHECKLIST                  |
+===========================================================================+
|                                                                           |
|  [x] Novel ST-DAM module for vision transformers in agriculture          |
|      (No prior work combines Sobel shape + Gabor texture attention       |
|       with Swin Transformer for plant disease)                            |
|                                                                           |
|  [x] First Sparse MoE classification head for plant disease              |
|      (Crop-family routing eliminates cross-crop confusion)                |
|                                                                           |
|  [x] Bayesian uncertainty estimation on mobile via MC-Dropout            |
|      (No prior plant disease work provides calibrated uncertainty        |
|       on edge devices — critical for farmer trust)                        |
|                                                                           |
|  [x] Quadruple knowledge distillation (logit + attn + ST-DAM + route)   |
|      (First to distill decoupled shape-texture attention maps)            |
|                                                                           |
|  [x] On-device SLM (Gemma 3n) for offline treatment advisory            |
|      (First integration of open-source on-device LLM for                 |
|       contextual, multilingual agricultural advice)                       |
|                                                                           |
|  [x] Complete end-to-end offline mobile pipeline with real benchmarks    |
|      (Detect -> Classify -> Uncertainty -> Treatment -> History          |
|       all running 100% offline on actual Android hardware)                |
|                                                                           |
|  [x] Cross-dataset generalization study (PlantVillage -> PlantDoc)       |
|      (Addresses the #1 gap identified across all 20 papers)              |
|                                                                           |
|  [x] Modern protocol integration (MCP + A2A) for agricultural IoT       |
|      (First application of these protocols in precision agriculture)      |
|                                                                           |
+===========================================================================+
|                                                                           |
|  COMPARISON TABLE FOR PAPER:                                              |
|                                                                           |
|  +------+----------+-------+------+-------+--------+--------+-------+   |
|  | Work | Backbone | Shape | MoE  | Uncrt | On-Dev | Offln  | E2E   |   |
|  |      |          | Textr |      |       | SLM    | Treat  | Mobile|   |
|  +------+----------+-------+------+-------+--------+--------+-------+   |
|  | [01] | ViT      |  No   |  No  |  No   |  No    |  No    | Partial|  |
|  | [02] | TinyCNN  |  No   |  No  |  No   |  No    |  No    |  No    |  |
|  | [03] | Swin+MNv3|  No   |  No  |  No   |  No    |  No    | Partial|  |
|  | [07] | CNN-MViT |  No   |  No  |  No   |  No    |  No    |  No    |  |
|  | [09] | CNN+LLM  |  No   |  No  |  No   | Cloud  | Cloud  |  No    |  |
|  | [16] | ViT+CNN  |  No   |  No  |  No   |  No    |  No    |  No    |  |
|  | [19] | NAS-CNN  | Shape |  No  |  No   |  No    |  No    |  No    |  |
|  | [20] | MobViT   |  No   |  No  |  No   |  No    |  No    |  No    |  |
|  +------+----------+-------+------+-------+--------+--------+-------+   |
|  | OURS | Swin-V2  | BOTH  | YES  | YES   | YES    | YES    | YES    |  |
|  |      | +FastViT | S+T   | 6exp | MC-Dr | Gemma  | Offln  | Full   |  |
|  |      | +ST-DAM  | Cross | Top2 | 5pass | 3n E2B | SQLite | Android|  |
|  |      | +MoE+KD  | Attn  | Sprs | Bayes | Local  | +SLM   | APK    |  |
|  +------+----------+-------+------+-------+--------+--------+-------+   |
|                                                                           |
+===========================================================================+
```

---

## DATASETS (Same as Before — Official Sources Only)

```
+===========================================================================+
|                         DATASETS (OFFICIAL LINKS)                         |
+===========================================================================+
|                                                                           |
|  PRIMARY:                                                                |
|  1. PlantVillage (54,305 images, 38 classes)                             |
|     GitHub: https://github.com/spMohanty/PlantVillage-Dataset            |
|     Paper:  https://arxiv.org/abs/1511.08060                             |
|                                                                           |
|  SUPPLEMENTARY:                                                          |
|  2. PlantDoc (2,598 real-world field images, 27 classes)                 |
|     GitHub:  https://github.com/pratikkayal/PlantDoc-Dataset             |
|     Paper:   https://dl.acm.org/doi/10.1145/3371158.3371196              |
|                                                                           |
|  3. iBean (1,296 images, 3 classes)                                      |
|     TF Datasets: https://www.tensorflow.org/datasets/catalog/beans       |
|     GitHub:      https://github.com/AI-Lab-Makerere/ibean                |
|                                                                           |
|  4. Mendeley Augmented PlantVillage (87,000+ images, 38 classes)         |
|     Mendeley: https://data.mendeley.com/datasets/tywbtsjrjv/1           |
|                                                                           |
|  TREATMENT DATA:                                                         |
|  5. EPPO Global Database (pest/disease reference)                        |
|     URL: https://gd.eppo.int/                                            |
|                                                                           |
+===========================================================================+
```
