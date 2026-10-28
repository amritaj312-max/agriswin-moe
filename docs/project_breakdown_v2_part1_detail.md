
---
---

# =====================================================
# PART 1: BUILD THE ML/CV MODEL (Modern Stack)
# =====================================================

---

## 1.1 — Why These Models? (Decision Rationale)

```
+===========================================================================+
|              MODEL SELECTION — WHY EACH TECHNOLOGY?                       |
+===========================================================================+
|                                                                           |
|  WHY SWIN TRANSFORMER V2 (Teacher)?                                      |
|  +-------------------------------------------------------------------+   |
|  | - Hierarchical shifted-window self-attention                      |   |
|  | - SOTA on ImageNet-1K (84.6% top-1 for Tiny variant)             |   |
|  | - Captures BOTH local texture + global context of leaf patterns   |   |
|  | - Cosine attention + log-spaced CPB = better generalization       |   |
|  | - Open source (MIT License): Microsoft Research                   |   |
|  | - Trainable on Colab T4 (28M params, ~3.5 GB VRAM for batch=16)  |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
|  WHY FastViT-SA12 (Student / Mobile)?                                    |
|  +-------------------------------------------------------------------+   |
|  | - Apple's hybrid CNN-Transformer with RepMixer                    |   |
|  | - Structural reparameterization: multi-branch at train time,      |   |
|  |   fused single-branch at inference = FAST mobile execution       |   |
|  | - 11.6M params, 1.8 GFLOPs — sweet spot for mobile               |   |
|  | - 3.0ms on iPhone 12 NPU, ~8ms on ARM CPU                        |   |
|  | - Clean ONNX/TFLite export path (no custom ops)                   |   |
|  | - Open source (Apple License)                                     |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
|  WHY RT-DETRv2 (Detection)?                                              |
|  +-------------------------------------------------------------------+   |
|  | - First real-time end-to-end detection transformer                |   |
|  | - NMS-free = simpler mobile export (no post-processing hacks)     |   |
|  | - R18 backbone variant fits mobile: ~20M params                   |   |
|  | - Discrete sampling ops = TFLite-friendly (no grid_sample)        |   |
|  | - Detects WHICH leaf region is diseased (not just "sick/healthy") |   |
|  | - Open source (Apache 2.0)                                       |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
|  WHY MoE Classification Head?                                             |
|  +-------------------------------------------------------------------+   |
|  | - 6 expert sub-networks, each specializing in a crop family       |   |
|  |   Expert 1: Tomato diseases (10 classes)                          |   |
|  |   Expert 2: Corn/Maize diseases (4 classes)                       |   |
|  |   Expert 3: Grape diseases (4 classes)                            |   |
|  |   Expert 4: Potato/Pepper diseases (5 classes)                    |   |
|  |   Expert 5: Apple/Cherry/Peach diseases (8 classes)               |   |
|  |   Expert 6: Other crops (7 classes)                               |   |
|  | - Top-2 gating: only 2 experts fire per image = sparse compute    |   |
|  | - Higher capacity than single dense head, same inference cost     |   |
|  | - Reduces confusion between visually similar cross-crop diseases  |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
|  WHY Gemma 3n E2B (On-Device SLM)?                                       |
|  +-------------------------------------------------------------------+   |
|  | - Google DeepMind, open weights (Gemma license)                   |   |
|  | - ~2B effective params via MatFormer nested architecture          |   |
|  | - Runs on phones with 2-4 GB RAM (PLE offloads to CPU)            |   |
|  | - Multimodal: can see the leaf image + generate text advice       |   |
|  | - 100% offline via LiteRT (formerly MediaPipe)                    |   |
|  | - Generates contextual, farmer-friendly treatment explanations    |   |
|  | - Supports Hindi, Telugu, Tamil + 30 languages for rural India    |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
|  WHY Knowledge Distillation (not just train FastViT directly)?           |
|  +-------------------------------------------------------------------+   |
|  | - Swin-V2-T achieves ~97% on PlantVillage (rich representations) |   |
|  | - Direct FastViT training: ~92-93% (misses fine-grained patterns) |   |
|  | - Distillation transfers attention maps + soft logits = ~95-96%   |   |
|  | - 2-3% accuracy GAIN for FREE (no extra mobile inference cost)    |   |
|  | - Attention distillation teaches FastViT WHERE to look on leaf    |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
+===========================================================================+
```

---

## 1.2 — Complete Training Pipeline

```
+===========================================================================+
|               TRAINING PIPELINE — MULTI-STAGE ON GOOGLE COLAB             |
+===========================================================================+

  STAGE 1: DATA PREPARATION
  =========================

  Download & Merge Datasets
       |
       v
  +------------------------------------------------------------------+
  |  PlantVillage (54,305)  +  PlantDoc (2,598)  +  iBean (1,296)   |
  |  + Mendeley Augmented (32,700 additional)                        |
  |  = ~90,899 total images across 38+ classes                       |
  +------------------------------------------------------------------+
       |
       v
  +------------------------------------------------------------------+
  |  CLEANING PIPELINE                                                |
  |  1. Remove corrupted / truncated images (PIL verify)             |
  |  2. Remove exact duplicates (perceptual hashing, dHash)          |
  |  3. Normalize filenames, fix label inconsistencies               |
  |  4. Resize to 256x256 (preserve aspect, pad if needed)          |
  |  5. Verify class distribution, flag imbalanced classes           |
  +------------------------------------------------------------------+
       |
       v
  +------------------------------------------------------------------+
  |  STRATIFIED SPLIT (per class)                                    |
  |                                                                   |
  |  Train  70%  ############################............  ~63,629   |
  |  Val    15%  ############............................ ~13,635    |
  |  Test   15%  ############............................ ~13,635    |
  |                                                                   |
  |  Hold-out: PlantDoc real-world images for OOD test               |
  +------------------------------------------------------------------+


  STAGE 2: TRAIN THE TEACHER (Swin Transformer V2-Tiny)
  =====================================================

  Google Colab (T4 16GB or A100 40GB)
       |
       v
  +------------------------------------------------------------------+
  |  MODEL: Swin-V2-Tiny                                              |
  |  Pretrained: ImageNet-1K (84.6% top-1)                           |
  |  Modified: Replace head -> Dense(38, softmax)                     |
  |                                                                   |
  |  TRAINING CONFIG:                                                 |
  |  +-----------------------------+------------------------------+  |
  |  |  Input Resolution           |  224 x 224 x 3               |  |
  |  |  Window Size                |  7 (shifted windows)         |  |
  |  |  Batch Size                 |  32 (Colab T4)               |  |
  |  |  Phase 1 (head only)        |  lr=1e-3, 5 epochs           |  |
  |  |  Phase 2 (full fine-tune)   |  lr=5e-5, 30 epochs          |  |
  |  |  Optimizer                  |  AdamW (wd=0.05)             |  |
  |  |  Scheduler                  |  CosineAnnealingWarmRestarts |  |
  |  |  Loss                       |  CE + Label Smoothing (0.1)  |  |
  |  |  Augmentation               |  RandAugment + Mixup + CutMix|  |
  |  |  Mixed Precision            |  FP16 (torch.cuda.amp)       |  |
  |  |  Early Stopping             |  patience=10, monitor=val_F1 |  |
  |  +-----------------------------+------------------------------+  |
  |                                                                   |
  |  TARGET: >= 96% val accuracy, >= 0.95 macro F1                   |
  +------------------------------------------------------------------+
       |
       |  Save best checkpoint: swin_v2_teacher.pth
       v


  STAGE 3: TRAIN RT-DETRv2 (Disease Region Detection)
  ====================================================

  Google Colab (T4/A100)
       |
       v
  +------------------------------------------------------------------+
  |  MODEL: RT-DETRv2 with ResNet-18 backbone                        |
  |  Task: Detect diseased regions / lesions on leaf images           |
  |                                                                   |
  |  Annotation: Convert classification data to detection format     |
  |  - Healthy leaves: no bounding box                               |
  |  - Diseased leaves: annotate lesion regions (LabelImg/CVAT)      |
  |  - Can use GradCAM from Stage 2 to semi-auto generate boxes     |
  |                                                                   |
  |  TRAINING CONFIG:                                                 |
  |  +-----------------------------+------------------------------+  |
  |  |  Input Resolution           |  416 x 416 (mobile-friendly) |  |
  |  |  Backbone                   |  ResNet-18 (lightweight)     |  |
  |  |  Decoder Layers             |  3 (reduced from 6)          |  |
  |  |  Queries                    |  100                         |  |
  |  |  Batch Size                 |  16                          |  |
  |  |  Epochs                     |  50                          |  |
  |  |  Optimizer                  |  AdamW, lr=1e-4              |  |
  |  +-----------------------------+------------------------------+  |
  |                                                                   |
  |  Export: ONNX -> TFLite (INT8), target ~8 MB                     |
  +------------------------------------------------------------------+
       |
       v


  STAGE 4: KNOWLEDGE DISTILLATION (Swin-V2 -> FastViT)
  =====================================================

  +------------------------------------------------------------------+
  |                                                                   |
  |  TEACHER (frozen)              STUDENT (training)                 |
  |  +-----------------+          +-----------------+                 |
  |  | Swin-V2-Tiny    |          | FastViT-SA12    |                 |
  |  | 28M params      |          | 11.6M params    |                 |
  |  | ~97% accuracy   |          | ImageNet init   |                 |
  |  +-------+---------+          +-------+---------+                 |
  |          |                            |                           |
  |          |   Same input image         |                           |
  |          |   fed to both              |                           |
  |          v                            v                           |
  |  +---------------+           +---------------+                    |
  |  | Teacher       |           | Student       |                    |
  |  | Logits (38)   |           | Logits (38)   |                    |
  |  | Attn Maps     |           | Attn Maps     |                    |
  |  +-------+-------+           +-------+-------+                    |
  |          |                            |                           |
  |          +----------+  +--------------+                           |
  |                     |  |                                          |
  |                     v  v                                          |
  |             +------------------+                                  |
  |             | HYBRID LOSS:     |                                  |
  |             |                  |                                  |
  |             | L = a * L_ce     |  (hard label cross-entropy)      |
  |             |   + b * L_kd     |  (soft logit KL-divergence)      |
  |             |   + g * L_attn   |  (attention map MSE loss)        |
  |             |                  |                                  |
  |             | a=1.0, b=4.0     |                                  |
  |             | g=2.0, T=4.0     |                                  |
  |             +--------+---------+                                  |
  |                      |                                            |
  |                      | Backprop through STUDENT only              |
  |                      v                                            |
  |             +------------------+                                  |
  |             | Student updated  |                                  |
  |             | Target: >= 95%   |                                  |
  |             +------------------+                                  |
  |                                                                   |
  |  DISTILLATION CONFIG:                                             |
  |  +-----------------------------+------------------------------+  |
  |  |  Temperature (T)            |  4.0                          |  |
  |  |  Alpha (hard label weight)  |  1.0                          |  |
  |  |  Beta  (logit KD weight)    |  4.0                          |  |
  |  |  Gamma (attention weight)   |  2.0                          |  |
  |  |  Epochs                     |  40                           |  |
  |  |  Learning Rate              |  3e-4 (cosine decay)         |  |
  |  |  Adapter Layers             |  1x1 Conv (channel align)    |  |
  |  +-----------------------------+------------------------------+  |
  +------------------------------------------------------------------+
       |
       v


  STAGE 5: ADD MoE CLASSIFICATION HEAD
  =====================================

  +------------------------------------------------------------------+
  |                                                                   |
  |  FastViT-SA12 Backbone (distilled)                                |
  |       |                                                           |
  |       | Global Average Pooling -> 768-dim feature vector          |
  |       v                                                           |
  |  +----------------------------------------------------------+    |
  |  |           MIXTURE OF EXPERTS (MoE) HEAD                   |    |
  |  |                                                           |    |
  |  |  +--------+    +----------+                               |    |
  |  |  | Router |    | Gating   |   Input: 768-dim features     |    |
  |  |  | Network|--->| Softmax  |   Output: top-2 expert scores |    |
  |  |  | (MLP)  |    | (top-2)  |                               |    |
  |  |  +--------+    +----+-----+                               |    |
  |  |                     |                                     |    |
  |  |        +------+-----+-----+------+------+------+         |    |
  |  |        |      |           |      |      |      |         |    |
  |  |        v      v           v      v      v      v         |    |
  |  |     +-----+ +-----+   +-----+ +-----+ +-----+ +-----+  |    |
  |  |     | E1  | | E2  |   | E3  | | E4  | | E5  | | E6  |  |    |
  |  |     |Tomato| |Corn |   |Grape| |Pot/ | |Apple| |Other|  |    |
  |  |     |10cls| |4cls |   |4cls | |Pepr | |Chry | |7cls |  |    |
  |  |     |     | |     |   |     | |5cls | |8cls | |     |  |    |
  |  |     +--+--+ +--+--+   +--+--+ +--+--+ +--+--+ +--+--+  |    |
  |  |        |      |           |      |      |      |         |    |
  |  |        +------+-----------+------+------+------+         |    |
  |  |                          |                                |    |
  |  |                          v                                |    |
  |  |              +---------------------+                      |    |
  |  |              | Weighted Sum of     |                      |    |
  |  |              | Top-2 Expert Outputs|                      |    |
  |  |              | (sparse activation) |                      |    |
  |  |              +----------+----------+                      |    |
  |  |                         |                                 |    |
  |  |                         v                                 |    |
  |  |              +---------------------+                      |    |
  |  |              | Final: 38 classes   |                      |    |
  |  |              | + confidence score  |                      |    |
  |  |              +---------------------+                      |    |
  |  |                                                           |    |
  |  | Each expert: Linear(768->256) -> ReLU -> Dropout(0.2)    |    |
  |  |              -> Linear(256->N_classes_for_expert)         |    |
  |  | Router:      Linear(768->6) -> Softmax -> Top-2          |    |
  |  | Load Balancing Loss added to prevent expert collapse      |    |
  |  +----------------------------------------------------------+    |
  +------------------------------------------------------------------+
       |
       v


  STAGE 6: MODEL EXPORT & QUANTIZATION
  =====================================

  +------------------------------------------------------------------+
  |                                                                   |
  |  TRAINED FastViT-SA12 + MoE Head (PyTorch)                       |
  |       |                                                           |
  |       |  Step 1: Structural Reparameterization                    |
  |       |  (Fuse multi-branch RepMixer into single convolutions)    |
  |       v                                                           |
  |  fastvit_fused.pth (~45 MB, FP32)                                |
  |       |                                                           |
  |       |  Step 2: Export to ONNX                                   |
  |       |  torch.onnx.export(opset=13, input=[1,3,224,224])        |
  |       v                                                           |
  |  fastvit_moe.onnx (~45 MB)                                       |
  |       |                                                           |
  |       |  Step 3: ONNX Simplify                                    |
  |       |  onnx-simplifier (remove redundant ops)                   |
  |       v                                                           |
  |  fastvit_moe_sim.onnx (~42 MB)                                   |
  |       |                                                           |
  |       |  Step 4: Convert to TFLite                                |
  |       |  onnx -> tensorflow SavedModel -> tflite                  |
  |       v                                                           |
  |  +-------------------------------+                                |
  |  |  QUANTIZATION OPTIONS:        |                                |
  |  |                               |                                |
  |  |  FP32:  model_fp32.tflite     |  ~42 MB  (reference only)     |
  |  |  FP16:  model_fp16.tflite     |  ~21 MB  (good balance)       |
  |  |  INT8:  model_int8.tflite     |  ~11 MB  (best for mobile)    |
  |  |  INT8+: model_int8_qat.tflite |  ~ 8 MB  (QAT, best quality) |
  |  |                               |                                |
  |  |  RECOMMENDED: INT8 with QAT   |                                |
  |  |  (Quantization-Aware Training |                                |
  |  |   preserves MoE routing       |                                |
  |  |   precision better than PTQ)  |                                |
  |  +-------------------------------+                                |
  |                                                                   |
  |  ALSO EXPORT RT-DETRv2:                                           |
  |  rtdetr_r18.onnx -> rtdetr_int8.tflite (~8 MB)                  |
  |                                                                   |
  |  +-------------------------------+                                |
  |  |  FINAL MOBILE MODEL BUNDLE:   |                                |
  |  |                               |                                |
  |  |  fastvit_moe_int8.tflite  8MB |                                |
  |  |  rtdetr_r18_int8.tflite   8MB |                                |
  |  |  class_labels.json        2KB |                                |
  |  |  disease_treatments.json  80KB|                                |
  |  |  gemma_3n_e2b.task       ~1GB | (optional, capable phones)    |
  |  |  _____________________________|                                |
  |  |  CORE (without SLM):    ~16MB | <-- works on ANY phone        |
  |  |  FULL (with SLM):      ~1.1GB | <-- modern phones (4GB+ RAM)  |
  |  +-------------------------------+                                |
  +------------------------------------------------------------------+
```

---

## 1.3 — Treatment Database Generation (Using Open-Source LLM)

```
+===========================================================================+
|       TREATMENT DATABASE GENERATION — LLAMA 3.1 8B ON GOOGLE COLAB       |
+===========================================================================+
|                                                                           |
|  This step uses Llama 3.1 8B (open source, Meta) at BUILD TIME ONLY     |
|  to generate comprehensive, structured treatment data.                   |
|  The LLM does NOT run on the phone. Output is a static JSON/SQLite DB.  |
|                                                                           |
|  PIPELINE:                                                               |
|                                                                           |
|  +------------------------------------------------------------------+   |
|  |  FOR each disease in 38 classes:                                  |   |
|  |                                                                   |   |
|  |  PROMPT to Llama 3.1:                                             |   |
|  |  "You are an agricultural pathologist. For the plant disease      |   |
|  |   '{disease_name}' affecting {crop_name}, provide:               |   |
|  |   1. Scientific name of pathogen                                  |   |
|  |   2. Symptoms (detailed, 3-5 points)                             |   |
|  |   3. Severity level (Low/Medium/High/Critical)                    |   |
|  |   4. Chemical treatment (compound, dosage, frequency)             |   |
|  |   5. Organic/biological treatment alternatives                    |   |
|  |   6. Prevention methods (3-5 points)                              |   |
|  |   7. Favorable conditions (temp, humidity, season)                |   |
|  |   8. Economic impact if untreated                                 |   |
|  |   Format as structured JSON."                                     |   |
|  |                                                                   |   |
|  |  OUTPUT for each class --> JSON object                            |   |
|  +------------------------------------------------------------------+   |
|       |                                                                   |
|       v                                                                   |
|  +------------------------------------------------------------------+   |
|  |  HUMAN VERIFICATION + CROSS-REFERENCE                            |   |
|  |  - Cross-check against EPPO Global Database (gd.eppo.int)        |   |
|  |  - Verify dosages against agricultural extension manuals          |   |
|  |  - Flag any hallucinated recommendations                         |   |
|  +------------------------------------------------------------------+   |
|       |                                                                   |
|       v                                                                   |
|  +------------------------------------------------------------------+   |
|  |  FINAL OUTPUT: disease_treatments.json  (~80-100 KB)              |   |
|  |                                                                   |   |
|  |  {                                                                |   |
|  |    "Tomato___Late_blight": {                                      |   |
|  |      "pathogen": "Phytophthora infestans",                       |   |
|  |      "severity": "Critical",                                     |   |
|  |      "symptoms": ["Dark water-soaked lesions...", ...],          |   |
|  |      "chemical_treatment": {                                      |   |
|  |        "compound": "Mancozeb 75% WP",                            |   |
|  |        "dosage": "2.5 g/L water",                                |   |
|  |        "frequency": "Every 7-10 days",                           |   |
|  |        "safety_interval": "14 days before harvest"               |   |
|  |      },                                                           |   |
|  |      "organic_treatment": ["Copper-based fungicide...", ...],    |   |
|  |      "prevention": ["Use resistant varieties...", ...],          |   |
|  |      "conditions": {"temp": "15-25C", "humidity": ">80%"}       |   |
|  |    },                                                             |   |
|  |    ...                                                            |   |
|  |  }                                                                |   |
|  +------------------------------------------------------------------+   |
|                                                                           |
+===========================================================================+
```

---

## 1.4 — MCP Integration (Development Pipeline Orchestration)

```
+===========================================================================+
|          MCP (MODEL CONTEXT PROTOCOL) — DEV PIPELINE INTEGRATION         |
+===========================================================================+
|                                                                           |
|  MCP is used DURING DEVELOPMENT (not on mobile device) to orchestrate    |
|  the training pipeline tools as a unified agentic workflow.              |
|                                                                           |
|                                                                           |
|                    +---------------------------+                          |
|                    |    MCP CLIENT              |                          |
|                    |    (Your IDE / CLI Agent)  |                          |
|                    +-------------+-------------+                          |
|                                  |                                        |
|           +----------------------+----------------------+                 |
|           |                      |                      |                 |
|           v                      v                      v                 |
|  +------------------+  +------------------+  +------------------+        |
|  |  MCP SERVER 1    |  |  MCP SERVER 2    |  |  MCP SERVER 3    |        |
|  |  "Dataset Ops"   |  |  "Training Ops"  |  |  "Export Ops"    |        |
|  |                  |  |                  |  |                  |        |
|  |  Tools:          |  |  Tools:          |  |  Tools:          |        |
|  |  - download_data |  |  - train_teacher |  |  - export_onnx   |        |
|  |  - clean_images  |  |  - distill_model |  |  - convert_tflite|        |
|  |  - split_dataset |  |  - train_rtdetr  |  |  - quantize_int8 |        |
|  |  - augment_data  |  |  - eval_metrics  |  |  - benchmark_lat |        |
|  |  - class_balance |  |  - gradcam_vis   |  |  - validate_model|        |
|  +------------------+  +------------------+  +------------------+        |
|                                                                           |
|  EXAMPLE AGENTIC WORKFLOW:                                                |
|  +-------------------------------------------------------------------+   |
|  |  Agent (via MCP) can run:                                         |   |
|  |  1. "Download PlantVillage dataset" -> calls download_data tool   |   |
|  |  2. "Clean and split 70/15/15"      -> calls clean + split tools  |   |
|  |  3. "Train Swin-V2-T for 30 epochs" -> calls train_teacher tool   |   |
|  |  4. "Show confusion matrix"         -> calls eval_metrics tool    |   |
|  |  5. "Distill to FastViT-SA12"       -> calls distill_model tool   |   |
|  |  6. "Export to TFLite INT8"         -> calls export pipeline      |   |
|  |  7. "Benchmark on test set"         -> calls validate_model       |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
|  This replaces ad-hoc Jupyter cells with a reproducible, tool-based      |
|  pipeline that any developer can re-run identically.                     |
|                                                                           |
+===========================================================================+
```

---

## 1.5 — Evaluation Dashboard

```
+===========================================================================+
|                    EVALUATION METRICS & TARGETS                            |
+===========================================================================+
|                                                                           |
|  +------------------------------+----------+-----------+--------------+  |
|  | METRIC                       | TEACHER  | STUDENT   | STUDENT+MoE  |  |
|  |                              | Swin-V2  | FastViT   | FastViT+MoE  |  |
|  +------------------------------+----------+-----------+--------------+  |
|  | Overall Accuracy             | >= 97%   | >= 94%    | >= 96%       |  |
|  | Macro F1-Score               | >= 0.96  | >= 0.93   | >= 0.95      |  |
|  | Per-Class Recall (min)       | >= 90%   | >= 85%    | >= 88%       |  |
|  | Model Size (TFLite INT8)     | N/A      | ~8 MB     | ~11 MB       |  |
|  | Inference Latency (CPU)      | N/A      | ~30 ms    | ~35 ms       |  |
|  | Inference Latency (GPU/NPU)  | N/A      | ~8 ms     | ~12 ms       |  |
|  +------------------------------+----------+-----------+--------------+  |
|                                                                           |
|  EVALUATION TOOLS:                                                        |
|  +-------------------------------------------------------------------+   |
|  | 1. Confusion Matrix (38x38) heatmap                               |   |
|  | 2. Per-class Precision/Recall/F1 bar charts                       |   |
|  | 3. Grad-CAM attention overlays (Teacher vs Student comparison)     |   |
|  | 4. MoE routing analysis (which experts activate for which crops)  |   |
|  | 5. Cross-dataset generalization (PlantVillage -> PlantDoc OOD)    |   |
|  | 6. Latency benchmark on real Android device (ARM CPU + NNAPI)     |   |
|  | 7. Quantization accuracy drop analysis (FP32 vs INT8)             |   |
|  | 8. RT-DETRv2 mAP@50 for lesion detection                         |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
+===========================================================================+
```
