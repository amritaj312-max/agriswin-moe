# 🌿 Plant Disease Detection — Complete Project Breakdown

---

## 📋 System Constraints Analysis

```
┌─────────────────────────────────────────────────────────────────────┐
│                     YOUR LOCAL MACHINE SPECS                        │
├──────────────────────┬──────────────────────────────────────────────┤
│  CPU                 │  13th Gen Intel Core i5-13420H              │
│                      │  8 Cores / 12 Logical Processors            │
│                      │  Base: 2.10 GHz                             │
├──────────────────────┼──────────────────────────────────────────────┤
│  RAM                 │  15.6 GB DDR (54% used idle)                │
│                      │  ~7 GB available for training               │
├──────────────────────┼──────────────────────────────────────────────┤
│  GPU 0 (Dedicated)   │  NVIDIA GeForce RTX 3050 Laptop             │
│                      │  6 GB Dedicated VRAM                        │
│                      │  7.8 GB Shared GPU Memory                   │
│                      │  DirectX 12 (FL 12.2)                       │
├──────────────────────┼──────────────────────────────────────────────┤
│  GPU 1 (Integrated)  │  Intel UHD Graphics                         │
├──────────────────────┼──────────────────────────────────────────────┤
│  Storage             │  SSD (NVMe) — Fast I/O ✓                    │
├──────────────────────┼──────────────────────────────────────────────┤
│  OS                  │  Windows 11                                  │
└──────────────────────┴──────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    TRAINING STRATEGY DECISION                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  LOCAL RTX 3050 (6 GB VRAM)                                        │
│  ├── ✅ Good for: Prototyping, small batch training, inference      │
│  ├── ✅ Good for: TFLite/ONNX conversion & testing                 │
│  ├── ⚠️  Limited: Full training of large models (batch size < 16)  │
│  └── ❌ Cannot: Train heavy models like EfficientNet-B7            │
│                                                                     │
│  GOOGLE COLAB (Free/Pro GPU)                                        │
│  ├── ✅ T4 (15 GB) / A100 (40 GB) for full training                │
│  ├── ✅ Large batch sizes (32–64)                                   │
│  ├── ✅ Hyperparameter sweeps & experiments                         │
│  └── ✅ RECOMMENDED for main training pipeline                      │
│                                                                     │
│  ══════════════════════════════════════════════════════════════════  │
│  VERDICT: Train on Colab → Export model → Test locally →            │
│           Convert to TFLite → Deploy on mobile                      │
│  ══════════════════════════════════════════════════════════════════  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ High-Level Architecture Overview

```
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                        PLANT DISEASE DETECTION SYSTEM                            ║
║                         End-to-End Architecture                                  ║
╠═══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║   ┌───────────────────────────────┐     ┌───────────────────────────────────┐    ║
║   │                               │     │                                   │    ║
║   │     PART 1: ML/CV MODEL       │────▶│    PART 2: MOBILE APPLICATION     │    ║
║   │     (Google Colab + Local)     │     │    (React Native / Flutter)       │    ║
║   │                               │     │                                   │    ║
║   └───────────────────────────────┘     └───────────────────────────────────┘    ║
║          │                                         │                             ║
║          ▼                                         ▼                             ║
║   ┌─────────────┐  ┌──────────┐  ┌────────┐  ┌─────────┐  ┌─────────────────┐  ║
║   │  Dataset     │  │ Training │  │ Export │  │ Offline │  │ Treatment       │  ║
║   │  Collection  │→ │ Pipeline │→ │ TFLite │→ │ Detect  │→ │ Recommendation  │  ║
║   │  & Cleaning  │  │ on Colab │  │ / ONNX │  │ Engine  │  │ Database        │  ║
║   └─────────────┘  └──────────┘  └────────┘  └─────────┘  └─────────────────┘  ║
║                                                                                  ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
```

---
---

# ═══════════════════════════════════════════════════════
# PART 1: BUILD THE ML/CV MODEL
# ═══════════════════════════════════════════════════════

---

## 1.1 — Data Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE OVERVIEW                            │
└─────────────────────────────────────────────────────────────────────────────┘

  STEP 1: ACQUIRE           STEP 2: CLEAN            STEP 3: AUGMENT
  ══════════════            ═══════════════           ═══════════════
  ┌──────────────┐         ┌──────────────┐         ┌──────────────────┐
  │ Download     │         │ Remove       │         │ Random Flip      │
  │ PlantVillage │────────▶│ Duplicates   │────────▶│ Random Rotation  │
  │ + PlantDoc   │         │ Fix Labels   │         │ Color Jitter     │
  │ + iBean      │         │ Resize 224²  │         │ Random Crop      │
  │              │         │ Verify Files │         │ Gaussian Noise   │
  └──────────────┘         └──────────────┘         │ Mixup / CutMix   │
                                                     └──────────────────┘
                                                             │
                                                             ▼
                                          ┌──────────────────────────────────┐
                                          │       STEP 4: SPLIT              │
                                          ├──────────────────────────────────┤
                                          │                                  │
                                          │  ████████████████░░░░  Train 70% │
                                          │  ████████░░░░░░░░░░░░  Val   15% │
                                          │  ████░░░░░░░░░░░░░░░░  Test  15% │
                                          │                                  │
                                          │  Stratified split per class      │
                                          │  ~87,000 total images            │
                                          └──────────────────────────────────┘
```

### Dataset Directory Structure

```
datasets/
├── raw/                          # Original downloaded data
│   ├── PlantVillage/
│   │   ├── Apple___Apple_scab/
│   │   ├── Apple___Black_rot/
│   │   ├── Apple___Cedar_apple_rust/
│   │   ├── Apple___healthy/
│   │   ├── Corn___Cercospora_leaf_spot/
│   │   ├── Corn___Common_rust/
│   │   ├── Grape___Black_rot/
│   │   ├── Potato___Early_blight/
│   │   ├── Potato___Late_blight/
│   │   ├── Tomato___Bacterial_spot/
│   │   ├── Tomato___Early_blight/
│   │   ├── Tomato___Late_blight/
│   │   ├── Tomato___Leaf_Mold/
│   │   ├── Tomato___Septoria_leaf_spot/
│   │   ├── Tomato___healthy/
│   │   └── ... (38 classes total)
│   ├── PlantDoc/
│   │   └── ... (27 classes, real-world images)
│   └── iBean/
│       └── ... (3 classes: angular_leaf_spot, bean_rust, healthy)
│
├── processed/                    # Cleaned, resized, labeled
│   ├── train/
│   ├── val/
│   └── test/
│
└── metadata/
    ├── class_labels.json         # {0: "Apple___Apple_scab", ...}
    ├── disease_info.json         # Treatment & recommendation data
    └── split_stats.csv           # Class distribution per split
```

---

## 1.2 — Model Architecture Selection

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     MODEL ARCHITECTURE COMPARISON MATRIX                         │
├──────────────────┬───────────┬──────────┬───────────┬────────────┬──────────────┤
│ Model            │ Params    │ Size     │ Accuracy* │ Latency**  │ Mobile Ready │
│                  │ (Million) │ (MB)     │ (Top-1)   │ (ms/img)   │              │
├──────────────────┼───────────┼──────────┼───────────┼────────────┼──────────────┤
│ MobileNetV3-S    │   2.5 M   │   10 MB  │  ~91%     │   5–8 ms   │  ★★★★★      │
│ MobileNetV3-L    │   5.4 M   │   22 MB  │  ~93%     │  10–15 ms  │  ★★★★☆      │
│ EfficientNet-B0  │   5.3 M   │   20 MB  │  ~94%     │  12–18 ms  │  ★★★★☆      │
│ EfficientNet-B2  │   9.2 M   │   35 MB  │  ~96%     │  20–30 ms  │  ★★★☆☆      │
│ ResNet-50        │  25.6 M   │   98 MB  │  ~95%     │  40–60 ms  │  ★★☆☆☆      │
│ EfficientNet-B4  │  19.3 M   │   75 MB  │  ~97%     │  50–80 ms  │  ★★☆☆☆      │
├──────────────────┴───────────┴──────────┴───────────┴────────────┴──────────────┤
│ * Accuracy estimates on PlantVillage benchmark (varies with augmentation)       │
│ ** Inference latency on a mid-range Android phone (Snapdragon 7xx)              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ╔═══════════════════════════════════════════════════════════════════════════╗   │
│  ║  RECOMMENDED STRATEGY:                                                   ║   │
│  ║                                                                          ║   │
│  ║  PRIMARY:   EfficientNet-B0  (best accuracy-to-size ratio)               ║   │
│  ║  FALLBACK:  MobileNetV3-Large (if B0 is too slow on target devices)      ║   │
│  ║  ENSEMBLE:  Train both, pick best per-class performer                    ║   │
│  ╚═══════════════════════════════════════════════════════════════════════════╝   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Recommended Model: EfficientNet-B0 with Transfer Learning

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    EFFICIENTNET-B0  ARCHITECTURE (MODIFIED)                  │
│                                                                              │
│  INPUT IMAGE (224 x 224 x 3)                                                │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐         │
│  │              PRETRAINED BACKBONE (ImageNet Weights)             │         │
│  │                                                                 │         │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐      ┌──────────┐   │         │
│  │  │ MBConv1  │─▶│ MBConv6  │─▶│ MBConv6  │─▶...▶│ MBConv6  │   │         │
│  │  │ k3, s1   │  │ k3, s2   │  │ k5, s2   │      │ k3, s1   │   │         │
│  │  │ 32→16    │  │ 16→24    │  │ 24→40    │      │ 192→320  │   │         │
│  │  └──────────┘  └──────────┘  └──────────┘      └──────────┘   │         │
│  │                                                                 │         │
│  │  Each MBConv block:                                             │         │
│  │  ┌─────────┐ ┌──────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐  │         │
│  │  │Expand   │→│Depthwise │→│Squeeze& │→│Project  │→│DropPath│  │         │
│  │  │1x1 Conv │ │3x3/5x5  │ │Excite   │ │1x1 Conv │ │        │  │         │
│  │  └─────────┘ └──────────┘ └─────────┘ └─────────┘ └────────┘  │         │
│  │                                                                 │         │
│  └─────────────────────────────────────────────────────────────────┘         │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐         │
│  │                   CUSTOM CLASSIFICATION HEAD                    │         │
│  │                                                                 │         │
│  │  Global Average Pooling (1280)                                  │         │
│  │       │                                                         │         │
│  │       ▼                                                         │         │
│  │  Batch Normalization                                            │         │
│  │       │                                                         │         │
│  │       ▼                                                         │         │
│  │  Dropout (p=0.3)                                                │         │
│  │       │                                                         │         │
│  │       ▼                                                         │         │
│  │  Dense (512, ReLU)                                              │         │
│  │       │                                                         │         │
│  │       ▼                                                         │         │
│  │  Dropout (p=0.2)                                                │         │
│  │       │                                                         │         │
│  │       ▼                                                         │         │
│  │  Dense (38, Softmax)  ◄── 38 disease/healthy classes            │         │
│  │                                                                 │         │
│  └─────────────────────────────────────────────────────────────────┘         │
│       │                                                                      │
│       ▼                                                                      │
│  OUTPUT: class_id + confidence_score                                         │
│  Example: {"class": "Tomato___Late_blight", "confidence": 0.94}             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 1.3 — Training Pipeline (Google Colab)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE ON GOOGLE COLAB                        │
└──────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────┐                                                      
    │ Google Drive │  ◄── Upload dataset ZIP here                         
    │ /MyDrive/    │                                                      
    │  PlantData/  │                                                      
    └──────┬──────┘                                                      
           │  mount                                                       
           ▼                                                              
    ┌─────────────────────────────────────────────────────────────────┐   
    │                      COLAB NOTEBOOK                             │   
    │                                                                 │   
    │  CELL 1: Setup & Mount Drive                                    │   
    │  ┌───────────────────────────────────────────────────────────┐  │   
    │  │  !pip install torch torchvision timm albumentations       │  │   
    │  │  from google.colab import drive                           │  │   
    │  │  drive.mount('/content/drive')                            │  │   
    │  └───────────────────────────────────────────────────────────┘  │   
    │                                                                 │   
    │  CELL 2: Data Loading & Augmentation                            │   
    │  ┌───────────────────────────────────────────────────────────┐  │   
    │  │  Train transforms:                                        │  │   
    │  │    RandomResizedCrop(224)                                  │  │   
    │  │    RandomHorizontalFlip(p=0.5)                             │  │   
    │  │    RandomVerticalFlip(p=0.3)                               │  │   
    │  │    ColorJitter(brightness=0.2, contrast=0.2,               │  │   
    │  │               saturation=0.2, hue=0.1)                    │  │   
    │  │    RandomRotation(30)                                      │  │   
    │  │    GaussianBlur(kernel_size=3)                              │  │   
    │  │    Normalize(ImageNet mean/std)                             │  │   
    │  │                                                            │  │   
    │  │  Val/Test transforms:                                      │  │   
    │  │    Resize(256) → CenterCrop(224) → Normalize               │  │   
    │  └───────────────────────────────────────────────────────────┘  │   
    │                                                                 │   
    │  CELL 3: Model Definition                                       │   
    │  ┌───────────────────────────────────────────────────────────┐  │   
    │  │  model = timm.create_model('efficientnet_b0',             │  │   
    │  │              pretrained=True, num_classes=38)              │  │   
    │  │                                                            │  │   
    │  │  # Freeze backbone initially                               │  │   
    │  │  for param in model.parameters():                          │  │   
    │  │      param.requires_grad = False                           │  │   
    │  │  # Unfreeze classifier head                                │  │   
    │  │  for param in model.classifier.parameters():               │  │   
    │  │      param.requires_grad = True                            │  │   
    │  └───────────────────────────────────────────────────────────┘  │   
    │                                                                 │   
    │  CELL 4: Training Loop                                          │   
    │  ┌───────────────────────────────────────────────────────────┐  │   
    │  │  PHASE 1: Frozen backbone (5 epochs)                      │  │   
    │  │    Optimizer: Adam, lr=1e-3                                │  │   
    │  │    Loss: CrossEntropyLoss (label smoothing=0.1)            │  │   
    │  │    Batch size: 32                                          │  │   
    │  │                                                            │  │   
    │  │  PHASE 2: Unfreeze all layers (25 epochs)                  │  │   
    │  │    Optimizer: AdamW, lr=1e-4, weight_decay=1e-4            │  │   
    │  │    Scheduler: CosineAnnealingWarmRestarts(T_0=5)           │  │   
    │  │    Early stopping: patience=7                              │  │   
    │  └───────────────────────────────────────────────────────────┘  │   
    │                                                                 │   
    │  CELL 5: Evaluation & Export                                    │   
    │  ┌───────────────────────────────────────────────────────────┐  │   
    │  │  ● Classification Report (precision, recall, F1)          │  │   
    │  │  ● Confusion Matrix (38 × 38)                             │  │   
    │  │  ● Per-class accuracy analysis                             │  │   
    │  │  ● Grad-CAM visualizations                                 │  │   
    │  │  ● Export: .pth → ONNX → TFLite                           │  │   
    │  └───────────────────────────────────────────────────────────┘  │   
    │                                                                 │   
    └─────────────────────────────────────────────────────────────────┘   
```

### Training Hyperparameters Summary

```
┌────────────────────────────────────────────────────────────────────┐
│                    HYPERPARAMETER CONFIGURATION                    │
├───────────────────────┬────────────────────────────────────────────┤
│  Input Resolution     │  224 × 224 × 3                             │
│  Backbone             │  EfficientNet-B0 (ImageNet pretrained)     │
│  Number of Classes    │  38 (expandable)                           │
│  Batch Size           │  32 (Colab) / 8 (Local RTX 3050)          │
│  Phase 1 LR           │  1e-3 (head only, 5 epochs)               │
│  Phase 2 LR           │  1e-4 (full fine-tune, 25 epochs)         │
│  Optimizer            │  AdamW (weight_decay=1e-4)                 │
│  Scheduler            │  CosineAnnealingWarmRestarts               │
│  Loss Function        │  CrossEntropy + Label Smoothing (0.1)      │
│  Regularization       │  Dropout(0.3, 0.2) + DropPath(0.2)        │
│  Early Stopping       │  patience=7, monitor=val_loss              │
│  Augmentation         │  Heavy (flip, rotate, jitter, blur, crop)  │
│  Mixed Precision      │  FP16 (torch.cuda.amp) — saves VRAM       │
│  Gradient Clipping    │  max_norm=1.0                              │
└───────────────────────┴────────────────────────────────────────────┘
```

---

## 1.4 — Model Optimization & Conversion for Mobile

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                  MODEL OPTIMIZATION PIPELINE (for Mobile Deployment)             │
└──────────────────────────────────────────────────────────────────────────────────┘

  TRAINED MODEL (.pth, ~20 MB)
       │
       ├──────────────────────────────────────────────────────┐
       │                                                      │
       ▼                                                      ▼
  ┌──────────────────────┐                       ┌──────────────────────┐
  │  ONNX EXPORT         │                       │  TorchScript Export  │
  │  torch.onnx.export() │                       │  torch.jit.trace()   │
  │  opset_version=13    │                       │                      │
  └──────────┬───────────┘                       └──────────┬───────────┘
             │                                               │
             ▼                                               ▼
  ┌──────────────────────┐                       ┌──────────────────────┐
  │  ONNX Simplify       │                       │  Quantization        │
  │  onnx-simplifier     │                       │  (Dynamic/Static)    │
  │  Remove redundant    │                       │  INT8 quantized      │
  │  operations          │                       │  ~5 MB final size    │
  └──────────┬───────────┘                       └──────────────────────┘
             │
             ▼
  ┌──────────────────────┐
  │  TFLite CONVERSION   │
  │  onnx → tf → tflite  │
  │                      │
  │  Options:            │
  │  ├─ FP32 (~20 MB)   │
  │  ├─ FP16 (~10 MB)   │  ◄── Good balance
  │  └─ INT8 (~ 5 MB)   │  ◄── Best for offline/low-end
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                    FINAL MODEL ARTIFACTS                         │
  │                                                                  │
  │  📁 models/                                                      │
  │  ├── best_model.pth              ~20 MB  (PyTorch checkpoint)    │
  │  ├── model.onnx                  ~20 MB  (ONNX format)          │
  │  ├── model_fp16.tflite           ~10 MB  (TFLite FP16)          │
  │  ├── model_int8.tflite           ~ 5 MB  (TFLite INT8)          │
  │  ├── class_labels.json           ~ 2 KB  (38 class mappings)    │
  │  └── disease_treatments.json     ~50 KB  (treatment database)   │
  │                                                                  │
  │  TOTAL MOBILE BUNDLE:  ~5–10 MB  ✅  (fits offline APK easily)  │
  └──────────────────────────────────────────────────────────────────┘
```

---

## 1.5 — Evaluation & Validation

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION METRICS DASHBOARD                            │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  TARGET PERFORMANCE:                                                             │
│  ┌──────────────────────────────────────────────────────────────────────────┐    │
│  │  Overall Accuracy          ≥ 93%    ████████████████████████████░░  93%  │    │
│  │  Macro F1-Score            ≥ 0.91   ███████████████████████████░░░  91%  │    │
│  │  Per-class Recall          ≥ 85%    █████████████████████████░░░░░  85%  │    │
│  │  Inference (TFLite, CPU)   ≤ 100ms  ████████████░░░░░░░░░░░░░░░░░  ~80  │    │
│  │  Model Size (TFLite INT8)  ≤ 10 MB  ███████░░░░░░░░░░░░░░░░░░░░░░  ~5   │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  EVALUATION TOOLS:                                                               │
│  ┌────────────────────────────────────────────────────────────────────┐          │
│  │  ● Confusion Matrix (38×38) heatmap                                │          │
│  │  ● Per-class Precision / Recall / F1 bar charts                    │          │
│  │  ● ROC-AUC curves (One-vs-Rest per class)                         │          │
│  │  ● Grad-CAM / Attention overlays on sample images                 │          │
│  │  ● Failure case analysis (top misclassified pairs)                 │          │
│  │  ● Cross-dataset generalization test (train PlantVillage →         │          │
│  │    test on PlantDoc real-world images)                             │          │
│  └────────────────────────────────────────────────────────────────────┘          │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---
---

# ═══════════════════════════════════════════════════════
# PART 2: DEVELOP THE MOBILE APPLICATION
# ═══════════════════════════════════════════════════════

---

## 2.1 — Technology Stack Decision

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                      MOBILE TECH STACK COMPARISON                                │
├────────────────────┬─────────────────────────┬───────────────────────────────────┤
│                    │  React Native            │  Flutter                          │
├────────────────────┼─────────────────────────┼───────────────────────────────────┤
│  Language          │  JavaScript/TypeScript   │  Dart                             │
│  TFLite Support    │  react-native-tflite ✓  │  tflite_flutter ✓                │
│  Camera Access     │  react-native-camera ✓  │  camera plugin ✓                 │
│  Offline SQLite    │  react-native-sqlite ✓  │  sqflite ✓                       │
│  Cross-Platform    │  iOS + Android ✓        │  iOS + Android ✓                 │
│  Performance       │  Good (bridge overhead)  │  Excellent (native compilation)  │
│  Community         │  Very Large              │  Large & Growing                  │
│  Hot Reload        │  ✓                       │  ✓                               │
├────────────────────┴─────────────────────────┴───────────────────────────────────┤
│                                                                                  │
│  ╔════════════════════════════════════════════════════════════════════════════╗   │
│  ║  RECOMMENDED: React Native (with Expo)                                   ║   │
│  ║  REASON: Larger ecosystem, better community, JS familiarity,             ║   │
│  ║          excellent TFLite bindings, simpler offline storage.              ║   │
│  ║                                                                          ║   │
│  ║  ALTERNATIVE: Flutter — if you prefer Dart & want best native perf.      ║   │
│  ╚════════════════════════════════════════════════════════════════════════════╝   │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2.2 — App Architecture (Offline-First Design)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    MOBILE APP ARCHITECTURE (OFFLINE-FIRST)                        │
└──────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              PRESENTATION LAYER                                  │
│                                                                                  │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐   │
│  │  Home    │  │  Camera /    │  │ Results  │  │ History  │  │  Settings  │   │
│  │  Screen  │  │  Gallery     │  │ Screen   │  │ Screen   │  │  Screen    │   │
│  │          │  │  Screen      │  │          │  │          │  │            │   │
│  └────┬─────┘  └──────┬───────┘  └────┬─────┘  └────┬─────┘  └─────┬──────┘   │
│       │               │               │              │              │           │
│       └───────────────┴───────────────┴──────────────┴──────────────┘           │
│                                        │                                         │
├────────────────────────────────────────┼─────────────────────────────────────────┤
│                              BUSINESS LOGIC LAYER                                │
│                                        │                                         │
│  ┌─────────────────────────────────────┼────────────────────────────────────┐    │
│  │                                     │                                    │    │
│  │  ┌──────────────────┐  ┌───────────┴─────────┐  ┌───────────────────┐  │    │
│  │  │  Image           │  │  Disease             │  │  Treatment        │  │    │
│  │  │  Preprocessor    │  │  Classifier           │  │  Recommender      │  │    │
│  │  │                  │  │                       │  │                   │  │    │
│  │  │  • Resize 224²   │  │  • Load TFLite model  │  │  • Query local DB │  │    │
│  │  │  • Normalize     │  │  • Run inference      │  │  • Match disease  │  │    │
│  │  │  • Crop/Center   │  │  • Parse confidence   │  │  • Return Rx data │  │    │
│  │  └────────┬─────────┘  └────────┬──────────────┘  └────────┬──────────┘  │    │
│  │           │                     │                          │              │    │
│  │           └─────────────────────┴──────────────────────────┘              │    │
│  │                                 │                                         │    │
│  └─────────────────────────────────┼─────────────────────────────────────────┘    │
│                                    │                                              │
├────────────────────────────────────┼──────────────────────────────────────────────┤
│                              DATA / STORAGE LAYER                                │
│                                    │                                              │
│  ┌─────────────────────────────────┼──────────────────────────────────────┐       │
│  │                                 │                                      │       │
│  │  ┌──────────────┐  ┌───────────┴──────┐  ┌───────────────────────┐   │       │
│  │  │  TFLite      │  │  SQLite          │  │  AsyncStorage /       │   │       │
│  │  │  Model File  │  │  Database         │  │  SharedPreferences    │   │       │
│  │  │              │  │                   │  │                       │   │       │
│  │  │  model.tflite│  │  • scan_history   │  │  • user_settings      │   │       │
│  │  │  (~5-10 MB)  │  │  • disease_info   │  │  • language_pref      │   │       │
│  │  │  Bundled in  │  │  • treatments     │  │  • notification_pref  │   │       │
│  │  │  APK assets  │  │  • crop_calendar  │  │  • model_version      │   │       │
│  │  └──────────────┘  └──────────────────┘  └───────────────────────┘   │       │
│  │                                                                       │       │
│  └───────────────────────────────────────────────────────────────────────┘       │
│                                                                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│                           OPTIONAL ONLINE LAYER                                  │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │  When connectivity is available:                                            │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────────────┐  │ │
│  │  │ Sync scan    │  │  Check for   │  │  Community forum / expert       │  │ │
│  │  │ history to   │  │  model       │  │  consultation (future scope)    │  │ │
│  │  │ cloud backup │  │  updates     │  │                                 │  │ │
│  │  └──────────────┘  └──────────────┘  └─────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2.3 — App Screens & User Flow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              USER FLOW DIAGRAM                                   │
└──────────────────────────────────────────────────────────────────────────────────┘

                         ┌──────────────────┐
                         │   APP LAUNCH      │
                         │   Splash Screen   │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │   HOME SCREEN     │
                         │                   │
                         │  ┌─────────────┐  │
                         │  │ "Scan Crop" │  │
                         │  │   Button    │  │
                         │  └──────┬──────┘  │
                         │         │         │
                         │  ┌──────┴──────┐  │
                         │  │  Quick Tips  │  │
                         │  │  & Recent    │  │
                         │  │  Scans       │  │
                         │  └─────────────┘  │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
           ┌────────────────┐         ┌────────────────┐
           │  CAMERA MODE   │         │  GALLERY MODE  │
           │                │         │                │
           │  ┌──────────┐  │         │  Pick image    │
           │  │ Live     │  │         │  from phone    │
           │  │ Viewfinder│ │         │  gallery       │
           │  │          │  │         │                │
           │  │  [📸]    │  │         │  [Select]      │
           │  └──────────┘  │         └───────┬────────┘
           └───────┬────────┘                 │
                   │                          │
                   └──────────┬───────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │  IMAGE PREPROCESSING│
                   │                     │
                   │  Resize → Normalize │
                   │  Center Crop        │
                   │  ~50ms              │
                   └─────────┬───────────┘
                             │
                             ▼
                   ┌─────────────────────┐
                   │  TFLITE INFERENCE   │
                   │                     │
                   │  Model prediction   │
                   │  ~80–100ms (CPU)    │
                   │  ~20–30ms (GPU/NNAPI│
                   └─────────┬───────────┘
                             │
                             ▼
                   ┌─────────────────────────────────────────┐
                   │           RESULTS SCREEN                 │
                   │                                          │
                   │  ┌────────────────────────────────────┐  │
                   │  │  🌿 Detected: Tomato Late Blight   │  │
                   │  │  📊 Confidence: 94.2%              │  │
                   │  │  ─────────────────────────────────  │  │
                   │  │  📸 Original image with Grad-CAM   │  │
                   │  │     overlay highlighting affected   │  │
                   │  │     leaf regions                    │  │
                   │  │  ─────────────────────────────────  │  │
                   │  │  💊 TREATMENT:                      │  │
                   │  │  • Apply Mancozeb 75% WP @          │  │
                   │  │    2.5g/L as foliar spray            │  │
                   │  │  • Remove infected leaves            │  │
                   │  │  • Improve air circulation            │  │
                   │  │  • Avoid overhead irrigation          │  │
                   │  │  ─────────────────────────────────  │  │
                   │  │  🛡️ PREVENTION:                     │  │
                   │  │  • Use resistant varieties            │  │
                   │  │  • Crop rotation (3-year cycle)       │  │
                   │  │  • Fungicide schedule                 │  │
                   │  │  ─────────────────────────────────  │  │
                   │  │  [Save] [Share] [Scan Again]        │  │
                   │  └────────────────────────────────────┘  │
                   │                                          │
                   └──────────────────────────────────────────┘
```

---

## 2.4 — Offline Capability Design

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   OFFLINE-FIRST ARCHITECTURE DETAIL                              │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│    WHAT'S BUNDLED IN THE APK (works 100% offline):                              │
│    ═══════════════════════════════════════════════                               │
│                                                                                  │
│    📦 APK Bundle (~25-35 MB total)                                              │
│    │                                                                             │
│    ├── 📁 assets/                                                                │
│    │   ├── model_int8.tflite .............. 5 MB   (quantized model)            │
│    │   ├── class_labels.json .............. 2 KB   (38 class names)             │
│    │   └── disease_db.json ................ 50 KB  (treatment data)             │
│    │                                                                             │
│    ├── 📁 database/                                                              │
│    │   └── treatments.db ................. 200 KB  (SQLite, pre-populated)       │
│    │       ├── TABLE diseases (id, name, crop, symptoms, severity)               │
│    │       ├── TABLE treatments (disease_id, chemical, organic, dosage)          │
│    │       ├── TABLE prevention (disease_id, method, season, notes)              │
│    │       └── TABLE crop_calendar (crop, region, sow_date, harvest_date)       │
│    │                                                                             │
│    └── 📁 app_code/                                                              │
│        └── ... (React Native / Flutter compiled code)                            │
│                                                                                  │
│                                                                                  │
│    CONNECTIVITY-AWARE BEHAVIOR:                                                  │
│    ════════════════════════════                                                   │
│                                                                                  │
│    ┌─────────────┐     ┌──────────────────┐     ┌──────────────────────────┐    │
│    │  NO NETWORK  │     │  SLOW / 2G       │     │  FULL CONNECTIVITY       │    │
│    │              │     │                  │     │                          │    │
│    │  ✅ Scan     │     │  ✅ Scan         │     │  ✅ Scan                  │    │
│    │  ✅ Detect   │     │  ✅ Detect       │     │  ✅ Detect                │    │
│    │  ✅ Treatment│     │  ✅ Treatment    │     │  ✅ Treatment             │    │
│    │  ✅ History  │     │  ✅ History      │     │  ✅ History               │    │
│    │  ❌ Sync     │     │  ⏳ Queue Sync   │     │  ✅ Cloud Sync            │    │
│    │  ❌ Updates  │     │  ⏳ Queue Update  │     │  ✅ Model Update Check    │    │
│    │  ❌ Community│     │  ❌ Community    │     │  ✅ Community / Expert     │    │
│    └─────────────┘     └──────────────────┘     └──────────────────────────┘    │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2.5 — Project File Structure (Mobile App)

```
PlantDiseaseDetection/
│
├── 📱 mobile_app/                        # React Native Project
│   ├── android/                          # Android native configs
│   ├── ios/                              # iOS native configs
│   ├── src/
│   │   ├── screens/
│   │   │   ├── HomeScreen.tsx
│   │   │   ├── CameraScreen.tsx
│   │   │   ├── ResultScreen.tsx
│   │   │   ├── HistoryScreen.tsx
│   │   │   └── SettingsScreen.tsx
│   │   │
│   │   ├── components/
│   │   │   ├── CameraView.tsx
│   │   │   ├── ResultCard.tsx
│   │   │   ├── TreatmentInfo.tsx
│   │   │   ├── ConfidenceBar.tsx
│   │   │   ├── ScanHistoryItem.tsx
│   │   │   └── OfflineBadge.tsx
│   │   │
│   │   ├── services/
│   │   │   ├── ModelService.ts            # TFLite model loading & inference
│   │   │   ├── ImageService.ts            # Preprocessing, resize, normalize
│   │   │   ├── DatabaseService.ts         # SQLite CRUD operations
│   │   │   ├── TreatmentService.ts        # Disease → Treatment lookup
│   │   │   ├── NetworkService.ts          # Connectivity detection
│   │   │   └── SyncService.ts             # Cloud sync when online
│   │   │
│   │   ├── database/
│   │   │   ├── schema.sql
│   │   │   ├── seed_data.json
│   │   │   └── migrations/
│   │   │
│   │   ├── assets/
│   │   │   ├── models/
│   │   │   │   ├── model_int8.tflite
│   │   │   │   └── class_labels.json
│   │   │   ├── images/                    # App icons, illustrations
│   │   │   └── disease_data/
│   │   │       └── treatments.json
│   │   │
│   │   ├── navigation/
│   │   │   └── AppNavigator.tsx
│   │   │
│   │   ├── hooks/
│   │   │   ├── useModel.ts
│   │   │   ├── useCamera.ts
│   │   │   └── useNetwork.ts
│   │   │
│   │   └── utils/
│   │       ├── constants.ts
│   │       ├── imageUtils.ts
│   │       └── formatters.ts
│   │
│   ├── package.json
│   └── app.json
│
├── 🧠 ml_pipeline/                       # Model Training (Colab-compatible)
│   ├── notebooks/
│   │   ├── 01_data_exploration.ipynb
│   │   ├── 02_data_preprocessing.ipynb
│   │   ├── 03_model_training.ipynb
│   │   ├── 04_evaluation.ipynb
│   │   └── 05_export_tflite.ipynb
│   │
│   ├── src/
│   │   ├── data/
│   │   │   ├── dataset.py
│   │   │   ├── augmentations.py
│   │   │   └── preprocessing.py
│   │   ├── models/
│   │   │   ├── efficientnet.py
│   │   │   ├── mobilenet.py
│   │   │   └── model_factory.py
│   │   ├── training/
│   │   │   ├── trainer.py
│   │   │   ├── losses.py
│   │   │   └── schedulers.py
│   │   ├── evaluation/
│   │   │   ├── metrics.py
│   │   │   ├── gradcam.py
│   │   │   └── confusion_matrix.py
│   │   └── export/
│   │       ├── to_onnx.py
│   │       ├── to_tflite.py
│   │       └── quantize.py
│   │
│   └── requirements.txt
│
├── 📊 data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── 📋 docs/
│   ├── API.md
│   ├── MODEL_CARD.md
│   └── DEPLOYMENT.md
│
├── README.md
├── .gitignore
└── LICENSE
```

---

## 2.6 — Complete Inference Flow on Device

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    ON-DEVICE INFERENCE PIPELINE (Step-by-Step)                    │
└──────────────────────────────────────────────────────────────────────────────────┘

  USER CAPTURES PHOTO
       │
       │  Raw image (variable size, e.g. 4032×3024 from phone camera)
       ▼
  ┌──────────────────────────┐
  │  STEP 1: LOAD IMAGE      │    Time: ~10 ms
  │  Decode JPEG/PNG to RGB  │
  │  bitmap in memory        │
  └───────────┬──────────────┘
              │
              ▼
  ┌──────────────────────────┐
  │  STEP 2: RESIZE           │    Time: ~5 ms
  │  Bilinear interpolation   │
  │  → 256 × 256              │
  └───────────┬──────────────┘
              │
              ▼
  ┌──────────────────────────┐
  │  STEP 3: CENTER CROP     │    Time: ~2 ms
  │  256×256 → 224×224        │
  │  (center region)          │
  └───────────┬──────────────┘
              │
              ▼
  ┌──────────────────────────┐
  │  STEP 4: NORMALIZE        │    Time: ~3 ms
  │  pixel / 255.0            │
  │  (x - mean) / std         │
  │  ImageNet: mean=[0.485,   │
  │   0.456, 0.406]           │
  │  std=[0.229,0.224,0.225]  │
  └───────────┬──────────────┘
              │
              ▼
  ┌──────────────────────────┐
  │  STEP 5: TFLITE INFERENCE│    Time: ~50–100 ms (CPU)
  │                           │          ~15–30 ms  (GPU delegate)
  │  Input:  Float32 tensor   │          ~20–40 ms  (NNAPI delegate)
  │          [1, 224, 224, 3] │
  │                           │
  │  Output: Float32 tensor   │
  │          [1, 38]          │
  │          (38 class probs) │
  └───────────┬──────────────┘
              │
              ▼
  ┌──────────────────────────┐
  │  STEP 6: POST-PROCESS    │    Time: ~1 ms
  │                           │
  │  Softmax → Top-K classes  │
  │  Apply confidence thresh  │
  │  (threshold = 0.60)       │
  │                           │
  │  if max_conf < 0.60:      │
  │    → "Uncertain, retake"  │
  │  else:                    │
  │    → Return prediction    │
  └───────────┬──────────────┘
              │
              ▼
  ┌──────────────────────────┐
  │  STEP 7: DB LOOKUP        │    Time: ~5 ms
  │                           │
  │  Query SQLite:            │
  │  SELECT treatment,        │
  │    prevention, severity   │
  │  FROM diseases            │
  │  WHERE name = ?           │
  └───────────┬──────────────┘
              │
              ▼
  ┌──────────────────────────┐
  │  STEP 8: RENDER RESULTS   │    Time: ~20 ms
  │                           │
  │  Display:                 │
  │  • Disease name           │
  │  • Confidence bar         │
  │  • Affected region overlay│
  │  • Treatment card         │
  │  • Prevention tips        │
  │  • Save to history        │
  └──────────────────────────┘

  ═══════════════════════════════════════════════
  TOTAL END-TO-END LATENCY:  ~100–150 ms (CPU)
                              ~60–90 ms  (GPU)
  ═══════════════════════════════════════════════
  USER PERCEPTION:  INSTANT (< 200 ms target) ✅
```

---
---

# 📦 DATASETS — Official Sources (No Kaggle)

> [!IMPORTANT]
> All datasets listed below are from **official repositories** with peer-reviewed backing. Do NOT use Kaggle mirrors as they may have version mismatches or label errors.

### PRIMARY DATASET

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  DATASET 1: PlantVillage (Primary — Lab-Controlled Images)                       │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  📊 Stats:   54,305 images | 38 classes | 14 crop species                       │
│  📏 Resolution: ~256×256 px (variable)                                          │
│  🏷️  Classes:  26 diseases + 12 healthy categories                              │
│  📝 Paper:   "An open access repository of images on plant health"              │
│              Hughes & Salathé, 2015 (arXiv: 1511.08060)                         │
│                                                                                  │
│  🔗 OFFICIAL SOURCE:                                                            │
│     GitHub: https://github.com/spMohanty/PlantVillage-Dataset                   │
│     Paper:  https://arxiv.org/abs/1511.08060                                    │
│                                                                                  │
│  📋 Classes include:                                                             │
│     Apple (4), Blueberry (1), Cherry (2), Corn (4), Grape (4),                  │
│     Orange (1), Peach (2), Pepper (2), Potato (3), Raspberry (1),               │
│     Soybean (1), Squash (1), Strawberry (2), Tomato (10)                        │
│                                                                                  │
│  ⚠️  LIMITATION: Lab-controlled images on plain backgrounds.                    │
│     Real-world field images look different → need supplementary data.            │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### SUPPLEMENTARY DATASETS

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  DATASET 2: PlantDoc (Supplementary — Real-World Field Images)                   │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  📊 Stats:   2,598 images | 27 classes | 13 crop species                        │
│  📏 Images:  Real-world, field conditions, varied backgrounds                   │
│  📝 Paper:   "PlantDoc: A Dataset for Visual Plant Disease Detection"           │
│              Singh et al., 2020 (ACM CoDS-COMAD)                                │
│                                                                                  │
│  🔗 OFFICIAL SOURCE:                                                            │
│     GitHub:  https://github.com/pratikkayal/PlantDoc-Dataset                    │
│     Paper:   https://dl.acm.org/doi/10.1145/3371158.3371196                     │
│                                                                                  │
│  ✅ WHY USE: Bridges the gap between lab images (PlantVillage) and real          │
│     field conditions. Essential for generalization testing.                      │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│  DATASET 3: iBean (Supplementary — Bean Diseases)                                │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  📊 Stats:   1,296 images | 3 classes (angular_leaf_spot, bean_rust, healthy)   │
│  📏 Images:  Field-taken, real-world bean leaves                                │
│  📝 Paper:   Associated with Makerere AI Lab research                           │
│                                                                                  │
│  🔗 OFFICIAL SOURCE:                                                            │
│     TF Datasets: https://www.tensorflow.org/datasets/catalog/beans              │
│     GitHub:      https://github.com/AI-Lab-Makerere/ibean                       │
│                                                                                  │
│  ✅ WHY USE: High-quality field images for bean crops, adds diversity.           │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│  DATASET 4: New Plant Diseases Dataset (Extended PlantVillage)                   │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  📊 Stats:   87,000+ images | 38 classes (augmented version of PlantVillage)    │
│  📏 Details: Includes offline augmentations of original PlantVillage images     │
│                                                                                  │
│  🔗 OFFICIAL SOURCE:                                                            │
│     Mendeley Data: https://data.mendeley.com/datasets/tywbtsjrjv/1             │
│     (Published by: Arun Pandian J, Geetharamani Gopal)                          │
│                                                                                  │
│  ✅ WHY USE: Pre-augmented, ready for training, larger volume.                  │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│  DATASET 5: EPPO Global Database (Reference — Disease Metadata)                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  📝 Type:   NOT image data — this is a pest/disease information database        │
│  🔗 URL:    https://gd.eppo.int/                                                │
│                                                                                  │
│  ✅ WHY USE: To populate your treatment/prevention database with                │
│     scientifically accurate disease information, symptoms, host plants,         │
│     and management recommendations.                                             │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Recommended Dataset Combination Strategy

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    DATASET COMBINATION STRATEGY                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│                    ┌──────────────────────────────┐                              │
│                    │       TRAINING SET            │                              │
│                    │                               │                              │
│                    │  PlantVillage    : 54,305     │                              │
│                    │  Mendeley Augm.  : +32,700    │                              │
│                    │  PlantDoc        : +2,598     │                              │
│                    │  iBean           : +1,296     │                              │
│                    │  ──────────────────────────   │                              │
│                    │  TOTAL:          ~90,899 imgs │                              │
│                    │                               │                              │
│                    └───────────────┬──────────────┘                              │
│                                   │                                              │
│                    ┌──────────────┴──────────────┐                               │
│                    │                             │                                │
│                    ▼                             ▼                                │
│           ┌────────────────┐           ┌────────────────┐                        │
│           │  IN-DOMAIN     │           │  OUT-OF-DOMAIN │                        │
│           │  TEST SET      │           │  TEST SET      │                        │
│           │                │           │                │                        │
│           │  15% of merged │           │  Held-out      │                        │
│           │  dataset       │           │  PlantDoc real │                        │
│           │  (stratified)  │           │  field images  │                        │
│           │                │           │  (generalize?) │                        │
│           └────────────────┘           └────────────────┘                        │
│                                                                                  │
│  This strategy ensures:                                                          │
│  ✅ Large training volume for robust learning                                   │
│  ✅ Lab-quality images for high base accuracy                                   │
│  ✅ Real-world images for field deployment readiness                            │
│  ✅ Separate in-domain vs out-of-domain evaluation                              │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---
---

# 📅 COMPLETE PROJECT TIMELINE

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         PROJECT TIMELINE (8–10 WEEKS)                            │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  PART 1: ML MODEL                                                                │
│  ════════════════                                                                │
│                                                                                  │
│  Week 1   ██████████  Data Collection, Cleaning, EDA                            │
│  Week 2   ██████████  Preprocessing, Augmentation Pipeline, Splits              │
│  Week 3   ██████████  Model Architecture, Transfer Learning Setup (Colab)       │
│  Week 4   ██████████  Training, Hyperparameter Tuning, Experiments              │
│  Week 5   ██████████  Evaluation, Grad-CAM, Error Analysis, TFLite Export       │
│                                                                                  │
│  PART 2: MOBILE APP                                                              │
│  ═════════════════                                                               │
│                                                                                  │
│  Week 5   ████░░░░░░  Setup React Native, Project Scaffolding (overlap)         │
│  Week 6   ██████████  Camera Integration, TFLite Inference, Image Pipeline      │
│  Week 7   ██████████  Treatment DB, Results UI, Offline Storage                 │
│  Week 8   ██████████  History, Settings, UI Polish, Offline Testing             │
│  Week 9   ██████████  End-to-End Testing, Performance Optimization              │
│  Week 10  ██████████  APK Build, Documentation, Demo Ready                      │
│                                                                                  │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│  MILESTONES:                                                                     │
│  🏁 M1 (End Week 2): Dataset ready, splits created                              │
│  🏁 M2 (End Week 4): Model trained, ≥93% accuracy achieved                     │
│  🏁 M3 (End Week 5): TFLite model exported & verified                           │
│  🏁 M4 (End Week 7): Core app functional (scan → detect → treat)                │
│  🏁 M5 (End Week 9): Full offline capability tested                              │
│  🏁 M6 (End Week 10): APK ready for demo/submission                              │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

# 🔑 KEY TECHNICAL DECISIONS SUMMARY

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           DECISION MATRIX                                        │
├────────────────────────┬─────────────────────────────────────────────────────────┤
│  Decision Area          │  Choice & Rationale                                     │
├────────────────────────┼─────────────────────────────────────────────────────────┤
│  ML Framework           │  PyTorch + timm (best research ecosystem)              │
│  Training Hardware      │  Google Colab T4/A100 (6 GB local VRAM too limited)    │
│  Base Model             │  EfficientNet-B0 (best accuracy/size for mobile)       │
│  Transfer Learning      │  ImageNet pretrained → 2-phase fine-tune              │
│  Quantization           │  INT8 post-training (5 MB model, <100ms inference)     │
│  Mobile Format          │  TFLite (widest Android/iOS compatibility)             │
│  Mobile Framework       │  React Native (cross-platform, strong TFLite support)  │
│  Offline DB             │  SQLite (treatments, history, metadata)                │
│  Primary Dataset        │  PlantVillage (54K images, 38 classes)                │
│  Supplementary Data     │  PlantDoc + iBean (real-world generalization)          │
│  Offline Design         │  Model + DB bundled in APK, zero network required     │
│  Treatment Data Source  │  EPPO + agricultural extension manuals                 │
│  Input Resolution       │  224 × 224 (EfficientNet-B0 standard)                 │
│  Confidence Threshold   │  0.60 (below → "uncertain, retake photo")             │
│  Target Latency         │  < 200 ms end-to-end on mid-range phone               │
│  Target APK Size        │  < 35 MB (model + DB + app)                           │
└────────────────────────┴─────────────────────────────────────────────────────────┘
```
