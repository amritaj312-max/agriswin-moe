# Plant Disease Detection — UPDATED Project Breakdown (Modern Tech Stack)

## System Constraints Recap

```
CPU:  Intel i5-13420H (8C/12T, 2.10 GHz)
RAM:  15.6 GB
GPU:  NVIDIA RTX 3050 Laptop (6 GB VRAM)
HEAVY TRAINING:  Google Colab (T4/A100)
CONSTRAINT:  100% Open Source, No Paid APIs, Offline Mobile
```

---

## UPDATED TECHNOLOGY STACK — 2025/2026 Cutting Edge

```
+===========================================================================+
|              MODERN TECHNOLOGY SELECTION MATRIX                            |
+=======================+===================================================+
| LAYER                 | TECHNOLOGY CHOSEN                                 |
+=======================+===================================================+
|                       |                                                   |
| VISION BACKBONE       | Swin Transformer V2-Tiny (TEACHER)                |
| (Training)            | 28M params, hierarchical shifted-window attention |
|                       | State-of-art accuracy on plant disease datasets   |
|                       | Open source: github.com/microsoft/Swin-Transformer |
|                       |                                                   |
+-----------------------+---------------------------------------------------+
|                       |                                                   |
| MOBILE VISION MODEL   | FastViT-SA12 (STUDENT — deployed on phone)        |
| (On-Device Inference) | 11M params, RepMixer + structural reparameterize  |
|                       | Hybrid CNN-Transformer, ~3ms on iPhone, ~8ms ARM  |
|                       | Open source: github.com/apple/ml-fastvit          |
|                       |                                                   |
+-----------------------+---------------------------------------------------+
|                       |                                                   |
| KNOWLEDGE DISTILL.    | Hybrid Distillation (Logit + Attention Map)       |
|                       | Teacher: Swin-V2-T --> Student: FastViT-SA12      |
|                       | L_total = a*L_ce + b*L_logit_KD + g*L_attn_KD    |
|                       |                                                   |
+-----------------------+---------------------------------------------------+
|                       |                                                   |
| DISEASE REGION        | RT-DETRv2 (Real-Time Detection Transformer v2)    |
| DETECTION             | NMS-free end-to-end, export-friendly discrete     |
|                       | sampling, R18 backbone for mobile                 |
|                       | Open source: github.com/lyuwenyu/RT-DETR          |
|                       |                                                   |
+-----------------------+---------------------------------------------------+
|                       |                                                   |
| CLASSIFICATION HEAD   | Sparse Mixture-of-Experts (MoE) Head              |
|                       | 6 expert sub-networks (1 per crop family)         |
|                       | Top-2 gating router, only 2 experts active/infer  |
|                       | Adds capacity WITHOUT adding inference cost       |
|                       |                                                   |
+-----------------------+---------------------------------------------------+
|                       |                                                   |
| ON-DEVICE SLM         | Gemma 3n E2B (Google DeepMind)                    |
| (Treatment Advice)    | ~2B effective params, runs on 2-4 GB RAM          |
|                       | Multimodal (text+image), fully offline via LiteRT |
|                       | Open source: huggingface.co/google/gemma-3n-E2B-it|
|                       |                                                   |
+-----------------------+---------------------------------------------------+
|                       |                                                   |
| TREATMENT DB GEN.     | Llama 3.1 8B (Meta) — on Google Colab             |
| (Build-Time Only)     | Generates comprehensive treatment database        |
|                       | Output: JSON/SQLite bundled in APK                |
|                       | Open source: huggingface.co/meta-llama            |
|                       |                                                   |
+-----------------------+---------------------------------------------------+
|                       |                                                   |
| DEV PIPELINE ORCH.    | MCP (Model Context Protocol) — Anthropic          |
| (Development Time)    | Tool orchestration for training pipeline          |
|                       | Dataset management, model versioning              |
|                       | Open source: modelcontextprotocol.io              |
|                       |                                                   |
+-----------------------+---------------------------------------------------+
|                       |                                                   |
| MULTI-AGENT BACKEND   | A2A Protocol (Agent-to-Agent) — Google/Linux Fdn  |
| (Online Mode)         | Agent Cards for discovery, JSON-RPC transport     |
|                       | Disease Agent <-> Treatment Agent <-> Weather     |
|                       | Open source: a2a-protocol.org                     |
|                       |                                                   |
+-----------------------+---------------------------------------------------+
|                       |                                                   |
| MOBILE FRAMEWORK      | React Native + Expo                               |
| ON-DEVICE INFERENCE   | TFLite via LiteRT (Google AI Edge SDK)            |
| SLM RUNTIME           | MediaPipe LLM Inference API / llama.cpp           |
| OFFLINE DATABASE      | SQLite (via react-native-sqlite-storage)          |
|                       |                                                   |
+-----------------------+---------------------------------------------------+
|                       |                                                   |
| MODEL EXPORT          | PyTorch -> ONNX -> TFLite (INT8 quantized)       |
| QUANTIZATION          | Post-Training Quantization (PTQ) + QAT            |
|                       |                                                   |
+=======================+===================================================+
```

---

## HIGH-LEVEL ARCHITECTURE (UPDATED)

```
+===============================================================================+
|                PLANT DISEASE DETECTION — MODERN ARCHITECTURE                   |
+===============================================================================+
|                                                                               |
|   +-----------------------------------------------------------------------+   |
|   |                    DEVELOPMENT / TRAINING PHASE                        |   |
|   |                    (Google Colab + Local Machine)                      |   |
|   |                                                                       |   |
|   |  +------------------+    +-------------------+    +----------------+  |   |
|   |  | Swin-V2-T Teacher|    | RT-DETRv2 R18     |    | Llama 3.1 8B  |  |   |
|   |  | (Classification) |--->| (Region Detection)|--->| (Generate     |  |   |
|   |  | Train on Colab   |    | Train on Colab    |    |  Treatment DB)|  |   |
|   |  +--------+---------+    +-------------------+    +-------+--------+  |   |
|   |           |                                               |           |   |
|   |           | Knowledge Distillation                        |           |   |
|   |           v                                               v           |   |
|   |  +------------------+                            +----------------+   |   |
|   |  | FastViT-SA12     |                            | treatments.db  |   |   |
|   |  | Student Model    |                            | (SQLite/JSON)  |   |   |
|   |  | + MoE Head       |                            +-------+--------+   |   |
|   |  +--------+---------+                                    |            |   |
|   |           |                                              |            |   |
|   |           | Export: ONNX -> TFLite INT8                  |            |   |
|   |           v                                              v            |   |
|   |  +------------------+                            +----------------+   |   |
|   |  | model_int8.tflite|                            | disease_db.json|   |   |
|   |  | (~6-8 MB)        |                            | (~100 KB)      |   |   |
|   |  +--------+---------+                            +-------+--------+   |   |
|   |           |                                              |            |   |
|   +-----------|----------------------------------------------+------------+   |
|               |                                              |                |
|               +--------------------+-------------------------+                |
|                                    |                                          |
|                                    v                                          |
|   +-----------------------------------------------------------------------+   |
|   |                    MOBILE APPLICATION (React Native)                   |   |
|   |                                                                       |   |
|   |  +======================== OFFLINE CORE ========================+    |   |
|   |  |                                                               |    |   |
|   |  |  +-----------+   +-------------+   +-----------+             |    |   |
|   |  |  | Camera /  |-->| FastViT-SA12|-->| MoE Class.|             |    |   |
|   |  |  | Gallery   |   | + RT-DETRv2 |   | Head      |             |    |   |
|   |  |  | Input     |   | (TFLite)    |   | (38 cls)  |             |    |   |
|   |  |  +-----------+   +-------------+   +-----+-----+             |    |   |
|   |  |                                         |                     |    |   |
|   |  |                                         v                     |    |   |
|   |  |                                   +----------+                |    |   |
|   |  |                                   | SQLite   |                |    |   |
|   |  |                                   | Lookup   |                |    |   |
|   |  |                                   +-----+----+                |    |   |
|   |  |                                         |                     |    |   |
|   |  |                                         v                     |    |   |
|   |  |                            +---------------------------+      |    |   |
|   |  |                            | Gemma 3n E2B (Optional)   |      |    |   |
|   |  |                            | Contextual Treatment      |      |    |   |
|   |  |                            | Advice in Local Language  |      |    |   |
|   |  |                            | (via LiteRT / llama.cpp)  |      |    |   |
|   |  |                            +---------------------------+      |    |   |
|   |  |                                                               |    |   |
|   |  +=========================== 100% OFFLINE ======================+    |   |
|   |                                                                       |   |
|   |  +======================== ONLINE MODE =========================+    |   |
|   |  |  (Activates when connectivity is available)                   |    |   |
|   |  |                                                               |    |   |
|   |  |  +------------------+    A2A     +------------------+         |    |   |
|   |  |  | On-Device Agent  |<---------->| Cloud Agents     |         |    |   |
|   |  |  | (MCP Client)     |  Protocol  | - Weather Agent  |         |    |   |
|   |  |  |                  |            | - Expert Agent   |         |    |   |
|   |  |  +------------------+            | - Update Agent   |         |    |   |
|   |  |                                  +------------------+         |    |   |
|   |  |                                                               |    |   |
|   |  +===============================================================+    |   |
|   +-----------------------------------------------------------------------+   |
+===============================================================================+
```
