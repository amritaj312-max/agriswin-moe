# Measured hardware limits

All numbers measured on this machine, not estimated. Re-run
`ml_pipeline/src/data/probe_batch.py` if the hardware changes.

**Machine:** RTX 3050 Laptop (6 GB, sm_86), driver 592.82, CUDA 12.6,
torch 2.14.1+cu126, Python 3.14.6.

## Set this before training

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This is not a micro-optimisation. Without it, Swin-V2-T OOMs above batch 16;
with it, batch 48 fits. The default allocator fragments badly on a 6 GB card,
and the OOM surfaces as `CUBLAS_STATUS_EXECUTION_FAILED`, which also poisons
the CUDA context — so a failure looks like a cuBLAS bug rather than memory
pressure.

## Max batch size, 256x256, AMP (fp16)

Measured by running real training steps (`probe_batch.py`), default allocator:

| backbone | max batch | peak VRAM | role |
|---|---|---|---|
| swinv2_tiny_window8_256 | 16 | 2.16 GB | teacher |
| efficientnet_b0 | 48 | 2.71 GB | baseline 1 |
| mobilevit_s | 32 | 3.18 GB | baseline 2 |
| fastvit_sa12 | 48 | 3.05 GB | student |

With `expandable_segments:True`, Swin-V2-T reaches batch 48 at 5.65 GB.

## Throughput — Swin-V2-T @ 256px, AMP, cudnn.benchmark

| batch | ms/step | ms/image | epoch | VRAM |
|---|---|---|---|---|
| 16 | 288 | 18.0 | 11 min | 2.16 GB |
| 24 | 417 | 17.4 | 11 min | 3.01 GB |
| 32 | 542 | 16.9 | 11 min | 3.89 GB |
| 48 | 793 | 16.5 | 10 min | 5.65 GB |

Throughput is flat across batch size, so the GPU is **compute-bound, not
memory-bound**. There is nothing to gain from pushing VRAM to its limit, and an
OOM 6 hours into a run costs far more than the 8% saved. **Use batch 32.**

Bottleneck breakdown at batch 16:

| stage | time |
|---|---|
| GPU forward+backward+step | 287 ms |
| dataloader, full augmentation, 4 workers | 69 ms |
| dataloader, eval transform, 4 workers | 34 ms |

The loader keeps up comfortably, so the heavy augmentation in
`dataset.train_transform` is effectively free. Do not weaken it to chase speed.

> Measure with warm-up. A naive timing loop from step 0 reports ~1300 ms/step
> because of cudnn autotuning and allocator growth — roughly 4.5x the true
> steady-state cost, which would make the project look infeasible when it is not.

## Planning numbers

- Teacher, 40 epochs: **~7.5 h**
- Each baseline (EfficientNet-B0, MobileViT-S), 30 epochs: ~4-6 h
- Full ablation matrix (~8 runs): ~2-3 days of continuous GPU time

This fits locally. Reserve Kaggle (30 GPU-h/week, P100/T4x2) for the runs that
need to go in parallel, not as the primary trainer.
