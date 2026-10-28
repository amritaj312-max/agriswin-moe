"""
Find the largest batch size that fits in VRAM, per backbone, by actually
running a training step rather than estimating.

A 6 GB card is the binding constraint on this project, so the training configs
should come from measurement. Run this once and put the numbers in the configs.

Each trial runs in its own subprocess. That is not defensive style for its own
sake: an out-of-memory condition on this card surfaces as
CUBLAS_STATUS_EXECUTION_FAILED, which leaves the CUDA context unusable, so
every later trial in the same process would fail regardless of batch size.

Usage:
    python ml_pipeline/src/data/probe_batch.py
    python ml_pipeline/src/data/probe_batch.py --img-size 224
"""

import argparse
import subprocess
import sys

MODELS = [
    ("swinv2_tiny_window8_256", "teacher backbone"),
    ("efficientnet_b0", "baseline 1"),
    ("mobilevit_s", "baseline 2"),
    ("fastvit_sa12", "student backbone"),
]

CANDIDATES = [48, 32, 24, 16, 12, 8, 4]

# markers the child prints so the parent can tell a fit from an OOM
FIT = "PROBE_FIT"
OOM = "PROBE_OOM"


def run_trial(model_name, batch_size, img_size, amp):
    """Child process: one training step. Prints PROBE_FIT <peak_gb> or PROBE_OOM."""
    import gc
    import timm
    import torch

    try:
        model = timm.create_model(model_name, pretrained=False, num_classes=38).cuda()
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)

        x = torch.randn(batch_size, 3, img_size, img_size, device="cuda")
        y = torch.randint(0, 38, (batch_size,), device="cuda")

        # two steps: the optimizer allocates its state on the first one
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = torch.nn.functional.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        torch.cuda.synchronize()

        print(f"{FIT} {torch.cuda.max_memory_allocated() / 1024**3:.2f}")
        return 0

    except Exception as exc:
        text = str(exc).lower()
        oom_signatures = ("out of memory", "cublas_status_execution_failed",
                          "cublas_status_alloc_failed", "cudnn_status_alloc_failed")
        if isinstance(exc, torch.OutOfMemoryError) or any(s in text for s in oom_signatures):
            print(OOM)
            return 0
        print(f"PROBE_ERROR {type(exc).__name__}: {str(exc)[:200]}")
        return 1
    finally:
        gc.collect()


def probe(model_name, batch_size, img_size, amp):
    """Parent side: run one trial in a fresh process. Returns peak GB, or None on OOM."""
    cmd = [sys.executable, __file__, "--trial", model_name,
           "--batch", str(batch_size), "--img-size", str(img_size)]
    if not amp:
        cmd.append("--no-amp")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    out = result.stdout

    if FIT in out:
        return float(out.split(FIT)[1].split()[0])
    if OOM in out:
        return None
    if "PROBE_ERROR" in out:
        raise RuntimeError(out.split("PROBE_ERROR")[1].strip())
    raise RuntimeError(f"unexpected trial output:\n{out}\n{result.stderr[-400:]}")


def main():
    parser = argparse.ArgumentParser(description="Probe max batch size per backbone")
    parser.add_argument("--img-size", type=int, default=256,
                        help="input resolution (default 256: native for 3 of 4 backbones)")
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    parser.add_argument("--trial", metavar="MODEL", help=argparse.SUPPRESS)
    parser.add_argument("--batch", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.trial:
        sys.exit(run_trial(args.trial, args.batch, args.img_size, not args.no_amp))

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device available.")

    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"{torch.cuda.get_device_name(0)}  ({total:.1f} GB)")
    print(f"resolution {args.img_size}x{args.img_size}   AMP {'off' if args.no_amp else 'on'}\n")
    print(f"{'backbone':<30}{'max batch':>11}{'peak VRAM':>12}   role")
    print("-" * 72)

    results = {}
    for name, role in MODELS:
        found = None
        for bs in CANDIDATES:
            peak = probe(name, bs, args.img_size, not args.no_amp)
            if peak is not None:
                found = (bs, peak)
                break
        results[name] = found
        if found:
            print(f"{name:<30}{found[0]:>11}{found[1]:>11.2f} G   {role}")
        else:
            print(f"{name:<30}{'OOM at 4':>11}{'--':>12}   {role}")

    print("\nSuggested training config (half the max, for dataloader + OS headroom):")
    for name, found in results.items():
        if found:
            safe = max(4, found[0] // 2)
            accum = max(1, 32 // safe)
            print(f"  {name:<30} batch={safe:<4} grad_accum={accum:<3} "
                  f"-> effective batch {safe * accum}")


if __name__ == "__main__":
    main()
