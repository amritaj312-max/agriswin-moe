"""
Verify the training environment before spending GPU hours on it.

Checks the Python/torch/CUDA stack, confirms the GPU is visible and can
actually run a matmul, reports usable VRAM, and sanity-checks that timm can
build the four backbones this project needs.

Usage:
    python ml_pipeline/src/data/check_env.py
"""

import importlib
import platform
import sys

# (backbone, why it is in the project)
BACKBONES = [
    ("swinv2_tiny_window8_256", "teacher backbone"),
    ("efficientnet_b0", "baseline 1"),
    ("mobilevit_s", "baseline 2"),
    ("fastvit_sa12", "student backbone"),
]


def ok(msg):
    print(f"  [ok]   {msg}")


def warn(msg):
    print(f"  [warn] {msg}")


def fail(msg):
    print(f"  [FAIL] {msg}")


def check_python():
    print("Python")
    print(f"  {sys.version.split()[0]} on {platform.system()} {platform.release()}")
    if sys.version_info < (3, 10):
        fail("Python 3.10+ required")
        return False
    ok("version supported")
    return True


def check_torch():
    print("\nPyTorch")
    try:
        import torch
    except ImportError:
        fail("torch not installed -- see README.md setup")
        return False, None

    print(f"  torch {torch.__version__}")
    if torch.version.cuda:
        ok(f"built against CUDA {torch.version.cuda}")
    else:
        fail("CPU-only build. Reinstall from the cu126 index (see README.md).")
        return False, torch

    if not torch.cuda.is_available():
        fail("CUDA build present but no GPU visible -- check the NVIDIA driver")
        return False, torch

    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    cap = torch.cuda.get_device_capability(0)
    ok(f"{name}  ({total:.1f} GB, sm_{cap[0]}{cap[1]})")

    # a real allocation + matmul, not just is_available()
    try:
        a = torch.randn(2048, 2048, device="cuda")
        (a @ a).sum().item()
        torch.cuda.synchronize()
        ok("GPU matmul succeeded")
    except Exception as exc:
        fail(f"GPU compute failed: {exc}")
        return False, torch

    try:
        with torch.autocast("cuda", dtype=torch.float16):
            (a @ a).sum().item()
        ok("AMP (float16) works -- needed to fit Swin-V2-T in 6 GB")
    except Exception as exc:
        warn(f"AMP failed: {exc}")

    del a
    torch.cuda.empty_cache()

    if total < 7:
        warn(f"{total:.1f} GB VRAM: use batch 16 + AMP + grad accumulation for Swin-V2-T")
    return True, torch


def check_packages():
    print("\nPackages")
    all_ok = True
    for mod, label in [("timm", "backbones"), ("numpy", "arrays"),
                       ("PIL", "image I/O"), ("sklearn", "metrics"),
                       ("pandas", "results tables"), ("matplotlib", "figures")]:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            ok(f"{mod:<12} {ver:<10} ({label})")
        except ImportError:
            fail(f"{mod:<12} missing       ({label})")
            all_ok = False
    return all_ok


def check_backbones():
    print("\nBackbones (timm)")
    try:
        import timm
        import torch
    except ImportError:
        warn("skipped -- timm or torch unavailable")
        return True

    all_ok = True
    for name, role in BACKBONES:
        try:
            # pretrained=False: this only checks the architecture exists and runs
            model = timm.create_model(name, pretrained=False, num_classes=38)
            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            size = 256 if "256" in name else 224
            with torch.no_grad():
                out = model(torch.randn(1, 3, size, size))
            assert out.shape == (1, 38), out.shape
            ok(f"{name:<28} {n_params:>5.1f} M   ({role})")
            del model
        except Exception as exc:
            fail(f"{name:<28} {type(exc).__name__}: {str(exc)[:60]}  ({role})")
            all_ok = False
    return all_ok


def check_classes():
    print("\nLabel space")
    try:
        import classes
    except ImportError:
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import classes

    if classes.NUM_CLASSES != 38:
        fail(f"expected 38 classes, got {classes.NUM_CLASSES}")
        return False
    counts = classes.expert_class_counts()
    if sum(counts.values()) != 38:
        fail("expert assignment does not cover all classes")
        return False
    ok(f"38 classes, {classes.NUM_EXPERTS} experts "
       f"({'/'.join(str(n) for n in counts.values())})")
    return True


def main():
    print("=" * 62)
    print("AgriSwin-MoE environment check")
    print("=" * 62 + "\n")

    results = [
        check_python(),
        check_torch()[0],
        check_packages(),
        check_backbones(),
        check_classes(),
    ]

    print("\n" + "=" * 62)
    if all(results):
        print("Environment is ready.")
        print("Next: python ml_pipeline/src/data/download.py --dataset all")
    else:
        print("Some checks failed -- see [FAIL] lines above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
