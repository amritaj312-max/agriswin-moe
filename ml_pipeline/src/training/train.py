"""
Train one model on the frozen PlantVillage splits.

Every run in the paper goes through this one loop -- the three baselines, each
ablation, the teacher -- so a bug fixed here is fixed everywhere, and no two rows
of a table can have been trained differently. Shared hyperparameters live in
configs/recipe.yaml; a run config names the model and little else.

Per run it writes:
    checkpoints/<run>/last.pt    full state, rewritten every epoch (for --resume)
    checkpoints/<run>/best.pt    weights at the best val macro-F1
    checkpoints/<run>/log.csv    one row per epoch
    checkpoints/<run>/train.log  everything printed
    results/<run>.json           config, environment, history, val and test metrics

The test split is evaluated exactly once, at the end, on the best-val weights. It
is never used to choose anything.

Usage:
    python ml_pipeline/src/training/train.py --config ml_pipeline/configs/baselines/efficientnet_b0.yaml
    python ml_pipeline/src/training/train.py --config <same> --resume
    python ml_pipeline/src/training/train.py --config <same> --stop-after 20     # split a long run
    python ml_pipeline/src/training/train.py --config <same> --set epochs=2 run_name=dev_effnet

Name anything that is not a paper result dev_*: those results are git-ignored.
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time

# Must be set before torch initialises CUDA. Without it Swin-V2-T OOMs above
# batch 16 on the 6 GB card, and the OOM surfaces as CUBLAS_STATUS_EXECUTION_FAILED
# rather than as an out-of-memory error. See configs/hardware.md.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# The Hugging Face cache falls back to copying files when Windows refuses
# symlinks. That is harmless; its warning is noise in every run's log.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import timm  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("data", "evaluation", "models"):
    sys.path.insert(0, os.path.join(SRC, sub))

import paths  # noqa: E402
from classes import NUM_CLASSES, PLANTVILLAGE_CLASSES  # noqa: E402
from classifier import Classifier  # noqa: E402
from dataset import build_loaders  # noqa: E402
from metrics import summarize  # noqa: E402

RECIPE = os.path.join(paths.REPO_ROOT, "ml_pipeline", "configs", "recipe.yaml")
CHECKPOINT_DIR = os.path.join(paths.REPO_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(paths.REPO_ROOT, "results")

RUN_KEYS = {"run_name", "model", "subset"}   # per-run keys the recipe does not hold
RESUMABLE_KEYS = {"num_workers"}             # may differ between sessions of one run
# what --set may change on a paper run; anything else departs from the recipe
# every row of a table has to share
PAPER_OVERRIDES = {"run_name", "seed", "num_workers"}
AMP_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}
LOG_EVERY = 100                              # steps between progress lines
CSV_FIELDS = ["epoch", "lr", "train_loss", "train_acc", "val_nll", "val_acc",
              "val_macro_f1", "val_ece", "time_s", "nonfinite_steps"]


# --- config ------------------------------------------------------------------

def load_config(path, overrides):
    """recipe.yaml, then the run config on top, then --set KEY=VALUE pairs."""
    with open(RECIPE, encoding="utf-8") as fh:
        recipe = yaml.safe_load(fh)
    cfg = dict(recipe, subset=None)

    layers = []
    if path:
        with open(path, encoding="utf-8") as fh:
            layers.append((path, yaml.safe_load(fh) or {}))
    sets = {}
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        sets[key] = yaml.safe_load(value)
    layers.append(("--set", sets))

    for source, layer in layers:
        unknown = set(layer) - set(cfg) - RUN_KEYS
        if unknown:
            raise SystemExit(f"{source}: unknown key(s) {sorted(unknown)}")
        cfg.update(layer)

    # YAML reads "1e-4" as a string; coerce anything the recipe types as a float
    for key, default in recipe.items():
        if isinstance(default, float) and isinstance(cfg[key], (int, str)):
            cfg[key] = float(cfg[key])

    if not cfg.get("model"):
        raise SystemExit("no model: pass --config, or --set model=<timm name>")
    if cfg["amp_dtype"] not in AMP_DTYPES:
        raise SystemExit(f"amp_dtype must be one of {sorted(AMP_DTYPES)}")
    cfg.setdefault("run_name", cfg["model"])
    return cfg


# --- optimisation ------------------------------------------------------------

def lr_at(step, total, warmup, peak, floor):
    """Linear warmup to `peak`, then cosine decay to `floor`."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


def param_groups(model, weight_decay):
    """AdamW groups. No decay on biases, on parameters that are vectors in
    disguise (norm gains, layer scales, Swin-V2's (heads, 1, 1) logit_scale), or
    on anything a submodule exempts through no_weight_decay().

    timm's optimizer factory is not used because it matches no_weight_decay()
    names exactly, while Swin-V2 returns module names (".../attn.cpb_mlp"), so its
    position-bias MLP would be decayed anyway; and logit_scale is a parameter,
    not a module, so Swin-V2 never lists it at all.
    """
    exempt = []
    for prefix, module in model.named_modules():
        if hasattr(module, "no_weight_decay"):
            exempt += [f"{prefix}.{name}" if prefix else name for name in module.no_weight_decay()]

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (p.squeeze().ndim <= 1 or name.endswith(".bias")
                or any(name == e or name.startswith(e + ".") for e in exempt)):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def train_one_epoch(model, loader, optimizer, scaler, cfg, step, lr_fn, epoch, log):
    model.train()
    amp_dtype = AMP_DTYPES[cfg["amp_dtype"]]
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    # accumulated on the GPU: reading them back every step would force a sync
    loss_sum = torch.zeros((), device="cuda")
    correct = torch.zeros((), device="cuda", dtype=torch.long)
    nonfinite = torch.zeros((), device="cuda", dtype=torch.long)
    n_steps, seen, start = len(loader), 0, time.time()
    mark_time, mark_seen = start, 0

    for i, (images, labels) in enumerate(loader, 1):
        if i == 1:
            # On Windows each worker is a fresh process that re-imports torch, so
            # the first batch of a session can take a minute. Report it apart from
            # the throughput, which would otherwise look several times too slow.
            mark_time = time.time()
            if mark_time - start > 10:
                log(f"  data workers took {mark_time - start:.0f}s to start")
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        lr = lr_fn(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        with torch.autocast("cuda", dtype=amp_dtype):
            logits = model(images)
            loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg["clip_grad"]:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])
        scaler.step(optimizer)          # the scaler skips the step if gradients overflowed
        scaler.update()

        loss = loss.detach()
        loss_sum += torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        nonfinite += (~torch.isfinite(loss)).long()
        correct += logits.detach().argmax(1).eq(labels).sum()
        seen += labels.numel()
        step += 1

        if i % LOG_EVERY == 0 or i == n_steps:
            now = time.time()
            rate = (seen - mark_seen) / max(now - mark_time, 1e-6)   # since the last line
            mark_time, mark_seen = now, seen
            finite_steps = max(1, i - int(nonfinite))
            log(f"  epoch {epoch} [{i:>4}/{n_steps}]  loss {loss_sum.item() / finite_steps:.4f}  "
                f"lr {lr:.2e}  {rate:.0f} img/s  eta {hms((n_steps - i) * labels.numel() / rate)}")

    bad = int(nonfinite)
    return {"train_loss": loss_sum.item() / max(1, n_steps - bad),
            "train_acc": correct.item() / seen,
            "nonfinite_steps": bad}, step


@torch.no_grad()
def predict(model, loader, amp_dtype):
    """Logits and labels for a whole split, in eval mode (dropout off)."""
    model.eval()
    logits, labels = [], []
    for images, y in loader:
        with torch.autocast("cuda", dtype=amp_dtype):
            logits.append(model(images.cuda(non_blocking=True)).float().cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


# --- bookkeeping -------------------------------------------------------------

def save(obj, path):
    """Write then rename, so a crash mid-save cannot corrupt the previous file."""
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def fingerprint(dataset):
    """Which images, with which labels, a split held. Equal fingerprints across
    runs prove every row of a table was scored on the same images. Sorted, so
    re-ordering a CSV cannot change it."""
    rows = sorted(f"{r['path']}\t{r['class_idx']}" for r in dataset.rows)
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()[:16]


def git(*args):
    """Output of a git command run in the repo, or None if git is unavailable."""
    try:
        out = subprocess.run(["git", *args], cwd=paths.REPO_ROOT, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def paper_run_problems(cfg, overridden):
    """Why this run could not stand as a paper result. Runs named dev_* are
    exempt: experimenting is what they are for.

    Checked before any GPU time is spent, because each problem would otherwise
    surface hours later as a number that cannot share a table with the others.
    """
    if cfg["run_name"].startswith("dev_"):
        return []
    problems = []
    changed = git("status", "--porcelain", "--", "ml_pipeline", "data/splits")
    if changed:
        files = "\n".join("      " + line for line in changed.splitlines())
        problems.append(f"uncommitted changes to the code, configs or splits, so the "
                        f"result could not be traced to a commit:\n{files}")
    departures = sorted(set(overridden) - PAPER_OVERRIDES)
    if departures:
        problems.append(f"--set {', '.join(departures)} departs from the recipe the other "
                        f"runs in its table use")
    if cfg["subset"]:
        problems.append("a subset run sees different images from every other run")
    return problems


def keep_awake():
    """Ask Windows not to sleep while this process runs, as a video player does
    during playback. No setting changes, and the request ends with the process.
    It cannot stop sleep when the lid closes, or an update restart.

    Returns None off Windows, else whether Windows accepted the request.
    """
    if sys.platform != "win32":
        return None
    import ctypes
    es_continuous, es_system_required = 0x80000000, 0x00000001
    set_state = ctypes.windll.kernel32.SetThreadExecutionState
    set_state.argtypes, set_state.restype = [ctypes.c_uint], ctypes.c_uint
    return set_state(es_continuous | es_system_required) != 0


def environment():
    dirty = git("status", "--porcelain", "--untracked-files=no")
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "timm": timm.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": None if dirty is None else bool(dirty),
    }


def scalars(metrics):
    return {k: v for k, v in metrics.items() if not isinstance(v, list)}


def hms(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def make_logger(path):
    """print() that also appends to the run's train.log."""
    fh = open(path, "a", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        fh.write(msg + "\n")
        fh.flush()
    return log


# --- main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train one model on the frozen PlantVillage splits")
    parser.add_argument("--config", help="run config, layered over configs/recipe.yaml")
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. --set epochs=2 run_name=dev_x")
    parser.add_argument("--resume", action="store_true", help="continue checkpoints/<run>/last.pt")
    parser.add_argument("--stop-after", type=int, metavar="EPOCH",
                        help="exit cleanly after this epoch; finish later with --resume")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    run = cfg["run_name"]
    run_dir = os.path.join(CHECKPOINT_DIR, run)
    last_path, best_path = os.path.join(run_dir, "last.pt"), os.path.join(run_dir, "best.pt")
    finished_marker = os.path.join(run_dir, "FINISHED")
    results_path = os.path.join(RESULTS_DIR, f"{run}.json")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; run ml_pipeline/src/data/check_env.py")
    if os.path.isfile(finished_marker):
        raise SystemExit(f"{run} already finished ({results_path}); delete {run_dir} to retrain")

    state = None
    if os.path.isfile(last_path):
        if not args.resume:
            raise SystemExit(f"{run_dir} already holds a run: pass --resume to continue it, "
                             f"or delete the directory to start over")
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        changed = sorted(k for k in set(cfg) | set(state["config"])
                         if k not in RESUMABLE_KEYS and cfg.get(k) != state["config"].get(k))
        if changed:
            raise SystemExit(f"config differs from the checkpoint in {changed}; resuming would "
                             f"mix two recipes in one result")
    elif args.resume:
        raise SystemExit(f"nothing to resume: {last_path} does not exist")

    problems = paper_run_problems(cfg, [item.partition("=")[0] for item in args.set])
    if problems:
        raise SystemExit(f"'{run}' is not named dev_..., so it has to be able to stand as a "
                         f"paper result, but:\n  - " + "\n  - ".join(problems)
                         + "\nCommit the changes first, or give the run a dev_ name to experiment.")

    os.makedirs(run_dir, exist_ok=True)
    log = make_logger(os.path.join(run_dir, "train.log"))
    awake = keep_awake()
    track = state["track"] if state else {
        "epoch": 0, "step": 0, "best_f1": -1.0, "best_epoch": 0, "bad_epochs": 0,
        "train_time_s": 0.0, "peak_vram_gb": 0.0, "training_done": False, "history": []}

    random.seed(cfg["seed"] + track["epoch"])
    torch.manual_seed(cfg["seed"] + track["epoch"])
    torch.backends.cudnn.benchmark = True
    amp_dtype = AMP_DTYPES[cfg["amp_dtype"]]

    # pretrained weights are pointless when a checkpoint is about to replace them
    model = Classifier(cfg["model"], NUM_CLASSES, pretrained=cfg["pretrained"] and state is None,
                       drop_rate=cfg["drop_rate"], drop_path_rate=cfg["drop_path_rate"])
    data_cfg = model.data_config()
    mean, std = tuple(data_cfg["mean"]), tuple(data_cfg["std"])
    model.cuda()
    n_params = sum(p.numel() for p in model.parameters())
    # the starting weights, by name: two machines with different timm versions
    # could otherwise start the "same" model from different weights unnoticed
    weights = ((model.backbone.pretrained_cfg.get("hf_hub_id") or "ImageNet-pretrained")
               if cfg["pretrained"] else None)

    train_loader, val_loader, test_loader = build_loaders(
        batch_size=cfg["batch_size"], img_size=cfg["img_size"], num_workers=cfg["num_workers"],
        mean=mean, std=std, subset=cfg["subset"])
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    sizes = {name: len(ld.dataset) for name, ld in loaders.items()}
    fingerprints = {name: fingerprint(ld.dataset) for name, ld in loaders.items()}

    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise SystemExit("the training split is smaller than one batch")
    total_steps = cfg["epochs"] * steps_per_epoch
    warmup_steps = min(round(cfg["warmup_epochs"] * steps_per_epoch), total_steps)

    def lr_fn(s):
        return lr_at(s, total_steps, warmup_steps, cfg["lr"], cfg["min_lr"])

    optimizer = torch.optim.AdamW(param_groups(model, cfg["weight_decay"]), lr=cfg["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])

    def fmt(values):
        return "(" + ", ".join(f"{v:.3f}" for v in values) + ")"

    subset_note = f"  SUBSET {cfg['subset']} -- not a paper result" if cfg["subset"] else ""
    session = f"resuming after epoch {track['epoch']}" if state else "fresh start"
    log("=" * 78)
    log(f"run     {run}  ({session})")
    log(f"model   {cfg['model']}  {n_params / 1e6:.2f} M params  "
        f"{'weights ' + weights if weights else 'from scratch'}")
    log(f"data    train {sizes['train']:,} / val {sizes['val']:,} / test {sizes['test']:,}{subset_note}")
    log(f"input   {cfg['img_size']}px  mean {fmt(mean)}  std {fmt(std)}")
    log(f"recipe  AdamW lr {cfg['lr']:g} wd {cfg['weight_decay']:g}  warmup {cfg['warmup_epochs']:g} ep, "
        f"cosine to {cfg['min_lr']:g}  {cfg['epochs']} ep x {steps_per_epoch} steps  "
        f"batch {cfg['batch_size']}  {cfg['amp_dtype']}")
    if awake is not None:
        log("power   " + ("Windows will not sleep while this runs (closing the lid still will)"
                          if awake else "WARNING: could not stop Windows sleeping; disable sleep yourself"))
    log("=" * 78)

    csv_path = os.path.join(run_dir, "log.csv")
    if not state:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(CSV_FIELDS)

    torch.cuda.reset_peak_memory_stats()
    patience = cfg["early_stop_patience"]
    epoch = track["epoch"]
    while not track["training_done"]:
        epoch += 1
        start = time.time()
        stats, track["step"] = train_one_epoch(model, train_loader, optimizer, scaler, cfg,
                                               track["step"], lr_fn, epoch, log)
        if stats["nonfinite_steps"] > 0.05 * steps_per_epoch:
            raise SystemExit(f"{stats['nonfinite_steps']} of {steps_per_epoch} steps had a "
                             f"non-finite loss: the run has diverged. Try amp_dtype: bfloat16, "
                             f"or a lower lr.")
        val = summarize(*predict(model, val_loader, amp_dtype), NUM_CLASSES)
        seconds = time.time() - start

        improved = val["macro_f1"] > track["best_f1"]
        if improved:
            track.update(best_f1=val["macro_f1"], best_epoch=epoch, bad_epochs=0)
            save({"model": model.state_dict(), "config": cfg, "epoch": epoch,
                  "val": scalars(val), "mean": mean, "std": std}, best_path)
        else:
            track["bad_epochs"] += 1

        row = {"epoch": epoch, "lr": lr_fn(track["step"] - 1), **stats,
               "val_nll": val["nll"], "val_acc": val["acc"], "val_macro_f1": val["macro_f1"],
               "val_ece": val["ece"], "time_s": seconds}
        track["history"].append(row)
        track["train_time_s"] += seconds
        track["peak_vram_gb"] = max(track["peak_vram_gb"],
                                    torch.cuda.max_memory_allocated() / 1024**3)
        try:
            with open(csv_path, "a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=CSV_FIELDS).writerow(row)
        except OSError:
            # Excel locks a CSV it has open. Losing a row of a convenience log
            # must not kill an overnight run; the checkpoint keeps every epoch.
            log(f"  could not write {csv_path} (open in Excel?); the epoch is still recorded")

        log(f"epoch {epoch:>2}/{cfg['epochs']}  train loss {stats['train_loss']:.4f} "
            f"acc {stats['train_acc']:.2%}  |  val loss {val['nll']:.4f} acc {val['acc']:.2%} "
            f"F1 {val['macro_f1']:.2%} ECE {val['ece']:.2%}  |  {hms(seconds)}"
            f"{'  * best' if improved else ''}")

        early = bool(patience) and track["bad_epochs"] >= patience
        track["epoch"] = epoch
        track["training_done"] = early or epoch >= cfg["epochs"]
        save({"config": cfg, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
              "scaler": scaler.state_dict(), "track": track}, last_path)

        if early:
            log(f"early stop: no val macro-F1 improvement for {patience} epochs")
        elif args.stop_after and epoch >= args.stop_after and not track["training_done"]:
            log(f"stopped after epoch {epoch} as asked; finish with --resume")
            return

    # The test split, exactly once, on the weights val chose. The train and val
    # workers go first: RAM, not VRAM, is the tighter limit on this machine.
    del train_loader, val_loader, loaders
    gc.collect()
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"])
    test = summarize(*predict(model, test_loader, amp_dtype), NUM_CLASSES)

    results = {
        "run_name": run,
        "status": "completed" if track["epoch"] >= cfg["epochs"] else "early_stopped",
        "config": cfg,
        "environment": environment(),
        "data": {"sizes": sizes, "fingerprint": fingerprints, "mean": list(mean), "std": list(std)},
        "model": {"params": n_params, "pretrained_weights": weights},
        "training": {"epochs_run": track["epoch"], "best_epoch": track["best_epoch"],
                     "train_time_s": track["train_time_s"], "peak_vram_gb": track["peak_vram_gb"],
                     "history": track["history"]},
        "val": best["val"],
        "test": {**scalars(test),
                 "per_class_f1": dict(zip(PLANTVILLAGE_CLASSES, test["per_class_f1"])),
                 "confusion_matrix": test["confusion_matrix"]},
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    with open(finished_marker, "w", encoding="utf-8") as fh:
        fh.write(results_path + "\n")

    log(f"\nbest epoch {track['best_epoch']}: val macro-F1 {best['val']['macro_f1']:.2%}")
    log(f"test   acc {test['acc']:.2%}  macro-F1 {test['macro_f1']:.2%}  "
        f"balanced acc {test['balanced_acc']:.2%}  ECE {test['ece']:.2%}  NLL {test['nll']:.4f}")
    log(f"time   {hms(track['train_time_s'])} training, peak VRAM {track['peak_vram_gb']:.2f} GB")
    log(f"saved  {os.path.relpath(results_path, paths.REPO_ROOT)}")


if __name__ == "__main__":
    main()
