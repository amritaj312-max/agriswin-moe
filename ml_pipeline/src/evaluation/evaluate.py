"""
Score a trained run on PlantVillage test and on PlantDoc, with uncertainty.

This is Phase 4 of docs/ACTION_PLAN.md and needs no training: it reads a run's
best.pt and produces the numbers training alone cannot give -- cross-dataset
accuracy, calibration, and what MC-dropout uncertainty buys under domain shift.
It exists before the ST-DAM ablations on purpose. The baselines land within
five test images of each other, so PlantVillage accuracy cannot show what a new
module adds; these are the measurements that can.

Per run it writes results/eval/<run>.json: deterministic and MC-dropout metrics
on both datasets, reliability bins, risk-coverage curves, and the per-sample
confidence and entropy the figures are drawn from.

Two things are checked rather than assumed. The deterministic PlantVillage
accuracy must equal what results/<run>.json recorded at training time -- if it
does not, the checkpoint and the result have drifted apart and the run is not
what the table says it is. And the split fingerprint must match, so the model
is scored on the same images it was scored on then.

MC-dropout runs the backbone once and only re-runs the head. The sole Dropout
module sits between the pooled features and the classifier (classifier.py says
why it is built there rather than taken from timm), so re-running the head is
exactly equivalent to a full forward pass, at a fraction of the cost. It also
makes T=20 passes as cheap as T=5, so the paper can report a proper estimate
while the phone still runs 5.

Usage:
    python ml_pipeline/src/evaluation/evaluate.py                  # every finished run
    python ml_pipeline/src/evaluation/evaluate.py --runs swinv2_tiny
    python ml_pipeline/src/evaluation/evaluate.py --passes 50
"""

import argparse
import glob
import json
import os
import sys
import time

import torch

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("data", "evaluation", "models"):
    sys.path.insert(0, os.path.join(SRC, sub))

import paths  # noqa: E402
from classes import NUM_CLASSES, PLANTVILLAGE_CLASSES  # noqa: E402
from classifier import Classifier  # noqa: E402
from dataset import PlantVillageSplit, eval_transform  # noqa: E402
from metrics import summarize as summarize_metrics  # noqa: E402
from plantdoc import MAPPED_CLASS_IDX, PlantDocDataset  # noqa: E402
from plantdoc import fingerprint as plantdoc_fingerprint  # noqa: E402
from uncertainty import summarize as summarize_uncertainty  # noqa: E402

CHECKPOINT_DIR = os.path.join(paths.REPO_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(paths.REPO_ROOT, "results")
EVAL_DIR = os.path.join(RESULTS_DIR, "eval")
# passes for the paper's estimate; 5 is also reported, as that is what the
# phone will run (project_breakdown_v3_final.md, Novelty 3)
DEFAULT_PASSES = 20
DEPLOYMENT_PASSES = 5
BATCH = 64
# test images allowed to change prediction when re-scoring on a GPU model other
# than the one that trained the run (see the integrity check in evaluate_run)
MAX_FLIPS = 2


def mc_dropout_modules(model):
    """The Dropout layers MC-dropout turns back on at inference: those that
    actually drop something. Swin-V2 carries ~49 Dropout(p=0) layers inside its
    blocks (the recipe sets timm's drop rates to 0), which do nothing in either
    mode and would only make the count misleading.

    Everything else, BatchNorm above all, stays in eval mode: switching the
    whole model to train() would also make BatchNorm use batch statistics,
    which changes predictions for reasons that have nothing to do with
    uncertainty.
    """
    return [m for m in model.modules() if isinstance(m, torch.nn.Dropout) and m.p > 0]


@torch.no_grad()
def features_and_logits(model, loader, amp_dtype, log):
    """Pooled features, deterministic logits and labels for a whole split."""
    model.eval()
    feats, logits, labels = [], [], []
    seen, start = 0, time.time()
    for images, y in loader:
        images = images.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            f = model.backbone(images)
            out = model.head(model.dropout(f))      # dropout is in eval mode here
        feats.append(f.float().cpu())
        logits.append(out.float().cpu())
        labels.append(y)
        seen += y.numel()
        if time.time() - start > 20:
            log(f"      {seen}/{len(loader.dataset)} images")
            start = time.time()
    return torch.cat(feats), torch.cat(logits), torch.cat(labels)


@torch.no_grad()
def mc_passes(model, feats, passes, seed):
    """(T, N, C) softmax probabilities, dropout on, from cached features."""
    model.eval()
    for module in mc_dropout_modules(model):
        module.train()
    torch.manual_seed(seed)                          # reproducible masks
    out = []
    for _ in range(passes):
        probs = []
        for i in range(0, len(feats), BATCH):
            chunk = feats[i:i + BATCH].cuda(non_blocking=True)
            probs.append(model.head(model.dropout(chunk)).float().softmax(1).cpu())
        out.append(torch.cat(probs))
    for module in mc_dropout_modules(model):
        module.eval()
    return torch.stack(out)


def restrict(logits, keep_idx):
    """Logits with every class PlantDoc cannot contain set to -inf."""
    masked = torch.full_like(logits, float("-inf"))
    masked[:, keep_idx] = logits[:, keep_idx]
    return masked


def evaluate_run(run, passes, log):
    result_path = os.path.join(RESULTS_DIR, f"{run}.json")
    best_path = os.path.join(CHECKPOINT_DIR, run, "best.pt")
    if not os.path.isfile(result_path) or not os.path.isfile(best_path):
        log(f"  {run}: needs both {result_path} and {best_path} -- skipped")
        return None

    with open(result_path, encoding="utf-8") as fh:
        trained = json.load(fh)
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    cfg = best["config"]
    mean, std = tuple(best["mean"]), tuple(best["std"])
    amp_dtype = torch.float16 if cfg["amp_dtype"] == "float16" else torch.bfloat16

    model = Classifier(cfg["model"], NUM_CLASSES, pretrained=False,
                       drop_rate=cfg["drop_rate"], drop_path_rate=cfg["drop_path_rate"])
    model.load_state_dict(best["model"])
    model.cuda()
    n_dropout = len(mc_dropout_modules(model))
    log(f"  {run}: {cfg['model']}, seed {cfg['seed']}, best epoch {best['epoch']}, "
        f"{n_dropout} active dropout layer(s) for MC")
    if n_dropout == 0:
        raise SystemExit(f"{run} has no Dropout module, so MC-dropout would return a "
                         f"constant: the checkpoint does not match classifier.py")

    transform = eval_transform(cfg["img_size"], mean, std)
    splits = {
        "plantvillage_test": PlantVillageSplit("test", transform),
        "plantdoc": PlantDocDataset(transform),
    }

    out = {}
    for name, dataset in splits.items():
        log(f"    {name}: {len(dataset)} images")
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=False,
                                             num_workers=cfg["num_workers"],
                                             pin_memory=True)
        feats, logits, labels = features_and_logits(model, loader, amp_dtype, log)
        entry = {"images": len(dataset), "deterministic": summarize_metrics(logits, labels, NUM_CLASSES)}
        # kept per image so compare.py can run paired tests: whether two models
        # differ is a question about the same images, not about two accuracies
        entry["predictions"] = logits.argmax(1).tolist()
        entry["labels"] = labels.tolist()

        if name == "plantdoc":
            # the 10 PlantVillage classes PlantDoc has no images of can never be
            # right here, so the masked number is the upper bound a perfect
            # class mapping would allow; the unmasked one is what deployment faces
            entry["deterministic_mapped_only"] = summarize_metrics(
                restrict(logits, MAPPED_CLASS_IDX), labels, NUM_CLASSES)

        probs = mc_passes(model, feats, passes, seed=cfg["seed"])
        entry["mc_dropout"] = summarize_uncertainty(probs, labels)
        entry[f"mc_dropout_{DEPLOYMENT_PASSES}_passes"] = {
            k: v for k, v in summarize_uncertainty(probs[:DEPLOYMENT_PASSES], labels).items()
            if k not in ("per_sample", "risk_coverage", "reliability")}
        out[name] = entry
        del feats, logits, probs
        torch.cuda.empty_cache()

    # --- the integrity check ------------------------------------------------
    # On the GPU model that trained the run, re-scoring must reproduce the
    # committed result exactly. On a different one (a laptop run scored on
    # Colab, say) fp16 kernels differ, and an image whose top two classes are
    # nearly tied can flip, so up to MAX_FLIPS test images may differ. A wrong
    # checkpoint is off by far more than that, so the check still catches it.
    pv = out["plantvillage_test"]["deterministic"]
    gpu_now = torch.cuda.get_device_name(0)
    same_gpu = gpu_now == trained["environment"].get("gpu")
    n_test = len(splits["plantvillage_test"])
    flips = round(abs(pv["acc"] - trained["test"]["acc"]) * n_test)
    allowed = 0 if same_gpu else MAX_FLIPS
    checks = {
        "test_images_differing_from_training": flips,
        "allowed": allowed,
        "trained_on_gpu": trained["environment"].get("gpu"),
        "split_fingerprint": trained["data"]["fingerprint"]["test"],
    }
    exact_f1 = abs(pv["macro_f1"] - trained["test"]["macro_f1"]) < 1e-9
    if flips > allowed or (same_gpu and not exact_f1):
        raise SystemExit(
            f"{run}: re-scoring best.pt gives test acc {pv['acc']:.6f} / macro-F1 "
            f"{pv['macro_f1']:.6f}, but results/{run}.json recorded "
            f"{trained['test']['acc']:.6f} / {trained['test']['macro_f1']:.6f} "
            f"({flips} images differ, {allowed} allowed on "
            f"{'the same' if same_gpu else 'a different'} GPU). The checkpoint and the "
            f"published result are not the same model.")
    log(f"    checks: PlantVillage test reproduces the committed result "
        + ("exactly" if flips == 0 else f"within {flips} image(s), on a different GPU"))

    del model
    torch.cuda.empty_cache()
    return {
        "run_name": run,
        "model": cfg["model"],
        "seed": cfg["seed"],
        "params": trained["model"]["params"],
        "best_epoch": best["epoch"],
        "passes": passes,
        "deployment_passes": DEPLOYMENT_PASSES,
        "trained_at_commit": trained["environment"]["git_commit"],
        "evaluated_on_gpu": torch.cuda.get_device_name(0),
        # which PlantDoc images, by content: compare.py refuses to put runs side
        # by side unless this matches, however their files were named on disk
        "plantdoc_fingerprint": plantdoc_fingerprint(),
        "checks": checks,
        "splits": out,
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset and uncertainty evaluation")
    parser.add_argument("--runs", nargs="+", help="run names (default: every finished run)")
    parser.add_argument("--passes", type=int, default=DEFAULT_PASSES,
                        help=f"MC-dropout forward passes (default {DEFAULT_PASSES})")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; run ml_pipeline/src/data/check_env.py")

    runs = args.runs or sorted(
        os.path.basename(p)[:-len(".json")] for p in glob.glob(os.path.join(RESULTS_DIR, "*.json"))
        if not os.path.basename(p).startswith("dev_"))
    os.makedirs(EVAL_DIR, exist_ok=True)

    def log(message):
        print(message, flush=True)

    log(f"Evaluating {len(runs)} run(s) with {args.passes} MC-dropout passes\n")
    done = []
    for run in runs:
        started = time.time()
        result = evaluate_run(run, args.passes, log)
        if result is None:
            continue
        with open(os.path.join(EVAL_DIR, f"{run}.json"), "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        log(f"    saved results/eval/{run}.json  ({time.time() - started:.0f}s)\n")
        done.append(result)

    if not done:
        return
    header = (f"{'run':<24}{'PV acc':>8}{'PV ECE':>8}{'PD acc':>8}{'PD acc*':>9}"
              f"{'PD ECE':>8}{'PD AURC':>9}{'PD@50%':>8}{'epist.':>8}")
    print(header)
    print("-" * len(header))
    for r in done:
        pv, pd = r["splits"]["plantvillage_test"], r["splits"]["plantdoc"]
        mc = pd["mc_dropout"]
        print(f"{r['run_name']:<24}{pv['deterministic']['acc']:>8.2%}{pv['deterministic']['ece']:>8.2%}"
              f"{pd['deterministic']['acc']:>8.2%}{pd['deterministic_mapped_only']['acc']:>9.2%}"
              f"{pd['deterministic']['ece']:>8.2%}{mc['aurc_by_confidence']:>9.3f}"
              f"{mc['acc_at_coverage']['0.5']:>8.2%}{mc['entropy']['epistemic']:>8.3f}")
    print("\nPV = PlantVillage test, PD = PlantDoc (cross-dataset, never trained on).")
    print("PD acc* restricts predictions to the 28 classes PlantDoc contains.")
    print("AURC: mean error over all coverages, lower is better. PD@50%: accuracy when the")
    print("least confident half is left to a human. epist.: mean epistemic entropy (nats).")


if __name__ == "__main__":
    main()
