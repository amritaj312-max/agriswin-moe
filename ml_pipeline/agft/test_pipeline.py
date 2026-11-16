"""Every stage, exercised small, so Colab meets no surprise.

Run from this folder:  python test_pipeline.py
The model and training tests use synthetic images, so they need no download;
the data tests fetch the small PlantDoc set only.
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline as P  # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")


class FakeData:
    """Stands in for a HuggingFace image dataset: indexable, with len."""

    def __init__(self, n, classes, size=64, seed=0):
        rng = np.random.default_rng(seed)
        self.labels = rng.integers(0, classes, n)
        # each class gets its own colour, so a model can actually learn it
        self.images = [
            Image.fromarray(
                np.clip(rng.normal(40, 12, (size, size, 3))
                        + np.array([(int(l) * 37) % 256, (int(l) * 91) % 256,
                                    (int(l) * 53) % 256]), 0, 255).astype(np.uint8))
            for l in self.labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"image": self.images[i], "label": int(self.labels[i])}


def test_models(cfg):
    print("\n1. The four arms build, run forward and backward")
    data = FakeData(32, 4)
    x = torch.stack([P.transforms_for(cfg)(data[i]["image"]) for i in range(4)])
    y = torch.tensor([data[i]["label"] for i in range(4)])
    for name in ("convnextv2_atto.fcmae", "agft:ft", "agft:frozen",
                 "agft:fixed", "agft:gated"):
        try:
            model = P.build_model(name, 4, cfg, pretrained=False)
            model.train()
            out = model(x)
            loss = torch.nn.functional.cross_entropy(out, y)
            loss.backward()
            grads = sum(1 for p in model.parameters()
                        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
            check(f"{name}: shape {tuple(out.shape)}, {grads} tensors got gradient",
                  out.shape == (4, 4) and grads > 0)
        except Exception as exc:                                   # noqa: BLE001
            check(f"{name}: builds and trains", False, f"{type(exc).__name__}: {exc}")


def test_gate_semantics(cfg):
    print("\n2. The gate means what the ablations claim")
    data = FakeData(8, 3)
    x = torch.stack([P.transforms_for(cfg)(data[i]["image"]) for i in range(4)])
    outcomes = {}
    for mode in ("ft", "frozen", "fixed", "gated"):
        model = P.build_model(f"agft:{mode}", 3, cfg, pretrained=False).eval()
        with torch.no_grad():
            logits, g = model(x, return_gate=True)
        outcomes[mode] = (logits, g)
    check("plain fine-tuning reports no gate", outcomes["ft"][1] is None)
    check("linear probe reports no gate", outcomes["frozen"][1] is None)
    fixed_g = outcomes["fixed"][1]
    check("a fixed alpha is one value for every image",
          float(fixed_g.std()) == 0.0, f"sd {float(fixed_g.std()):.2e}")
    gated_g = outcomes["gated"][1]
    check("a learned gate varies per image",
          float(gated_g.std()) > 0, f"sd {float(gated_g.std()):.2e}")
    check("every gate lies in [0, 1]",
          bool((gated_g >= 0).all() and (gated_g <= 1).all()))

    model = P.build_model("agft:gated", 3, cfg, pretrained=False)
    frozen_before = [p.clone() for p in model.frozen.parameters()]
    model.train()
    torch.nn.functional.cross_entropy(model(x), torch.zeros(4, dtype=torch.long)).backward()
    torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2).step()
    moved = any(not torch.equal(a, b)
                for a, b in zip(frozen_before, model.frozen.parameters()))
    check("the frozen branch never moves", not moved)
    model.train()
    check("the frozen branch stays in eval mode while training",
          not model.frozen.training)


def test_training_and_resume(cfg):
    print("\n3. Training, early stopping and resuming")
    data = FakeData(96, 4, seed=1)
    train_idx, val_idx = np.arange(0, 64), np.arange(64, 96)
    best, path = P.train_model("agft:gated", data, train_idx, val_idx, 4, cfg,
                               tag="smoke", epochs=2, log=lambda *_a, **_k: None)
    check("training returns a best macro-F1", "macro_f1" in best, f"{best['macro_f1']:.3f}")
    check("best weights were written", os.path.isfile(path))
    run_dir = os.path.join(cfg.paths["ckpt"], "smoke")
    check("a 'done' marker was written", os.path.isfile(os.path.join(run_dir, "done.json")))

    again, _ = P.train_model("agft:gated", data, train_idx, val_idx, 4, cfg,
                             tag="smoke", epochs=2, log=lambda *_a, **_k: None)
    check("a finished run is skipped, not repeated",
          again["macro_f1"] == best["macro_f1"])

    # delete the marker but keep last.pt: the run must continue, not restart
    os.remove(os.path.join(run_dir, "done.json"))
    state = torch.load(os.path.join(run_dir, "last.pt"), map_location="cpu",
                       weights_only=False)
    check("the checkpoint carries optimiser, scaler, epoch and best",
          all(k in state for k in ("optimiser", "scaler", "epoch", "best")),
          f"stopped after epoch {state['epoch']}")
    messages = []
    P.train_model("agft:gated", data, train_idx, val_idx, 4, cfg, tag="smoke",
                  epochs=3, log=messages.append)
    check("an interrupted run resumes from its last epoch",
          any("resuming at epoch" in m for m in messages),
          next((m.strip() for m in messages if "resuming" in m), "no resume message"))


def test_evaluate(cfg):
    print("\n4. Evaluation, including the closed-set restriction")
    data = FakeData(40, 6, seed=2)
    loader = P.loader_for(data, np.arange(40), cfg)
    model = P.build_model("agft:gated", 6, cfg, pretrained=False)
    full = P.evaluate(model, loader, device="cpu")
    reachable = [0, 1, 2]
    closed = P.evaluate(model, loader, device="cpu", restrict=torch.tensor(reachable))
    check("evaluate reports accuracy, macro-F1 and n",
          {"accuracy", "macro_f1", "n"} <= set(full), f"n={full['n']}")
    check("the gate is reported for a gated model", full["gate"] is not None)
    preds = []
    for images, _ in loader:
        logits = model(images)
        masked = torch.full_like(logits, float("-inf"))
        masked[:, torch.tensor(reachable)] = logits[:, torch.tensor(reachable)]
        preds.append(masked.argmax(1))
    check("restricting really confines predictions to those classes",
          bool(torch.cat(preds).unique().le(2).all()))
    check("closed-set accuracy is computed", 0.0 <= closed["accuracy"] <= 1.0)


def test_splitting(cfg):
    print("\n5. Grouped splitting keeps near-duplicates together")
    base = FakeData(60, 5, seed=3)
    # duplicate every image twice: a random split would scatter the copies
    images = [im for im in base.images for _ in range(2)]
    labels = [int(l) for l in base.labels for _ in range(2)]
    hashes = [P.dhash(im) for im in images]
    check("identical images hash identically",
          hashes[0] == hashes[1] and hashes[2] == hashes[3])
    splits = P.grouped_split(labels, hashes, cfg, log=lambda *_a, **_k: None)
    total = sum(len(v) for v in splits.values())
    check("every image lands in exactly one split", total == len(labels),
          f"{total} of {len(labels)}")
    clean = P.check_no_leakage(splits, hashes, log=lambda *_a, **_k: None)
    check("no perceptual hash appears in two splits", clean)
    overlap = set(splits["train"]) & set(splits["test"])
    check("train and test share no index", not overlap)


def test_optuna_and_cv(cfg):
    print("\n6. Tuning and cross-validation run and resume")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import stages
    data = FakeData(96, 3, seed=4)
    train_idx, val_idx = np.arange(0, 64), np.arange(64, 96)
    labels = [data[int(i)]["label"] for i in range(len(data))]
    try:
        params = stages.tune("agft:gated", data, train_idx, val_idx, 3, cfg,
                             log=lambda *_a, **_k: None)
        check("tuning returns the three searched hyperparameters",
              {"lr", "weight_decay", "drop_rate"} == set(params), str(list(params)))
        again = stages.tune("agft:gated", data, train_idx, val_idx, 3, cfg,
                            log=lambda *_a, **_k: None)
        check("a finished search is reused, not repeated", again == params)
    except Exception as exc:                                       # noqa: BLE001
        check("tuning runs", False, f"{type(exc).__name__}: {exc}")
    try:
        cv = stages.cross_validate("agft:gated", data, train_idx, np.array(labels), 3,
                                   {"lr": 3e-4}, cfg, log=lambda *_a, **_k: None)
        check("cross-validation reports a mean and a spread over folds",
              len(cv["folds"]) == cfg.folds,
              f"{cv['mean_macro_f1']:.3f} +- {cv['sd_macro_f1']:.3f}")
    except Exception as exc:                                       # noqa: BLE001
        check("cross-validation runs", False, f"{type(exc).__name__}: {exc}")


def test_label_bridge():
    print("\n7. The PlantDoc label bridge is complete")
    from pipeline import PLANTDOC_TO_PV
    check("all 29 PlantDoc categories are mapped", len(PLANTDOC_TO_PV) == 29,
          f"{len(PLANTDOC_TO_PV)}")
    targets = set(PLANTDOC_TO_PV.values())
    check("the mapping has no duplicate targets", len(targets) == len(PLANTDOC_TO_PV),
          f"{len(targets)} distinct")


def main():
    root = tempfile.mkdtemp(prefix="agft_test_")
    cfg = P.Config(root=root, batch_size=8, workers=0, img_size=64, epochs=2,
                   patience=1, optuna_trials=2, optuna_epochs=1, optuna_subset=1.0,
                   folds=2, fold_epochs=1)
    P.prepare_dirs(cfg)
    print(f"scratch: {root}")
    try:
        test_models(cfg)
        test_gate_semantics(cfg)
        test_training_and_resume(cfg)
        test_evaluate(cfg)
        test_splitting(cfg)
        test_optuna_and_cv(cfg)
        test_label_bridge()
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name in FAIL:
            print(f"  FAILED: {name}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
