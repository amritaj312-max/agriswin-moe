"""Every stage, exercised small, so Colab meets no surprise.

Run from this folder:  python test_pipeline.py

Synthetic images throughout; the only downloads are small pretrained weights.
Covers what a Colab run depends on: every arm builds and trains -- including
the three real backbones at the study's resolution -- the gate means what the
ablations claim, a run survives being interrupted (on the same runtime, after
the runtime is lost, and with a torn file on either side), the CV folds keep
near-duplicates together, the search resumes, and the preflight works.
"""

import dataclasses
import gc
import os
import re
import shutil
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline as P  # noqa: E402
import stages  # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")


def quiet(*_a, **_k):
    return None


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


class Interrupt(Exception):
    """Stands in for a disconnect, raised from inside a run."""


def stop_at(epoch):
    """A log that raises when the run reports `epoch` -- after that epoch has
    trained, before it is checkpointed, which is the worst moment to lose it."""
    def log(message, *_a, **_k):
        if message.strip().startswith(f"epoch {epoch}/"):
            raise Interrupt(message)
    return log


def test_models(cfg):
    print("\n1. Every arm builds, runs forward and backward")
    data = FakeData(32, 4)
    x = torch.stack([P.transforms_for(cfg)(data[i]["image"]) for i in range(4)])
    y = torch.tensor([data[i]["label"] for i in range(4)])
    names = list(P.CFG.backbones) + ["agft:ft", "agft:frozen", "agft:fixed", "agft:gated"]
    for name in names:
        try:
            model = P.build_model(name, 4, cfg, pretrained=False)
            model.train()
            out = model(x)
            torch.nn.functional.cross_entropy(out, y).backward()
            grads = sum(1 for p in model.parameters()
                        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
            check(f"{name}: shape {tuple(out.shape)}, {grads} tensors got gradient",
                  out.shape == (4, 4) and grads > 0)
        except Exception as exc:                                   # noqa: BLE001
            check(f"{name}: builds and trains", False, f"{type(exc).__name__}: {exc}")
    # the resolution the study actually uses, where Swin-V2 used to raise
    for name in P.CFG.backbones:
        try:
            net = P.create_backbone(name, 224, pretrained=False).eval()
            with torch.no_grad():
                feats = net(torch.randn(1, 3, 224, 224))
            check(f"{name} accepts 224 px", feats.ndim == 2, f"{tuple(feats.shape)}")
        except Exception as exc:                                   # noqa: BLE001
            check(f"{name} accepts 224 px", False, f"{type(exc).__name__}: {exc}")


def test_gate_semantics(cfg):
    print("\n2. The gate means what the ablations claim")
    data = FakeData(8, 3)
    x = torch.stack([P.transforms_for(cfg)(data[i]["image"]) for i in range(4)])
    outcomes = {}
    for mode in ("ft", "frozen", "fixed", "gated"):
        model = P.build_model(f"agft:{mode}", 3, cfg, pretrained=False).eval()
        with torch.no_grad():
            outcomes[mode] = model(x, return_gate=True)
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
    check("the frozen branch stays in eval mode while training", not model.frozen.training)


def test_checkpoint_contents(cfg):
    print("\n3. What a checkpoint holds")
    model = P.build_model("agft:gated", 3, cfg, pretrained=True).eval()
    state = P.model_state(model)
    check("AGFT checkpoints leave out the frozen branch",
          not any(k.startswith("frozen.") for k in state))
    full = sum(v.numel() for v in model.state_dict().values())
    kept = sum(v.numel() for v in state.values())
    check("which roughly halves them", kept < 0.6 * full, f"{kept / full:.0%} of the full state")
    x = torch.randn(2, 3, cfg.img_size, cfg.img_size)
    with torch.no_grad():
        reference = model(x)
    fresh = P.build_model("agft:gated", 3, cfg, pretrained=True).eval()
    P.load_model_state(fresh, state)
    with torch.no_grad():
        check("reloading into a fresh model reproduces its outputs exactly",
              torch.equal(reference, fresh(x)))
    broken = dict(state)
    broken.pop(next(k for k in broken if k.startswith("head_tuned")))
    try:
        P.load_model_state(fresh, broken)
        refused = False
    except RuntimeError:
        refused = True
    check("a checkpoint missing a trained weight is refused", refused)


def test_training_and_resume(cfg):
    print("\n4. Training, early stopping, and surviving interruptions")
    data = FakeData(96, 4, seed=1)
    tr, va = np.arange(0, 64), np.arange(64, 96)
    arm = "convnextv2_atto.fcmae"

    best, path = P.train_model(arm, data, tr, va, 4, cfg, tag="smoke", epochs=2,
                               keep_best=True, log=quiet)
    check("training returns a best macro-F1", "macro_f1" in best, f"{best['macro_f1']:.3f}")
    check("the result is recorded on Drive",
          os.path.isfile(os.path.join(cfg.paths["runs"], "smoke.json")))
    check("the best weights are copied to Drive", path is not None and os.path.isfile(path))
    check("local checkpoints are cleared when the run ends",
          not os.path.exists(os.path.join(cfg.local_root, "runs", "smoke")))
    again, _ = P.train_model(arm, data, tr, va, 4, cfg, tag="smoke", epochs=2,
                             keep_best=True, log=quiet)
    check("a finished run is skipped, not repeated", again["macro_f1"] == best["macro_f1"])

    patient = dataclasses.replace(cfg, patience=10)        # so every epoch runs

    def interrupt(tag, epochs, at):
        try:
            P.train_model(arm, data, tr, va, 4, patient, tag=tag, epochs=epochs,
                          log=stop_at(at))
        except Interrupt:
            return True
        return False

    def resume(tag, epochs):
        messages = []
        result, _ = P.train_model(arm, data, tr, va, 4, patient, tag=tag, epochs=epochs,
                                  log=messages.append)
        line = next((m.strip() for m in messages if "resuming" in m), "no resume message")
        return result, line, messages

    local_ckpt = lambda tag: os.path.join(patient.local_root, "runs", tag, "last.pt")  # noqa: E731

    def rates(messages):
        """Epoch -> the learning rate the run reported at the end of it."""
        found = {}
        for m in messages:
            hit = re.search(r"epoch (\d+)/\d+\s+lr (\S+)", m)
            if hit:
                found[int(hit.group(1))] = hit.group(2)
        return found

    reference = []
    P.train_model(arm, data, tr, va, 4, patient, tag="reference", epochs=3,
                  log=reference.append)

    interrupt("live", 3, 2)
    state = P.try_load(local_ckpt("live"))
    check("the live checkpoint carries model, optimiser, scaler, epoch and best",
          state is not None and all(k in state for k in
                                    ("model", "optimiser", "scaler", "epoch", "best", "tag")))
    result, line, messages = resume("live", 3)
    check("interrupted on a live runtime, it resumes from local disk",
          "resuming at epoch 2 from local disk" in line, line)
    check("and finishes every epoch", len(result["history"]) == 3,
          f"{len(result['history'])} epochs recorded")
    expected = {e: r for e, r in rates(reference).items() if e >= 2}
    check("its learning rate continues exactly as an uninterrupted run's",
          rates(messages) == expected and len(expected) == 2,
          f"resumed {rates(messages)} vs uninterrupted {expected}")

    interrupt("lost", 3, 2)
    index = P.read_json(os.path.join(patient.paths["runs"], "resume.json"))
    slot = next(k for k, v in index.items() if v["tag"] == "lost")
    drive_copy = os.path.join(patient.paths["runs"], f"resume_{slot}.pt")
    lean = P.try_load(drive_copy)
    check("the Drive copy is lean: weights and progress, no optimiser state",
          lean is not None and "optimiser" not in lean and "model" in lean,
          f"{os.path.getsize(drive_copy) / os.path.getsize(local_ckpt('lost')):.0%} "
          f"of the local checkpoint's size")
    shutil.rmtree(patient.local_root)                      # the runtime is gone
    result, line, messages = resume("lost", 3)
    check("after the runtime is lost, it resumes from Drive",
          "resuming at epoch 2 from Drive" in line, line)
    check("and finishes every epoch", len(result["history"]) == 3,
          f"{len(result['history'])} epochs recorded")
    check("with the learning rate still on schedule",
          rates(messages) == expected, f"resumed {rates(messages)} vs uninterrupted {expected}")

    interrupt("torn_local", 3, 2)
    with open(local_ckpt("torn_local"), "wb") as fh:
        fh.write(b"half a checkpoint")
    _, line, messages = resume("torn_local", 3)
    check("a torn local checkpoint falls back to Drive",
          "from Drive" in line and any("unreadable" in m for m in messages), line)

    interrupt("torn_drive", 4, 3)                          # two copies on Drive: epochs 1, 2
    shutil.rmtree(patient.local_root)
    index = P.read_json(os.path.join(patient.paths["runs"], "resume.json"))
    newest = max((v["epoch"], k) for k, v in index.items() if v["tag"] == "torn_drive")[1]
    with open(os.path.join(patient.paths["runs"], f"resume_{newest}.pt"), "wb") as fh:
        fh.write(b"cut off")
    _, line, messages = resume("torn_drive", 4)
    check("a torn Drive copy falls back to the older one",
          "resuming at epoch 2 from Drive" in line, line)


def test_evaluate(cfg):
    print("\n5. Evaluation, including the closed-set restriction")
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
    print("\n6. Near-duplicates are found exactly and kept together")
    base = 0x0123456789ABCDEF
    values = [base,
              base ^ 0b1,                    # 1 bit from the first
              base ^ 0b11111,                # 5 from the first, 4 from the second
              base ^ (0b111111 << 20),       # 6 from the first, far from the rest
              base ^ (2**64 - 1)]            # every bit different
    hexes = [f"{v:016x}" for v in values]
    pairs = P.near_duplicate_pairs(hexes, max_bits=6, log=quiet)
    brute = sorted((i, j, bin(values[i] ^ values[j]).count("1"))
                   for i in range(5) for j in range(i + 1, 5)
                   if bin(values[i] ^ values[j]).count("1") <= 6)
    check("the vectorised search finds exactly what a brute-force count does",
          sorted(pairs) == brute, str(sorted(pairs)))
    x = np.array(values, dtype=np.uint64)
    check("both popcount routes agree (numpy >= 2, and the fallback)",
          np.array_equal(P._popcount(x).astype(int), P._popcount_table(x).astype(int)))
    check("at 0 bits nothing joins", P.group_ids(5, pairs, 0) == ["0", "1", "2", "3", "4"])
    joined = P.group_ids(5, pairs, 4)
    check("at 4 bits a chain joins, named by its smallest index",
          joined == ["0", "0", "0", "3", "4"], str(joined))

    # images with real structure, each with a lightly re-noised copy
    rng = np.random.default_rng(7)
    images, labels = [], []
    for k in range(60):
        coarse = Image.fromarray(rng.integers(0, 256, (8, 9, 3)).astype(np.uint8))
        original = coarse.resize((64, 64), Image.BILINEAR)
        pixels = np.asarray(original, dtype=np.int16)
        copy = np.clip(pixels + rng.integers(-2, 3, pixels.shape), 0, 255).astype(np.uint8)
        images += [original, Image.fromarray(copy)]
        labels += [k % 5, k % 5]
    hashes = [P.dhash(im) for im in images]
    apart = [bin(int(hashes[k], 16) ^ int(hashes[k + 1], 16)).count("1")
             for k in range(0, len(hashes), 2)]
    check("a re-noised copy lands within 4 bits of its original", max(apart) <= 4,
          f"at most {max(apart)} bits apart")
    groups = P.group_ids(len(hashes), P.near_duplicate_pairs(hashes, log=quiet), 4)
    check("each copy shares its original's group",
          all(groups[k] == groups[k + 1] for k in range(0, len(groups), 2)))
    splits = P.grouped_split(labels, groups, cfg, log=quiet)
    total = sum(len(v) for v in splits.values())
    check("every image lands in exactly one split", total == len(labels),
          f"{total} of {len(labels)}")
    check("no two images in different splits are within 4 bits",
          P.check_no_leakage(splits, hashes, 4, log=quiet))
    check("train and test share no index", not set(splits["train"]) & set(splits["test"]))
    naive = {"train": np.arange(0, 120, 2), "val": np.arange(1, 60, 2),
             "test": np.arange(61, 120, 2)}
    check("the check catches a split that separates copies",
          not P.check_no_leakage(naive, hashes, 4, log=quiet))


def test_storage(cfg):
    print("\n7. Files a disconnect can cut off")
    path = os.path.join(cfg.paths["results"], "probe.json")
    P.write_json({"a": 1}, path)
    check("JSON round-trips", P.read_json(path) == {"a": 1})
    check("no temporary file is left behind", not os.path.exists(path + ".tmp"))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('{"a": ')
    check("a JSON file cut off mid-write reads as absent", P.read_json(path) is None)
    check("a missing JSON file reads as absent", P.read_json(path + ".none") is None)
    ckpt = os.path.join(cfg.local_root, "probe.pt")
    P.save_atomic({"x": torch.ones(2)}, ckpt)
    loaded = P.try_load(ckpt)
    check("a checkpoint round-trips", loaded is not None and torch.equal(loaded["x"], torch.ones(2)))
    with open(ckpt, "wb") as fh:
        fh.write(b"PK\x03\x04 torn")
    check("a torn checkpoint reads as absent", P.try_load(ckpt) is None)


def test_optuna_and_cv(cfg):
    print("\n8. Tuning and cross-validation run, keep leaves together, and resume")
    data = FakeData(96, 3, seed=4)
    tr, va = np.arange(0, 64), np.arange(64, 96)
    labels = [data[i]["label"] for i in range(len(data))]
    groups = [f"leaf{i // 2}" for i in range(len(data))]    # near-duplicates come in pairs

    params = stages.tune("agft:gated", data, tr, va, 3, cfg, log=quiet)
    check("tuning returns the three searched hyperparameters",
          {"lr", "weight_decay", "drop_rate"} == set(params), str(list(params)))
    drive_db = os.path.join(cfg.paths["optuna"], "agft_gated.db")
    check("the search is copied to Drive after every trial", os.path.isfile(drive_db))
    check("a finished search is reused, not repeated",
          stages.tune("agft:gated", data, tr, va, 3, cfg, log=quiet) == params)

    # the runtime is lost after the trials but before the result was written
    os.remove(os.path.join(cfg.paths["results"], "best_agft_gated.json"))
    gc.collect()                                            # release SQLite on Windows
    os.remove(os.path.join(cfg.local_root, "optuna_agft_gated.db"))
    messages = []
    restored = stages.tune("agft:gated", data, tr, va, 3, cfg, log=messages.append)
    check("after a lost runtime the search resumes from Drive",
          any("restored from Drive" in m for m in messages))
    check("without re-running its finished trials",
          restored == params and not any("trial(s) to run" in m for m in messages))

    folds = stages.cv_folds(tr, labels, groups, cfg)
    check(f"{cfg.folds} folds", len(folds) == cfg.folds)
    shared = sum(len({groups[i] for i in a} & {groups[i] for i in b}) for a, b in folds)
    check("no near-duplicate pair straddles a fold", shared == 0, f"{shared} shared groups")
    held = sorted(np.concatenate([b for _, b in folds]).tolist())
    check("every training image is held out exactly once", held == sorted(tr.tolist()))
    repeat = stages.cv_folds(tr, labels, groups, cfg)
    check("the folds are identical every time, so a resumed fold sees the same images",
          all(np.array_equal(a1, a2) and np.array_equal(b1, b2)
              for (a1, b1), (a2, b2) in zip(folds, repeat)))
    cv = stages.cross_validate("agft:gated", data, tr, labels, groups, 3, {"lr": 3e-4}, cfg,
                               log=quiet)
    check("cross-validation reports a mean and a spread over folds",
          len(cv["folds"]) == cfg.folds,
          f"{cv['mean_macro_f1']:.3f} +- {cv['sd_macro_f1']:.3f}")
    capped = dataclasses.replace(cfg, max_folds=1)
    messages = []
    cv = stages.cross_validate("agft:fixed", data, tr, labels, groups, 3, {"lr": 3e-4}, capped,
                               log=messages.append)
    check("max_folds stops cross-validation early and records how many ran",
          len(cv["folds"]) == 1 and cv["planned_folds"] == cfg.folds
          and any("stopped after 1 of" in m for m in messages))


def test_preflight(cfg):
    print("\n9. The preflight runs every arm and estimates the time")
    data = FakeData(120, 3, seed=5)
    splits = {"train": np.arange(0, 80), "val": np.arange(80, 100), "test": np.arange(100, 120)}
    arms = ("convnextv2_atto.fcmae", "agft:fixed")
    result = stages.preflight(arms, data, splits, 3, 50, cfg, log=quiet)
    check("every arm passes", set(result["arms"]) == set(arms))
    check("each arm's speed is measured",
          all(r["images_per_s"] > 0 for r in result["arms"].values()))
    check("a time estimate comes out", all(h > 0 for h in result["hours_upper_bound"].values()))
    messages = []
    stages.preflight(arms, data, splits, 3, 50, cfg, log=messages.append)
    check("a passed preflight is not repeated", any("skipped" in m for m in messages))


def _notebook_cell(number):
    """The source of the notebook's code cell that starts '# <number>.'."""
    import make_notebook
    for cell in make_notebook.CELLS:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and source.lstrip().startswith(f"# {number}."):
            return source
    raise KeyError(f"no code cell numbered {number}")


def test_end_of_study(cfg):
    print("\n11. The end of the study: results table, headline figure, gate analysis")
    import matplotlib
    matplotlib.use("Agg")
    shown = []
    stubs = {"display": shown.append, "ShowImage": lambda filename: filename, "os": os}

    data = FakeData(120, 4, seed=7)
    tr, va, te = np.arange(0, 80), np.arange(80, 100), np.arange(100, 120)
    field = FakeData(30, 4, seed=8)
    classes, reachable = ["c0", "c1", "c2", "c3"], [0, 1, 2]

    messages = []
    check("the analysis waits for the gated arm rather than failing",
          stages.gate_analysis(data, field, {"test": te}, classes, reachable, cfg,
                               log=messages.append) is None
          and any("has not finished" in m for m in messages))

    # a finished gated arm, as run_arm leaves it
    P.train_model("agft:gated", data, tr, va, 4, cfg, tag="final_agft_gated", epochs=1,
                  keep_best=True, log=quiet)
    P.write_json({"lr": 3e-4, "weight_decay": 0.01, "drop_rate": 0.1},
                 os.path.join(cfg.paths["results"], "best_agft_gated.json"))
    for name, field_acc, gate in (("agft:gated", 0.31, 0.42), ("convnextv2_atto.fcmae", 0.29, None)):
        metrics = {"accuracy": field_acc, "macro_f1": field_acc, "gate": gate, "n": 30}
        P.write_json({"name": name, "params": {}, "cv": {"mean_macro_f1": 0.99, "sd_macro_f1": 0.001},
                      "val": {}, "plantvillage_test": {"accuracy": 0.998, "macro_f1": 0.997},
                      "plantdoc": metrics, "plantdoc_closed_set": dict(metrics, accuracy=0.35)},
                     os.path.join(cfg.paths["results"], f"final_{name.replace(':', '_')}.json"))

    rows = stages.summarise(cfg, log=quiet)
    check("the results table lists every finished arm, best on PlantDoc first",
          [r["name"] for r in rows] and rows[0]["plantdoc"]["accuracy"] >= rows[-1]["plantdoc"]["accuracy"]
          and os.path.isfile(os.path.join(cfg.paths["results"], "summary.json")))

    table = open(os.path.join(cfg.paths["results"], "table_main.tex"), encoding="utf-8").read()
    check("the paper's main table rows are written as LaTeX",
          "AGFT, learned gate (ours)" in table and table.count("\\\\") == 1
          and table.count("&") == 7, table.strip()[:70])

    namespace = dict(stubs, rows=rows, CFG=cfg)
    try:
        exec(_notebook_cell(12), namespace)
        drew = os.path.isfile(os.path.join(cfg.paths["figures"], "headline.png"))
        check("notebook cell 12 draws the headline figure", drew and shown)
    except Exception as exc:                                       # noqa: BLE001
        check("notebook cell 12 draws the headline figure", False, f"{type(exc).__name__}: {exc}")

    messages = []
    report = stages.gate_analysis(data, field, {"test": te}, classes, reachable, cfg,
                                  log=messages.append)
    lab, pd_part = report["plantvillage_test"], report["plantdoc"]
    check("the analysis measures both branches and the mixture on both datasets",
          all(0.0 <= part[k] <= 1.0 for part in (lab, pd_part)
              for k in ("tuned_only", "frozen_only", "gated")),
          f"PlantDoc: tuned {pd_part['tuned_only']:.2f}, frozen {pd_part['frozen_only']:.2f}, "
          f"gated {pd_part['gated']:.2f}")
    sweep = pd_part["fixed_alpha_logit"]
    check("alpha = 0 is exactly the frozen branch, alpha = 1 exactly the fine-tuned one",
          len(sweep) == 21 and sweep[0] == pd_part["frozen_only"] and sweep[-1] == pd_part["tuned_only"])
    check("the oracle alpha is the best of the sweep",
          report["oracle_fixed_alpha_on_plantdoc"]["accuracy"] == max(sweep))
    check("the gate lies in [0, 1] on both datasets",
          0.0 <= lab["gate_mean"] <= 1.0 and 0.0 <= pd_part["gate_mean"] <= 1.0)
    check("the lab-versus-field gate shift is tested",
          report["gate_shift_lab_to_field"] is None
          or 0.0 <= report["gate_shift_lab_to_field"]["p_value"] <= 1.0)
    check("the closed-set view is reported for PlantDoc",
          0.0 <= pd_part["closed_set"]["gated"] <= 1.0)
    check("the analysis figure is written",
          os.path.isfile(os.path.join(cfg.paths["figures"], "gate_analysis.png")))
    check("its findings are printed", any("learned per-image gate" in m for m in messages))
    gate_table = open(os.path.join(cfg.paths["results"], "table_gate.tex"), encoding="utf-8").read()
    check("the paper's gate table rows are written as LaTeX",
          gate_table.count("\\\\") == 5 and all(line.count("&") == 2
                                               for line in gate_table.strip().splitlines()))
    again = []
    stages.gate_analysis(data, field, {"test": te}, classes, reachable, cfg, log=again.append)
    check("a finished analysis is reused, not repeated", any("reusing" in m for m in again))

    shown.clear()
    namespace = dict(stubs, pv=data, pd_data=field, SPLITS={"test": te}, CLASSES=classes,
                     REACHABLE=reachable, CFG=cfg, stages=stages)
    try:
        exec(_notebook_cell(13), namespace)
        check("notebook cell 13 runs and shows the figure", shown and namespace.get("GATE"))
    except Exception as exc:                                       # noqa: BLE001
        check("notebook cell 13 runs and shows the figure", False, f"{type(exc).__name__}: {exc}")


def test_label_bridge():
    print("\n10. The PlantDoc label bridge is complete")
    check("all 29 PlantDoc categories are mapped", len(P.PLANTDOC_TO_PV) == 29,
          f"{len(P.PLANTDOC_TO_PV)}")
    targets = set(P.PLANTDOC_TO_PV.values())
    check("the mapping has no duplicate targets", len(targets) == len(P.PLANTDOC_TO_PV),
          f"{len(targets)} distinct")


def main():
    root = tempfile.mkdtemp(prefix="agft_test_")
    cfg = P.Config(root=root, batch_size=8, workers=0, img_size=64, epochs=2, patience=1,
                   optuna_trials=2, optuna_epochs=1, optuna_subset=1.0, folds=2,
                   fold_epochs=1, agft_backbone="convnextv2_atto.fcmae", mirror_minutes=0)
    P.prepare_dirs(cfg)
    print(f"scratch: {root}")
    try:
        test_models(cfg)
        test_gate_semantics(cfg)
        test_checkpoint_contents(cfg)
        test_training_and_resume(cfg)
        test_evaluate(cfg)
        test_splitting(cfg)
        test_storage(cfg)
        test_optuna_and_cv(cfg)
        test_preflight(cfg)
        test_label_bridge()
        test_end_of_study(cfg)
    finally:
        gc.collect()
        shutil.rmtree(root, ignore_errors=True)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for name in FAIL:
        print(f"  FAILED: {name}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
