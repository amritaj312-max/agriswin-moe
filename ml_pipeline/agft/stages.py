"""The stages of the AGFT study: preflight, tune, cross-validate, train, test.

Split from pipeline.py only to keep each file readable; the notebook runs
them in order. Every stage writes its result to Drive and returns early if
that result is already there, so re-running after a dropped Colab session
continues rather than repeating.
"""

# %% CELL: stage imports
import glob
import os
import shutil
import time

import numpy as np
import torch

from pipeline import (CFG, Diverged, amp_dtype, build_model, copy_atomic, evaluate,
                      load_model_state, loader_for, read_json, train_model, write_json)


def _safe(name):
    return name.replace(":", "_").replace("/", "_")


# %% CELL: preflight


def estimate_hours(speeds, sizes, cfg=CFG):
    """An upper bound on each arm's GPU time, from its measured speed.

    Upper, because it assumes every run uses its full epoch budget; early
    stopping and Optuna's pruning cut the real figure, often by a third.
    Evaluation is a forward pass only, counted at a third of a training step.
    """
    n_train, n_val, n_test, n_field = (sizes[k] for k in ("train", "val", "test", "field"))
    sub_train = min(n_train, max(256, int(n_train * cfg.optuna_subset)))
    sub_val = min(n_val, max(128, int(n_val * cfg.optuna_subset)))
    trained = (cfg.optuna_trials * cfg.optuna_epochs * sub_train
               + cfg.folds * cfg.fold_epochs * n_train * (cfg.folds - 1) / cfg.folds
               + cfg.epochs * n_train)
    scored = (cfg.optuna_trials * cfg.optuna_epochs * sub_val
              + cfg.folds * cfg.fold_epochs * n_train / cfg.folds
              + cfg.epochs * n_val + n_test + 2 * n_field)
    return {arm: (trained + scored / 3) / max(speed, 1e-9) / 3600 for arm, speed in speeds.items()}


def preflight(arms, data, splits, num_classes, n_field, cfg=CFG, log=print):
    """Every arm on the real GPU for a few steps, before hours are committed.

    Builds each model with its pretrained weights -- which also downloads them
    now rather than mid-study -- then runs five optimiser steps at the real
    batch size through the real loader, and one evaluation batch. What only
    shows on the hardware (out of memory, a backbone that rejects the input
    size, mixed precision) shows here in minutes instead of hours into the
    long cell. The measured speeds become the time estimate printed at the end.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    out = os.path.join(cfg.paths["results"], "preflight.json")
    key = {"arms": list(arms), "batch_size": cfg.batch_size, "img_size": cfg.img_size,
           "device": gpu}
    saved = read_json(out)
    if saved is not None and saved.get("key") == key:
        log(f"preflight: passed earlier on {gpu} with these settings -- skipped")
        return saved
    if device != "cuda":
        log("WARNING: no GPU visible. Runtime > Change runtime type > L4 GPU, then Run all.")
    dtype = amp_dtype(device)
    log(f"preflight on {gpu}, mixed precision {str(dtype).replace('torch.', '')}")

    rng = np.random.default_rng(cfg.seed)
    train_idx = splits["train"]
    sample = rng.choice(train_idx, min(len(train_idx), cfg.batch_size * 6), replace=False)
    rows = {}
    for arm in arms:
        try:
            model = build_model(arm, num_classes, cfg).to(device)
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimiser = torch.optim.AdamW(trainable, lr=cfg.lr)
            scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and dtype == torch.float16)
            loader = loader_for(data, sample, cfg, train=True)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            times = []
            model.train()
            for images, y in loader:
                tick = time.time()
                images, y = images.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimiser.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=dtype, enabled=device == "cuda"):
                    loss = torch.nn.functional.cross_entropy(model(images), y)
                if not torch.isfinite(loss):
                    raise RuntimeError("the loss is not finite")
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(trainable, 5.0)
                scaler.step(optimiser)
                scaler.update()
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append(time.time() - tick)
            scored = evaluate(model, loader_for(data, sample[:cfg.batch_size], cfg), device)
            peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
            speed = cfg.batch_size / float(np.median(times[1:] or times))
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(f"preflight: {arm} ran out of GPU memory at batch size "
                               f"{cfg.batch_size}. In the configuration cell set "
                               f"batch_size=32, then Run all.") from exc
        except Exception as exc:
            raise RuntimeError(f"preflight: {arm} failed -- {type(exc).__name__}: {exc}") from exc
        rows[arm] = {"images_per_s": round(speed, 1), "peak_gib": round(peak, 2),
                     "scored": scored["n"]}
        log(f"  OK  {arm:42s} {speed:6.0f} img/s   peak memory {peak:5.1f} GiB")
        del model, optimiser, loader
        if device == "cuda":
            torch.cuda.empty_cache()

    sizes = {"train": len(splits["train"]), "val": len(splits["val"]),
             "test": len(splits["test"]), "field": n_field}
    hours = estimate_hours({a: r["images_per_s"] for a, r in rows.items()}, sizes, cfg)
    log(f"\nevery arm runs on {gpu}. Upper bound on GPU time, if no run stops early:")
    for arm, h in hours.items():
        log(f"  {arm:42s} {h:5.1f} h")
    log(f"  {'total':42s} {sum(hours.values()):5.1f} h")
    result = {"key": key, "arms": rows, "hours_upper_bound": hours}
    write_json(result, out)
    return result


# %% CELL: hyperparameter search


def tune(name, data, train_idx, val_idx, num_classes, cfg=CFG, log=print):
    """Optuna over learning rate, weight decay and dropout.

    Trials run on a subset for a few epochs and are pruned once they fall
    behind: a full-data search for every arm would cost more GPU than the
    whole study. SQLite on a network mount fails with "disk I/O error", so
    the study's database lives on local disk and a copy goes to Drive after
    every trial; after a disconnect the search resumes from that copy with
    its finished trials intact.
    """
    import optuna

    safe = _safe(name)
    out = os.path.join(cfg.paths["results"], f"best_{safe}.json")
    found = read_json(out)
    if found is not None:
        log(f"  {name}: reusing tuned {found}")
        return found

    local_db = os.path.join(cfg.local_root, f"optuna_{safe}.db")
    drive_db = os.path.join(cfg.paths["optuna"], f"{safe}.db")
    if not os.path.isfile(local_db) and os.path.isfile(drive_db):
        shutil.copyfile(drive_db, local_db)
        log(f"  {name}: search restored from Drive")

    def open_study():
        storage = optuna.storages.RDBStorage(f"sqlite:///{local_db}")
        study = optuna.create_study(
            direction="maximize", study_name=f"agft_{name}", storage=storage,
            load_if_exists=True, pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
        return storage, study

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        storage, study = open_study()
    except Exception:                                    # noqa: BLE001 -- torn copy
        log(f"  {name}: saved search unreadable -- starting it again")
        os.remove(local_db)
        storage, study = open_study()
    try:
        return _search(name, study, local_db, drive_db, out, data, train_idx,
                       val_idx, num_classes, safe, cfg, log)
    finally:
        storage.engine.dispose()            # release the file: copyable, removable


def _search(name, study, local_db, drive_db, out, data, train_idx, val_idx,
            num_classes, safe, cfg, log):
    import optuna

    rng = np.random.default_rng(cfg.seed)

    def subset(pool, floor):
        """A slice of `pool` for the search, never larger than the pool itself."""
        want = min(len(pool), max(floor, int(len(pool) * cfg.optuna_subset)))
        return rng.choice(pool, want, replace=False)

    small, small_val = subset(train_idx, 256), subset(val_idx, 128)

    def objective(trial):
        params = {
            "lr": trial.suggest_float("lr", 5e-5, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
            "drop_rate": trial.suggest_float("drop_rate", 0.0, 0.4),
        }
        try:
            best, _ = train_model(name, data, small, small_val, num_classes, cfg,
                                  tag=f"tune_{safe}", epochs=cfg.optuna_epochs, trial=trial,
                                  resumable=False, log=lambda *_a, **_k: None, **params)
        except Diverged:
            raise optuna.TrialPruned() from None
        return best["macro_f1"]

    def keep(study, trial):
        copy_atomic(local_db, drive_db)
        value = f"{trial.value:.4f}" if trial.value is not None else "-"
        log(f"    trial {trial.number}: {trial.state.name.lower():8s} macro-F1 {value}  "
            + "  ".join(f"{k}={v:.2g}" for k, v in trial.params.items()))

    done = len([t for t in study.trials if t.state.name in ("COMPLETE", "PRUNED")])
    remaining = cfg.optuna_trials - done
    if remaining > 0:
        log(f"  {name}: {remaining} trial(s) to run on {len(small)} images")
        study.optimize(objective, n_trials=remaining, callbacks=[keep])
    found = dict(study.best_params)
    log(f"  {name}: best macro-F1 {study.best_value:.4f} with {found}")
    write_json(found, out)
    return found


# %% CELL: cross-validation


def cv_folds(train_idx, labels, groups, cfg=CFG):
    """The k folds as (train, held-out) index arrays into the full dataset.

    Grouped by perceptual hash, for the same reason the split is: the training
    split still holds near-duplicate photographs of one leaf, and folds drawn
    image by image would put copies on both sides and inflate every fold's
    score. Deterministic, so a fold resumed after a disconnect trains on
    exactly the images it started with.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    train_idx = np.asarray(train_idx)
    y = np.array([labels[i] for i in train_idx])
    g = np.array([groups[i] for i in train_idx])
    splitter = StratifiedGroupKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)
    return [(train_idx[a], train_idx[b]) for a, b in splitter.split(train_idx, y, g)]


def cross_validate(name, data, train_idx, labels, groups, num_classes, params, cfg=CFG,
                   log=print):
    """Stratified, grouped k-fold over the training split only (see cv_folds).

    The held-out test split and PlantDoc stay untouched. Each fold is its own
    resumable run.
    """
    safe = _safe(name)
    out = os.path.join(cfg.paths["results"], f"cv_{safe}.json")
    saved = read_json(out)
    if saved is not None:
        log(f"  {name}: reusing CV {saved['mean_macro_f1']:.4f} +- {saved['sd_macro_f1']:.4f}")
        return saved

    folds = cv_folds(train_idx, labels, groups, cfg)
    run = folds[:cfg.max_folds] if cfg.max_folds else folds
    scores = []
    for fold, (a, b) in enumerate(run):
        log(f"  {name}: fold {fold + 1}/{cfg.folds} ({len(a)} train, {len(b)} held out)")
        best, _ = train_model(name, data, a, b, num_classes, cfg,
                              tag=f"cv_{safe}_f{fold}", epochs=cfg.fold_epochs, log=log,
                              **params)
        scores.append(best["macro_f1"])
    if len(run) < len(folds):
        log(f"  {name}: cross-validation stopped after {len(run)} of {len(folds)} folds "
            f"(max_folds={cfg.max_folds})")
    summary = {"folds": scores, "planned_folds": len(folds),
               "mean_macro_f1": float(np.mean(scores)),
               "sd_macro_f1": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0}
    write_json(summary, out)
    log(f"  {name}: CV macro-F1 {summary['mean_macro_f1']:.4f} +- {summary['sd_macro_f1']:.4f}")
    return summary


# %% CELL: final training and the two tests


def run_arm(name, pv, pd_data, splits, labels, groups, classes, reachable, cfg=CFG,
            log=print):
    """Tune, cross-validate, train once on all of train, then test twice.

    PlantDoc is scored two ways: over all 38 outputs, which is what a deployed
    model faces, and restricted to the classes PlantDoc contains, which is the
    closed-set number most papers report. Both are given, because the distance
    between them is large and rarely stated.
    """
    safe = _safe(name)
    out = os.path.join(cfg.paths["results"], f"final_{safe}.json")
    saved = read_json(out)
    if saved is not None:
        log(f"  {name}: already complete")
        return saved

    log(f"\n=== {name} ===")
    params = tune(name, pv, splits["train"], splits["val"], len(classes), cfg, log)
    cv = cross_validate(name, pv, splits["train"], labels, groups, len(classes), params,
                        cfg, log)
    log(f"  {name}: final training on all {len(splits['train'])} training images")
    best, weights = train_model(name, pv, splits["train"], splits["val"], len(classes), cfg,
                                tag=f"final_{safe}", keep_best=True, log=log, **params)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(name, len(classes), cfg, params.get("drop_rate")).to(device)
    load_model_state(model, torch.load(weights, map_location="cpu", weights_only=False)["model"])
    field_idx = np.arange(len(pd_data))
    result = {
        "name": name, "params": params, "cv": cv, "val": best,
        "plantvillage_test": evaluate(model, loader_for(pv, splits["test"], cfg), device),
        "plantdoc": evaluate(model, loader_for(pd_data, field_idx, cfg), device),
        "plantdoc_closed_set": evaluate(model, loader_for(pd_data, field_idx, cfg), device,
                                        restrict=torch.tensor(reachable)),
    }
    write_json(result, out)
    gate = result["plantdoc"].get("gate")
    log(f"  {name}: PlantVillage {result['plantvillage_test']['accuracy']:.4f}  |  "
        f"PlantDoc {result['plantdoc']['accuracy']:.4f} open, "
        f"{result['plantdoc_closed_set']['accuracy']:.4f} closed"
        + (f"  mean gate {gate:.3f}" if gate is not None else ""))
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


# %% CELL: results table


def summarise(cfg=CFG, log=print):
    """Every finished arm, as the table the paper reports."""
    rows = [r for r in (read_json(p) for p in sorted(
        glob.glob(os.path.join(cfg.paths["results"], "final_*.json")))) if r is not None]
    if not rows:
        log("nothing has finished yet")
        return rows
    log(f"\n{'arm':42s}{'CV macro-F1':>18s}{'PV test':>10s}{'PlantDoc':>10s}"
        f"{'closed':>9s}{'gate':>8s}")
    log("-" * 97)
    for r in sorted(rows, key=lambda r: -r["plantdoc"]["accuracy"]):
        gate = r["plantdoc"].get("gate")
        log(f"{r['name']:42s}{r['cv']['mean_macro_f1']:11.4f} "
            f"+-{r['cv']['sd_macro_f1']:.4f}{r['plantvillage_test']['accuracy']:10.4f}"
            f"{r['plantdoc']['accuracy']:10.4f}{r['plantdoc_closed_set']['accuracy']:9.4f}"
            + (f"{gate:8.3f}" if gate is not None else f"{'-':>8s}"))
    write_json(rows, os.path.join(cfg.paths["results"], "summary.json"))
    log("\nPlantVillage is saturated: every arm lands within a point of the others, so")
    log("the column that separates them is PlantDoc, which none of them trained on.")
    latex = latex_rows(rows)
    with open(os.path.join(cfg.paths["results"], "table_main.tex"), "w", encoding="utf-8") as fh:
        fh.write(latex + "\n")
    log("\nLaTeX rows for the paper's main table (also saved as results/table_main.tex):\n")
    log(latex)
    return rows


PAPER_NAMES = {
    "convnextv2_tiny.fcmae_ft_in22k_in1k": "ConvNeXt-V2-T",
    "mobilenetv4_conv_medium.e500_r256_in1k": "MobileNetV4-M",
    "swinv2_tiny_window8_256.ms_in1k": "Swin-V2-T",
    "agft:ft": "AGFT, $g{=}1$ (fine-tune)",
    "agft:frozen": "AGFT, $g{=}0$ (linear probe)",
    "agft:fixed": "AGFT, fixed $\\alpha$",
    "agft:gated": "AGFT, learned gate (ours)",
}


def latex_rows(rows):
    """One booktabs row per arm, in the paper's order: CV macro-F1, then
    accuracy and macro-F1 on the PlantVillage test split and on PlantDoc, the
    closed-set PlantDoc accuracy, and the mean gate on PlantDoc."""
    def pct(x):
        return f"{100 * x:.2f}"

    lines = []
    for name in PAPER_NAMES:
        r = next((r for r in rows if r["name"] == name), None)
        if r is None:
            continue
        gate = r["plantdoc"].get("gate")
        ran = len(r["cv"].get("folds", []))
        capped = 0 < ran < r["cv"].get("planned_folds", ran)
        lines.append(" & ".join([
            PAPER_NAMES[name],
            f"{pct(r['cv']['mean_macro_f1'])} $\\pm$ {pct(r['cv']['sd_macro_f1'])}"
            + ("$^{\\dagger}$" if capped else ""),
            pct(r["plantvillage_test"]["accuracy"]), pct(r["plantvillage_test"]["macro_f1"]),
            pct(r["plantdoc"]["accuracy"]), pct(r["plantdoc"]["macro_f1"]),
            pct(r["plantdoc_closed_set"]["accuracy"]),
            f"{gate:.3f}" if gate is not None else "--",
        ]) + " \\\\")
    return "\n".join(lines)


# %% CELL: what the gate does


def _branch_outputs(model, loader, device):
    """Per image: the fine-tuned branch's logits, the frozen branch's, and the
    gate, read straight off a trained AGFT model."""
    dtype = amp_dtype(device)
    tuned, frozen, gates, labels = [], [], [], []
    model.eval()
    with torch.no_grad():
        for images, y in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=dtype, enabled=device == "cuda"):
                f_frozen = model.frozen(images)
                logits_frozen = model.head_frozen(f_frozen)
                logits_tuned = model.head_tuned(model.tuned(images))
                gate = model.gate_value(f_frozen)
            tuned.append(logits_tuned.float().cpu())
            frozen.append(logits_frozen.float().cpu())
            gates.append(gate.float().cpu().reshape(-1))
            labels.append(y)
    return torch.cat(tuned), torch.cat(frozen), torch.cat(gates), torch.cat(labels)


def _accuracy(logits, labels, restrict=None):
    if restrict is not None:
        masked = torch.full_like(logits, float("-inf"))
        masked[:, restrict] = logits[:, restrict]
        logits = masked
    return float((logits.argmax(1) == labels).float().mean())


def _gate_shift(lab, field):
    """Mann-Whitney U between the gate's values on field and lab images, with
    the rank-biserial correlation as effect size (negative: lower on field)."""
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:                                  # scipy ships with sklearn
        return None
    u, p = mannwhitneyu(field.numpy(), lab.numpy(), alternative="two-sided")
    return {"u": float(u), "p_value": float(p),
            "rank_biserial": float(2.0 * u / (len(field) * len(lab)) - 1.0)}


def gate_analysis(pv, pd_data, splits, classes, reachable, cfg=CFG, log=print):
    """What the learned gate does, measured on the trained AGFT-gated model.

    One inference pass over the PlantVillage test split and PlantDoc records,
    per image, both branches' logits and the gate. From those alone:

      * each branch's accuracy on its own, and the learned mixture's;
      * every fixed trade-off alpha in [0, 1] -- the WiSE-FT-style family the
        gate is meant to beat -- mixed in logit space as the model was trained,
        and in probability space as a post-hoc check; the best alpha chosen
        with hindsight on PlantDoc is an oracle no fixed method could pick;
      * whether the gate routes: its distribution on lab versus field images,
        and its value where only the frozen or only the fine-tuned branch is
        right;
      * each branch's logit scale, which decides how a logit-space mixture
        actually weighs the two whatever the gate says.

    No training: about a minute on an L4. Needs the gated arm to have finished.
    """
    out = os.path.join(cfg.paths["results"], "gate_analysis.json")
    figure = os.path.join(cfg.paths["figures"], "gate_analysis.png")
    saved = read_json(out)
    if saved is not None and os.path.isfile(figure):
        log("gate analysis: done earlier -- reusing it")
        return saved
    weights = os.path.join(cfg.paths["models"], "final_agft_gated.pt")
    finished = read_json(os.path.join(cfg.paths["results"], "final_agft_gated.json"))
    if finished is None or not os.path.isfile(weights):
        log("gate analysis: the gated arm has not finished yet -- run cell 10 to completion")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    params = read_json(os.path.join(cfg.paths["results"], "best_agft_gated.json")) or {}
    model = build_model("agft:gated", len(classes), cfg, params.get("drop_rate")).to(device)
    load_model_state(model, torch.load(weights, map_location="cpu", weights_only=False)["model"])
    raw = {"plantvillage_test": _branch_outputs(model, loader_for(pv, splits["test"], cfg), device),
           "plantdoc": _branch_outputs(model, loader_for(pd_data, np.arange(len(pd_data)), cfg),
                                       device)}
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    alphas = [round(float(a), 2) for a in np.linspace(0.0, 1.0, 21)]
    restrict = torch.tensor(reachable)
    report = {"alphas": alphas}
    for name, (lt, lf, g, y) in raw.items():
        pt, pf = lt.softmax(1), lf.softmax(1)
        mixed = g[:, None] * lt + (1 - g[:, None]) * lf
        tuned_right, frozen_right = lt.argmax(1) == y, lf.argmax(1) == y
        only_frozen, only_tuned = frozen_right & ~tuned_right, tuned_right & ~frozen_right
        part = {
            "n": int(len(y)),
            "tuned_only": _accuracy(lt, y),
            "frozen_only": _accuracy(lf, y),
            "gated": _accuracy(mixed, y),
            "gated_probability_space": _accuracy(g[:, None] * pt + (1 - g[:, None]) * pf, y),
            "either_branch_right": float((tuned_right | frozen_right).float().mean()),
            "gate_mean": float(g.mean()),
            "gate_quantiles": {str(q): float(torch.quantile(g, q))
                               for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
            "gate_where_only_frozen_right":
                float(g[only_frozen].mean()) if bool(only_frozen.any()) else None,
            "gate_where_only_tuned_right":
                float(g[only_tuned].mean()) if bool(only_tuned.any()) else None,
            "count_only_frozen_right": int(only_frozen.sum()),
            "count_only_tuned_right": int(only_tuned.sum()),
            "logit_scale_tuned": float(lt.std(1).mean()),
            "logit_scale_frozen": float(lf.std(1).mean()),
            "fixed_alpha_logit": [_accuracy(a * lt + (1 - a) * lf, y) for a in alphas],
            "fixed_alpha_probability": [_accuracy(a * pt + (1 - a) * pf, y) for a in alphas],
        }
        if name == "plantdoc":
            part["closed_set"] = {
                "tuned_only": _accuracy(lt, y, restrict),
                "frozen_only": _accuracy(lf, y, restrict),
                "gated": _accuracy(mixed, y, restrict),
                "fixed_alpha_logit": [_accuracy(a * lt + (1 - a) * lf, y, restrict)
                                      for a in alphas],
            }
        report[name] = part
    field = report["plantdoc"]
    best = int(np.argmax(field["fixed_alpha_logit"]))
    report["oracle_fixed_alpha_on_plantdoc"] = {"alpha": alphas[best],
                                                "accuracy": field["fixed_alpha_logit"][best]}
    report["gate_shift_lab_to_field"] = _gate_shift(raw["plantvillage_test"][2],
                                                    raw["plantdoc"][2])
    write_json(report, out)
    _plot_gate_analysis(report, raw, figure)

    lab = report["plantvillage_test"]
    log(f"gate analysis on {lab['n']} lab and {field['n']} field images")
    log(f"  {'':30s}{'PV test':>9s}{'PlantDoc':>10s}")
    for key, label in (("tuned_only", "fine-tuned branch alone"),
                       ("frozen_only", "frozen branch alone"),
                       ("gated", "learned per-image gate")):
        log(f"  {label:30s}{lab[key]:9.4f}{field[key]:10.4f}")
    oracle = report["oracle_fixed_alpha_on_plantdoc"]
    log(f"  {'best fixed alpha (oracle)':30s}{'':9s}{oracle['accuracy']:10.4f}"
        f"   alpha={oracle['alpha']}, chosen with hindsight on PlantDoc")
    log(f"  mean gate: lab {lab['gate_mean']:.3f}, field {field['gate_mean']:.3f}")
    shift = report["gate_shift_lab_to_field"]
    if shift:
        log(f"  lab vs field gate: Mann-Whitney p={shift['p_value']:.2e}, "
            f"rank-biserial {shift['rank_biserial']:+.3f}")
    if field["gate_where_only_frozen_right"] is not None and \
            field["gate_where_only_tuned_right"] is not None:
        log(f"  on PlantDoc the gate averages {field['gate_where_only_frozen_right']:.3f} where "
            f"only the frozen branch is right ({field['count_only_frozen_right']} images), and "
            f"{field['gate_where_only_tuned_right']:.3f} where only the fine-tuned one is "
            f"({field['count_only_tuned_right']})")
    log(f"  logit scale, fine-tuned vs frozen: {field['logit_scale_tuned']:.2f} vs "
        f"{field['logit_scale_frozen']:.2f} on PlantDoc")
    latex = (
        f"Fine-tuned branch alone & {100 * lab['tuned_only']:.2f} & {100 * field['tuned_only']:.2f} \\\\\n"
        f"Frozen branch alone & {100 * lab['frozen_only']:.2f} & {100 * field['frozen_only']:.2f} \\\\\n"
        f"Learned per-image gate & {100 * lab['gated']:.2f} & {100 * field['gated']:.2f} \\\\\n"
        f"Best fixed $\\alpha$ (oracle, $\\alpha{{=}}{oracle['alpha']}$) & -- & "
        f"{100 * oracle['accuracy']:.2f} \\\\\n"
        f"Either branch right (routing bound) & {100 * lab['either_branch_right']:.2f} & "
        f"{100 * field['either_branch_right']:.2f} \\\\")
    with open(os.path.join(cfg.paths["results"], "table_gate.tex"), "w", encoding="utf-8") as fh:
        fh.write(latex + "\n")
    log("\nLaTeX rows for the paper's gate table (also saved as results/table_gate.tex):\n")
    log(latex)
    return report


def _plot_gate_analysis(report, raw, dest):
    """Three panels: the gate on lab versus field images; accuracy against a
    fixed alpha with the learned gate marked; the gate by which branch is right."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(16, 4.4))
    axes = fig.subplots(1, 3)
    lab_gate, field_gate = raw["plantvillage_test"][2].numpy(), raw["plantdoc"][2].numpy()
    bins = np.linspace(0, 1, 41)
    axes[0].hist(lab_gate, bins=bins, density=True, alpha=0.6, label="PlantVillage test (lab)")
    axes[0].hist(field_gate, bins=bins, density=True, alpha=0.6, label="PlantDoc (field)")
    axes[0].set_xlabel("gate g(x): weight on the fine-tuned branch")
    axes[0].set_ylabel("density")
    axes[0].set_title("Where the gate sends each image")
    axes[0].legend()

    alphas, field = report["alphas"], report["plantdoc"]
    axes[1].plot(alphas, field["fixed_alpha_logit"], "o-", ms=3, label="fixed alpha, logit space")
    axes[1].plot(alphas, field["fixed_alpha_probability"], "s--", ms=3,
                 label="fixed alpha, probability space")
    axes[1].axhline(field["gated"], color="crimson", lw=1.5, label="learned per-image gate")
    axes[1].set_xlabel("fixed alpha (0 = frozen branch, 1 = fine-tuned branch)")
    axes[1].set_ylabel("PlantDoc accuracy")
    axes[1].set_title("Per-image gate against every fixed trade-off")
    axes[1].legend(fontsize=8)

    lt, lf, g, y = raw["plantdoc"]
    tuned_right, frozen_right = lt.argmax(1) == y, lf.argmax(1) == y
    groups = [("both right", tuned_right & frozen_right),
              ("only frozen", frozen_right & ~tuned_right),
              ("only fine-tuned", tuned_right & ~frozen_right),
              ("both wrong", ~tuned_right & ~frozen_right)]
    names = [f"{n}\n(n={int(m.sum())})" for n, m in groups]
    means = [float(g[m].mean()) if bool(m.any()) else 0.0 for _, m in groups]
    axes[2].bar(range(len(groups)), means, color=["#55a868", "#4c72b0", "#dd8452", "#999999"])
    axes[2].set_xticks(range(len(groups)))
    axes[2].set_xticklabels(names, fontsize=8)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("mean gate on PlantDoc")
    axes[2].set_title("Does the gate favour the branch that is right?")
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
