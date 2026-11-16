"""The four stages of the AGFT study: tune, cross-validate, train, test.

Split from pipeline.py only to keep each file readable; the notebook runs
them in order. Every stage writes its result to Drive and returns early if
that result is already there, so re-running after a dropped Colab session
continues rather than repeating.
"""

# %% CELL: stage imports
import glob
import json
import os

import numpy as np
import torch

from pipeline import CFG, build_model, evaluate, loader_for, train_model


# %% CELL: hyperparameter search


def tune(name, data, train_idx, val_idx, num_classes, cfg=CFG, log=print):
    """Optuna over learning rate, weight decay and dropout.

    The study lives in a SQLite file on Drive, so an interrupted search
    resumes with its finished trials intact rather than starting over. Trials
    run on a subset for a few epochs and are pruned once they fall behind: a
    full-data search for every arm would cost more GPU than the whole study.
    """
    import optuna

    out = os.path.join(cfg.paths["results"], f"best_{name.replace(':', '_')}.json")
    if os.path.isfile(out):
        with open(out, encoding="utf-8") as fh:
            found = json.load(fh)
        log(f"  {name}: reusing tuned {found}")
        return found

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
        best, _ = train_model(name, data, small, small_val, num_classes, cfg,
                              tag=f"tune_{name.replace(':', '_')}_t{trial.number}",
                              epochs=cfg.optuna_epochs, trial=trial,
                              log=lambda *_a, **_k: None, **params)
        return best["macro_f1"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", study_name=f"agft_{name}",
        storage=f"sqlite:///{os.path.join(cfg.root, 'optuna.db')}", load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    done = len([t for t in study.trials if t.state.name in ("COMPLETE", "PRUNED")])
    remaining = cfg.optuna_trials - done
    if remaining > 0:
        log(f"  {name}: {remaining} trial(s) to run on {len(small)} images")
        study.optimize(objective, n_trials=remaining)
    found = dict(study.best_params)
    log(f"  {name}: best macro-F1 {study.best_value:.4f} with {found}")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(found, fh, indent=2)
    return found


# %% CELL: cross-validation


def cross_validate(name, data, train_idx, labels, num_classes, params, cfg=CFG, log=print):
    """Stratified k-fold over the training split only.

    The folds are drawn from the training indices, so the held-out test split
    and PlantDoc stay untouched. Each fold writes its own checkpoint and is
    skipped on a re-run, so a disconnect costs one fold at most.
    """
    from sklearn.model_selection import StratifiedKFold

    out = os.path.join(cfg.paths["results"], f"cv_{name.replace(':', '_')}.json")
    if os.path.isfile(out):
        with open(out, encoding="utf-8") as fh:
            saved = json.load(fh)
        log(f"  {name}: reusing CV {saved['mean_macro_f1']:.4f} "
            f"+- {saved['sd_macro_f1']:.4f}")
        return saved

    y = np.array([labels[i] for i in train_idx])
    splitter = StratifiedKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)
    scores = []
    for fold, (a, b) in enumerate(splitter.split(train_idx, y)):
        log(f"  {name}: fold {fold + 1}/{cfg.folds}")
        best, _ = train_model(name, data, train_idx[a], train_idx[b], num_classes, cfg,
                              tag=f"cv_{name.replace(':', '_')}_f{fold}",
                              epochs=cfg.fold_epochs, log=log, **params)
        scores.append(best["macro_f1"])
    summary = {"folds": scores, "mean_macro_f1": float(np.mean(scores)),
               "sd_macro_f1": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    log(f"  {name}: CV macro-F1 {summary['mean_macro_f1']:.4f} "
        f"+- {summary['sd_macro_f1']:.4f}")
    return summary


# %% CELL: final training and the two tests


def run_arm(name, pv, pd_data, splits, labels, classes, reachable, cfg=CFG, log=print):
    """Tune, cross-validate, train once on all of train, then test twice.

    PlantDoc is scored two ways: over all 38 outputs, which is what a deployed
    model faces, and restricted to the 28 classes PlantDoc contains, which is
    the closed-set number most papers report. Both are given, because the
    distance between them is large and rarely stated.
    """
    out = os.path.join(cfg.paths["results"], f"final_{name.replace(':', '_')}.json")
    if os.path.isfile(out):
        with open(out, encoding="utf-8") as fh:
            saved = json.load(fh)
        log(f"  {name}: already complete")
        return saved

    log(f"\n=== {name} ===")
    params = tune(name, pv, splits["train"], splits["val"], len(classes), cfg, log)
    cv = cross_validate(name, pv, splits["train"], labels, len(classes), params, cfg, log)
    log(f"  {name}: final training on all {len(splits['train'])} training images")
    best, best_path = train_model(name, pv, splits["train"], splits["val"], len(classes),
                                  cfg, tag=f"final_{name.replace(':', '_')}", log=log,
                                  **params)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(name, len(classes), cfg, params.get("drop_rate")).to(device)
    model.load_state_dict(torch.load(best_path, map_location="cpu",
                                     weights_only=False)["model"])
    field_idx = np.arange(len(pd_data))
    result = {
        "name": name, "params": params, "cv": cv, "val": best,
        "plantvillage_test": evaluate(model, loader_for(pv, splits["test"], cfg), device),
        "plantdoc": evaluate(model, loader_for(pd_data, field_idx, cfg), device),
        "plantdoc_closed_set": evaluate(model, loader_for(pd_data, field_idx, cfg), device,
                                        restrict=torch.tensor(reachable)),
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    gate = result["plantdoc"].get("gate")
    log(f"  {name}: PlantVillage {result['plantvillage_test']['accuracy']:.4f}  |  "
        f"PlantDoc {result['plantdoc']['accuracy']:.4f} open, "
        f"{result['plantdoc_closed_set']['accuracy']:.4f} closed"
        + (f"  mean gate {gate:.3f}" if gate is not None else ""))
    del model
    torch.cuda.empty_cache()
    return result


# %% CELL: results table


def summarise(cfg=CFG, log=print):
    """Every finished arm, as the table the paper reports."""
    rows = []
    for path in sorted(glob.glob(os.path.join(cfg.paths["results"], "final_*.json"))):
        with open(path, encoding="utf-8") as fh:
            rows.append(json.load(fh))
    if not rows:
        log("nothing has finished yet")
        return rows
    log(f"\n{'arm':36s}{'CV macro-F1':>18s}{'PV test':>10s}{'PlantDoc':>10s}"
        f"{'closed':>9s}{'gate':>8s}")
    log("-" * 91)
    for r in sorted(rows, key=lambda r: -r["plantdoc"]["accuracy"]):
        gate = r["plantdoc"].get("gate")
        log(f"{r['name']:36s}{r['cv']['mean_macro_f1']:11.4f} "
            f"+-{r['cv']['sd_macro_f1']:.4f}{r['plantvillage_test']['accuracy']:10.4f}"
            f"{r['plantdoc']['accuracy']:10.4f}{r['plantdoc_closed_set']['accuracy']:9.4f}"
            + (f"{gate:8.3f}" if gate is not None else f"{'-':>8s}"))
    with open(os.path.join(cfg.paths["results"], "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    log("\nPlantVillage is saturated: every arm lands within a point of the others, so")
    log("the column that separates them is PlantDoc, which none of them trained on.")
    return rows
