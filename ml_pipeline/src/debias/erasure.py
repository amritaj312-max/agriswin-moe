"""Cross-class session erasure: learn a photo session's signature where it can
be separated from the class, then remove it everywhere.

The problem this exists for
---------------------------
In PlantVillage, 31 of 38 classes were photographed in a single session, so
"what this disease looks like" and "what that session looked like" are one
signal in the training data. A model trained on it reads the session, and in
the field there is no session to read. The usual cures -- GroupDRO, DANN,
IRM, DFR -- all need most class-by-session cells to hold images. Here most
are empty, so asking a model to ignore the session asks it to ignore the
class.

The idea
--------
Seven classes *were* photographed by two or more sessions. Inside those, and
only inside those, session and class vary independently, so the difference
between two session means of the same class is a session effect with the
class subtracted out. Pool those differences, take their leading directions,
and that is a session signature estimated without ever seeing a class label
move. Remove it from all 38 classes.

The estimate is class-conditional, which is what makes it safe: the marginal
alternative -- the directions separating session means over the whole dataset,
ignoring class -- is almost the class directions themselves, and erasing it
erases the labels. That alternative is kept as the `marginal` baseline
precisely so the difference can be shown rather than asserted.

How far the data lets this go
-----------------------------
Six of the seven classes hold two sessions and one holds three, so the
pooled scatter has rank at most eight. Eight directions out of 768 is the
entire budget this dataset supports, and no choice of method changes that.

Prior work this is built from, and must cite
--------------------------------------------
Projecting out a nuisance subspace estimated from within-class, across-
condition variation is Nuisance Attribute Projection, standard in speaker
verification since the mid-2000s (Solomonoff et al. 2005; Campbell et al.).
The linear-erasure lineage is INLP (Ravfogel 2020) and LEACE (Belrose 2023);
refitting only the last layer is DFR (Kirichenko 2023); conditioning the
invariance on the class is CIDDG (Li 2018).

What is new here is not the projection. It is that the nuisance label was
recovered from filenames nobody had used, that the signature is estimated in
the seven classes where it is identifiable and then applied to the thirty-one
where it is not, and that the erasure is checked against a session that spans
fifteen classes to show the class signal survived it.

Usage:
    python ml_pipeline/src/debias/erasure.py              # the whole experiment
    python ml_pipeline/src/debias/erasure.py --k 4        # one fixed subspace size
"""

import argparse
import collections
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
for sub in ("data", "evaluation"):
    sys.path.insert(0, os.path.join(SRC, sub))

import paths  # noqa: E402
import sources  # noqa: E402
from boxed_leaves import measures, within_crop_balanced  # noqa: E402
from features import CACHE_DIR  # noqa: E402

NUM_CLASSES = 38
RESULT_FILE = os.path.join(paths.REPO_ROOT, "results", "erasure.json")
CHECK_SESSION = "RS"          # the session that spans 15 classes over 7 crops
HEAD_EPOCHS = 40
HEAD_LR = 1e-3


def load(view):
    return torch.load(os.path.join(CACHE_DIR, f"{view}.pt"), map_location="cpu",
                      weights_only=False)


def session_scatter(features, class_names, session_names, conditional=True):
    """Between-session scatter, and the rank the data can support.

    Conditional (the method): sessions are compared only with other sessions
    of the *same* class, so a class difference can never enter the estimate.
    Marginal (the baseline): session means are compared with the global mean,
    which in this dataset is nearly the same thing as comparing classes.
    """
    groups = collections.defaultdict(list)
    for index, (cls, session) in enumerate(zip(class_names, session_names)):
        groups[(cls, session) if conditional else (None, session)].append(index)

    by_class = collections.defaultdict(list)
    for (cls, session), rows in groups.items():
        if len(rows) >= sources.MIN_IMAGES:
            by_class[cls].append((session, torch.tensor(rows)))

    scatter = torch.zeros(features.shape[1], features.shape[1], dtype=torch.float64)
    rank, used = 0, []
    for cls, members in by_class.items():
        if conditional and len(members) < 2:
            continue                      # one session: nothing to compare it with
        pooled = torch.cat([rows for _, rows in members])
        centre = features[pooled].mean(0).double()
        for session, rows in members:
            delta = (features[rows].mean(0).double() - centre).unsqueeze(1)
            scatter += len(rows) * delta @ delta.T
        rank += len(members) - 1
        used.append((cls, [s for s, _ in members]))
    return scatter, rank, used


def subspace(scatter, k):
    """The k leading directions of a scatter matrix, as orthonormal rows."""
    values, vectors = torch.linalg.eigh(scatter)
    return vectors[:, torch.argsort(values, descending=True)[:k]].T.float()


def erase(features, basis):
    """Features with the subspace removed: f - U(U f)."""
    if basis is None or basis.shape[0] == 0:
        return features
    return features - (features @ basis.T) @ basis


def fit_head(features, labels, classes=NUM_CLASSES, seed=0, log=None):
    """A linear probe on frozen features, standardised, trained on the GPU."""
    torch.manual_seed(seed)
    mean, std = features.mean(0), features.std(0).clamp_min(1e-6)
    x = ((features - mean) / std).cuda()
    y = labels.cuda()
    head = torch.nn.Linear(features.shape[1], classes).cuda()
    optimiser = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, HEAD_EPOCHS)
    for _ in range(HEAD_EPOCHS):
        order = torch.randperm(len(y), device=y.device)
        for start in range(0, len(y), 1024):
            rows = order[start:start + 1024]
            loss = torch.nn.functional.cross_entropy(head(x[rows]), y[rows])
            optimiser.zero_grad(); loss.backward(); optimiser.step()
        schedule.step()
    if log:
        log(f"        head trained, final loss {loss.item():.4f}")
    return head, mean, std


@torch.no_grad()
def apply_head(head, mean, std, features):
    return head(((features - mean) / std).cuda()).cpu()


def check_class_signal(features, class_names, session_names, basis, seed=0):
    """Does the class survive the erasure where the session is held fixed?

    Inside one session that photographed fifteen classes, every image shares
    a session, so any class signal there is class signal. If erasing the
    subspace costs accuracy here, it took the label with it.
    """
    rows = torch.tensor([i for i, s in enumerate(session_names) if s == CHECK_SESSION])
    if len(rows) < 200:
        return None
    names = [class_names[i] for i in rows.tolist()]
    index = {name: i for i, name in enumerate(sorted(set(names)))}
    y = torch.tensor([index[n] for n in names])
    split = int(len(rows) * 0.7)
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed))
    train, test = order[:split], order[split:]
    x = erase(features[rows], basis)
    head, mean, std = fit_head(x[train], y[train], classes=len(index), seed=seed)
    predicted = apply_head(head, mean, std, x[test]).argmax(1)
    return (predicted == y[test]).float().mean().item()


def evaluate(features, labels, basis, fields, seeds, log):
    """Fit the head once per seed and score every field view.

    The head is refitted rather than reused because its own randomness is the
    floor any claimed gain has to clear, and on these features that floor is
    a few tenths of a point -- small enough that a null here is a real bound
    rather than a failure to measure.
    """
    x_train = erase(features, basis)
    runs = {view: [] for view in fields}
    for seed in seeds:
        head, mean, std = fit_head(x_train, labels, seed=seed)
        for view, field in fields.items():
            logits = apply_head(head, mean, std, erase(field["features"], basis))
            entry = measures(logits.argmax(1), field["labels"])
            entry["within_crop_balanced"], _ = within_crop_balanced(logits, field["labels"])
            runs[view].append(entry)
    summary = {}
    for view, entries in runs.items():
        summary[view] = {key: {"mean": mean_of(entries, key), "sd": sd_of(entries, key),
                               "seeds": [e[key] for e in entries]}
                         for key in entries[0]}
        stat = summary[view]
        log(f"      {view:14s} acc {stat['accuracy']['mean']:6.2%} +-{stat['accuracy']['sd']:5.2%}"
            f"   crop {stat['crop_accuracy']['mean']:6.2%}"
            f"   within-crop {stat['within_crop_balanced']['mean']:6.2%}"
            f"   sink {stat['sink_rate']['mean']:5.2%}")
    return summary


def mean_of(entries, key):
    return sum(e[key] for e in entries) / len(entries)


def sd_of(entries, key):
    values = [e[key] for e in entries]
    if len(values) < 2:
        return 0.0
    centre = sum(values) / len(values)
    return (sum((v - centre) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def run(ks, log, train_view="leaf", seeds=(0, 1, 2, 3, 4)):
    train = load(train_view)
    log(f"  training view: {train_view}")
    fields = {"plantdoc": load("plantdoc"), "plantdoc_box": load("plantdoc_box")}
    rows_train = [i for i, s in enumerate(train["split"]) if s == "train"]
    rows_train = torch.tensor(rows_train)
    features = train["features"]
    class_names, session_names = train["class_name"], train["source"]

    conditional, rank, used = session_scatter(features, class_names, session_names, True)
    marginal, _, _ = session_scatter(features, class_names, session_names, False)
    log(f"  session signature estimated in {len(used)} classes; the data supports "
        f"at most rank {rank}")
    for cls, members in sorted(used):
        log(f"    {cls.replace('___', ' / '):52s} {', '.join(members)}")

    baseline = check_class_signal(features, class_names, session_names, None)
    log(f"\n  class signal inside session {CHECK_SESSION}, before erasing: {baseline:.1%}")

    results = {"rank_supported": rank, "classes_used": used, "seeds": seeds,
               "check_before": baseline, "settings": {}}
    log("\n  no erasure (the frozen-probe baseline this has to beat)")
    results["settings"]["none"] = {
        "check_after": baseline,
        "fields": evaluate(features[rows_train], train["labels"][rows_train],
                           None, fields, seeds, log)}

    for name, scatter in (("conditional", conditional), ("marginal", marginal)):
        for k in ks:
            if name == "conditional" and k > rank:
                continue
            basis = subspace(scatter, k)
            kept = check_class_signal(features, class_names, session_names, basis)
            log(f"\n  {name}, k={k}: class signal inside {CHECK_SESSION} "
                f"{kept:.1%} ({kept - baseline:+.1%})")
            results["settings"][f"{name}_k{k}"] = {
                "check_after": kept,
                "fields": evaluate(features[rows_train], train["labels"][rows_train],
                                   basis, fields, seeds, log)}

    log("\n  against the baseline, in units of the larger seed spread:")
    base = results["settings"]["none"]["fields"]
    for name, entry in results["settings"].items():
        if name == "none":
            continue
        for view in fields:
            gap = entry["fields"][view]["accuracy"]["mean"] - base[view]["accuracy"]["mean"]
            floor = max(entry["fields"][view]["accuracy"]["sd"], base[view]["accuracy"]["sd"])
            ratio = abs(gap) / floor if floor else float("inf")
            log(f"    {name:16s} {view:14s} {gap * 100:+6.2f} pp   {ratio:4.1f}x")

    with open(RESULT_FILE, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    log(f"\n  saved {os.path.relpath(RESULT_FILE, paths.REPO_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--k", nargs="+", type=int, default=[1, 2, 4, 8],
                        help="subspace sizes to try")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                        help="head seeds; their spread is the floor a gain must clear")
    parser.add_argument("--train-view", default="leaf", choices=["leaf", "colour"],
                        help="'leaf' is the method; 'colour' leaves the session in the "
                             "background, where it is trivially readable")
    args = parser.parse_args()
    run(args.k, lambda message: print(message, flush=True), args.train_view, args.seeds)


if __name__ == "__main__":
    main()
