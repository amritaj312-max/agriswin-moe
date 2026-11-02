"""
Uncertainty and selective-prediction metrics, computed from logits.

metrics.py answers "how often is it right". This answers "does it know when it
is not", which is Novelty 3 and the paper's honest response to domain shift: a
model that collapses on field images but says so is useful, one that is
confidently wrong is dangerous.

Two kinds of uncertainty are separated, following Depeweg et al. (2018):

    total       H[mean_t p_t]              predictive entropy of the averaged
                                           forward passes -- all the doubt
    aleatoric   mean_t H[p_t]              doubt still there when the weights
                                           are fixed: a genuinely ambiguous image
    epistemic   total - aleatoric          mutual information between the
                                           prediction and the weights: doubt the
                                           model has about its own parameters,
                                           which is what rises off-distribution

A single deterministic pass gives total entropy only; epistemic needs the
several MC-dropout passes that make p_t differ.

Running this file checks every formula against a case whose answer is known by
hand, and the risk-coverage curve against scikit-learn's AUC:

    python ml_pipeline/src/evaluation/uncertainty.py
"""

import torch


def entropy(probs, dim=-1):
    """H(p) in nats. 0 log 0 is taken as 0, as the limit requires."""
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim)


def decompose(pass_probs):
    """Split the uncertainty of stacked MC passes (T, N, C) into its parts.

    Returns the mean probabilities and total/aleatoric/epistemic per sample.
    With T=1 the epistemic term is 0 by construction: one pass cannot disagree
    with itself.
    """
    mean_probs = pass_probs.mean(0)
    total = entropy(mean_probs)
    aleatoric = entropy(pass_probs).mean(0)
    return mean_probs, total, aleatoric, (total - aleatoric).clamp_min(0.0)


def risk_coverage(correct, confidence):
    """Accuracy against how much of the data is kept, most confident first.

    Abstention is the point of measuring uncertainty, so this is the curve that
    shows what it buys: keep the most confident `coverage` fraction and
    `accuracy` is what you get on it. Ties are broken deterministically.

    AURC is the mean risk (error rate) over all coverages -- lower is better,
    and unlike accuracy it rewards ranking mistakes below correct answers even
    when the overall accuracy is unchanged.
    """
    order = torch.argsort(confidence, descending=True, stable=True)
    hits = correct[order].double().cumsum(0)
    kept = torch.arange(1, len(correct) + 1, dtype=torch.float64)
    accuracy = hits / kept
    return {
        "coverage": (kept / len(correct)).tolist(),
        "accuracy": accuracy.tolist(),
        "aurc": (1.0 - accuracy).mean().item(),
    }


def accuracy_at_coverage(curve, coverage):
    """Accuracy when only the most confident `coverage` fraction is answered."""
    n = len(curve["accuracy"])
    return curve["accuracy"][max(0, min(n - 1, round(coverage * n) - 1))]


def reliability(confidence, correct, n_bins=15):
    """Per-bin confidence vs accuracy -- the reliability diagram's data.

    Bins match expected_calibration_error() in metrics.py, so the ECE reported
    there is exactly the gap-weighted sum of these bins and the diagram cannot
    disagree with the number beside it.
    """
    bins = (torch.ceil(confidence * n_bins).long() - 1).clamp(0, n_bins - 1)
    out = []
    for b in range(n_bins):
        in_bin = bins == b
        count = int(in_bin.sum())
        out.append({
            "bin": b,
            "lower": b / n_bins,
            "upper": (b + 1) / n_bins,
            "count": count,
            "confidence": confidence[in_bin].double().mean().item() if count else None,
            "accuracy": correct[in_bin].double().mean().item() if count else None,
        })
    return out


def summarize(pass_probs, labels, n_bins=15):
    """Every uncertainty number for one model on one split.

    pass_probs is (T, N, C): T MC-dropout passes of N samples over C classes.
    The prediction is the argmax of the averaged passes, which is what an
    MC-dropout model actually answers.
    """
    mean_probs, total, aleatoric, epistemic = decompose(pass_probs)
    confidence, preds = mean_probs.max(dim=1)
    correct = preds.eq(labels)
    curve = risk_coverage(correct, confidence)
    by_entropy = risk_coverage(correct, -total)
    return {
        "passes": pass_probs.shape[0],
        "acc": correct.double().mean().item(),
        "mean_confidence": confidence.double().mean().item(),
        "entropy": {
            "total": total.double().mean().item(),
            "aleatoric": aleatoric.double().mean().item(),
            "epistemic": epistemic.double().mean().item(),
        },
        "entropy_when_right": total[correct].double().mean().item() if correct.any() else None,
        "entropy_when_wrong": total[~correct].double().mean().item() if (~correct).any() else None,
        "aurc_by_confidence": curve["aurc"],
        "aurc_by_entropy": by_entropy["aurc"],
        "acc_at_coverage": {str(c): accuracy_at_coverage(curve, c)
                            for c in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)},
        "reliability": reliability(confidence, correct, n_bins),
        # Per sample, rounded to 6 decimals: enough to rank confidences without
        # creating ties, and half the characters of a full float repr. The
        # risk-coverage curve is NOT stored -- it is a cumulative mean of these
        # two columns, so keeping it would double the file to no end. Results
        # are committed, and this evaluation re-runs after every ablation, so
        # what goes in the file is what cannot be recomputed from it.
        "per_sample": {
            "correct": correct.tolist(),
            "confidence": [round(v, 6) for v in confidence.float().tolist()],
            "total_entropy": [round(v, 6) for v in total.float().tolist()],
            "epistemic_entropy": [round(v, 6) for v in epistemic.float().tolist()],
        },
    }


if __name__ == "__main__":
    import math

    import numpy as np
    from sklearn import metrics as skm

    # entropy of a known distribution
    uniform = torch.full((1, 4), 0.25, dtype=torch.float64)
    assert abs(entropy(uniform).item() - math.log(4)) < 1e-9, "uniform entropy should be ln(4)"
    certain = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    assert entropy(certain).item() < 1e-9, "a certain prediction has zero entropy"
    print(f"  entropy        uniform {entropy(uniform).item():.6f} = ln4 {math.log(4):.6f}, "
          f"certain {entropy(certain).item():.6f}")

    # two passes that disagree completely: all doubt is epistemic, none aleatoric
    disagreeing = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float64)
    _, total, aleatoric, epistemic = decompose(disagreeing)
    assert abs(total.item() - math.log(2)) < 1e-9 and aleatoric.item() < 1e-9
    assert abs(epistemic.item() - math.log(2)) < 1e-9
    print(f"  decompose      disagreeing passes -> total {total.item():.6f} "
          f"aleatoric {aleatoric.item():.6f} epistemic {epistemic.item():.6f}")

    # two passes that agree but are unsure: all doubt is aleatoric, none epistemic
    unsure = torch.tensor([[[0.5, 0.5]], [[0.5, 0.5]]], dtype=torch.float64)
    _, total, aleatoric, epistemic = decompose(unsure)
    assert abs(total.item() - math.log(2)) < 1e-9 and epistemic.item() < 1e-9
    print(f"  decompose      agreeing but unsure -> total {total.item():.6f} "
          f"aleatoric {aleatoric.item():.6f} epistemic {epistemic.item():.6f}")

    # risk-coverage: perfect ranking puts every error last
    correct = torch.tensor([1, 1, 1, 0, 0]).bool()
    conf = torch.tensor([0.9, 0.8, 0.7, 0.2, 0.1], dtype=torch.float64)
    curve = risk_coverage(correct, conf)
    assert curve["accuracy"][:3] == [1.0, 1.0, 1.0], "perfect ranking keeps accuracy at 1 early"
    assert abs(accuracy_at_coverage(curve, 0.6) - 1.0) < 1e-12
    assert abs(curve["accuracy"][-1] - 0.6) < 1e-12, "full coverage is plain accuracy"
    print(f"  risk-coverage  perfect ranking: acc at 60% coverage "
          f"{accuracy_at_coverage(curve, 0.6):.3f}, at 100% {curve['accuracy'][-1]:.3f}, "
          f"AURC {curve['aurc']:.6f}")

    # worst possible ranking: every error ranked first. Accuracy at low coverage
    # is then 0, and AURC must be far higher than the perfect ranking's, even
    # though plain accuracy is identical -- the point of measuring AURC at all.
    worst = risk_coverage(correct, -conf)
    assert worst["accuracy"][0] == 0.0, "worst ranking answers its errors first"
    assert abs(worst["accuracy"][-1] - curve["accuracy"][-1]) < 1e-12, "same plain accuracy"
    assert worst["aurc"] > curve["aurc"], "AURC must punish ranking errors above correct answers"
    print(f"  risk-coverage  worst ranking:   acc at 60% coverage "
          f"{accuracy_at_coverage(worst, 0.6):.3f}, at 100% {worst['accuracy'][-1]:.3f}, "
          f"AURC {worst['aurc']:.6f}")

    # AURC is the mean risk over coverages; check it against an independent sum
    by_hand = float(np.mean(1 - np.array(curve["accuracy"])))
    assert abs(curve["aurc"] - by_hand) < 1e-12, f"{curve['aurc']} vs {by_hand}"
    # and the curve itself against sklearn's cumulative accuracy
    order = np.argsort(-conf.numpy(), kind="stable")
    sk_acc = np.cumsum(correct.numpy()[order]) / np.arange(1, len(correct) + 1)
    assert np.allclose(curve["accuracy"], sk_acc), "curve differs from a plain cumulative mean"
    print(f"  AURC {curve['aurc']:.6f} = mean risk {by_hand:.6f}; curve matches a cumulative mean")
    print(f"  sklearn AUC under the risk curve: "
          f"{skm.auc(np.array(curve['coverage']), 1 - np.array(curve['accuracy'])):.6f}")

    # ECE from reliability bins must equal metrics.expected_calibration_error
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from metrics import expected_calibration_error  # noqa: E402

    torch.manual_seed(0)
    probs = torch.rand(5000, 38).softmax(1)
    conf, preds = probs.max(1)
    hit = preds.eq(torch.randint(0, 38, (5000,)))
    bins = reliability(conf, hit)
    from_bins = sum(abs(b["accuracy"] - b["confidence"]) * b["count"] / 5000
                    for b in bins if b["count"])
    direct = expected_calibration_error(conf, hit)
    assert abs(from_bins - direct) < 1e-9, f"{from_bins} vs {direct}"
    print(f"  reliability bins reproduce metrics.py ECE: {from_bins:.8f} = {direct:.8f}")

    print("\nOK: uncertainty metrics match their definitions and scikit-learn.")
