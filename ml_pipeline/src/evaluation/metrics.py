"""
Classification and calibration metrics, computed from logits.

Every number in the paper's tables passes through here, so running this file
checks it against scikit-learn:

    python ml_pipeline/src/evaluation/metrics.py

Pure torch, and every value returned is a plain Python float or list, so results
serialise straight to JSON and checkpoints load with torch.load(weights_only=True).
"""

import torch


def confusion_matrix(preds, labels, num_classes):
    """cm[true, predicted] counts."""
    flat = labels * num_classes + preds
    return torch.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def expected_calibration_error(confidences, correct, n_bins=15):
    """ECE over equal-width confidence bins (b/n, (b+1)/n], as in Guo et al. (2017)."""
    bins = (torch.ceil(confidences * n_bins).long() - 1).clamp(0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            gap = correct[in_bin].double().mean() - confidences[in_bin].double().mean()
            ece += gap.abs().item() * in_bin.double().mean().item()
    return ece


def summarize(logits, labels, num_classes, n_bins=15):
    """Every per-split metric, from raw logits.

    Macro-F1 averages over all classes, including any with no support. The
    frozen splits have none, so this only affects --set subset=... debug runs.
    """
    logits = logits.float().cpu()
    labels = labels.long().cpu()
    log_probs = logits.log_softmax(dim=1)
    confidences, preds = log_probs.exp().max(dim=1)
    correct = preds.eq(labels)

    cm = confusion_matrix(preds, labels, num_classes)
    tp = cm.diag().double()
    support = cm.sum(dim=1).double()      # how many of each class there are
    predicted = cm.sum(dim=0).double()    # how often each class was predicted
    f1 = 2 * tp / (support + predicted).clamp(min=1)
    recall = tp / support.clamp(min=1)

    return {
        "acc": correct.double().mean().item(),
        "macro_f1": f1.mean().item(),
        "balanced_acc": recall.mean().item(),
        "nll": -log_probs.gather(1, labels[:, None]).double().mean().item(),
        "ece": expected_calibration_error(confidences, correct, n_bins),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": cm.tolist(),
    }


if __name__ == "__main__":
    import numpy as np
    from sklearn import metrics as skm

    K, N = 38, 5000
    torch.manual_seed(0)
    labels = torch.arange(N) % K                      # every class present
    logits = torch.randn(N, K) * 2
    logits[torch.arange(N), labels] += 3.0            # mostly right, like a trained model
    m = summarize(logits, labels, K)

    y, p = labels.numpy(), logits.argmax(1).numpy()
    probs = logits.double().softmax(1).numpy()
    reference = {
        "acc": skm.accuracy_score(y, p),
        "macro_f1": skm.f1_score(y, p, average="macro", labels=range(K), zero_division=0),
        "balanced_acc": skm.balanced_accuracy_score(y, p),
        "nll": skm.log_loss(y, probs, labels=range(K)),
    }
    for key, ref in reference.items():
        assert abs(m[key] - ref) < 1e-6, f"{key}: {m[key]} vs sklearn {ref}"
        print(f"  {key:<13} {m[key]:.6f}   sklearn {ref:.6f}")

    per_class = skm.f1_score(y, p, average=None, labels=range(K), zero_division=0)
    assert np.allclose(m["per_class_f1"], per_class), "per-class F1 differs from sklearn"
    assert (np.array(m["confusion_matrix"]) == skm.confusion_matrix(y, p, labels=range(K))).all()
    print("  per-class F1 and confusion matrix match sklearn")

    # ECE has no sklearn equivalent: check a case with a known answer (every
    # prediction 80% confident, 60% right -> 0.2), then an independent reference
    known = expected_calibration_error(torch.full((10,), 0.8, dtype=torch.float64),
                                       torch.tensor([1] * 6 + [0] * 4).bool())
    assert abs(known - 0.2) < 1e-12, known

    conf = logits.softmax(1).max(1).values.double().numpy()
    hit = (p == y).astype(float)
    edges = np.linspace(0, 1, 16)
    ref_ece = sum(abs(hit[s].mean() - conf[s].mean()) * s.mean()
                  for s in ((conf > lo) & (conf <= hi) for lo, hi in zip(edges[:-1], edges[1:]))
                  if s.any())
    assert abs(m["ece"] - ref_ece) < 1e-6, f"ece {m['ece']} vs reference {ref_ece}"
    print(f"  ece           {m['ece']:.6f}   reference {ref_ece:.6f}")

    print("\nOK: metrics match scikit-learn and the ECE reference.")
