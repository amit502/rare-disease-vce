import warnings
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    accuracy_score, matthews_corrcoef, classification_report,
    roc_auc_score,
)


def compute_metrics(labels, preds, class_names, probs=None):
    """
    Compute the full set of metrics used across VCE papers for fair comparison.
    Returns a flat dict that can be written directly to a CSV row.
    probs: softmax probabilities (N, C) — required for AUC metrics.
    """
    labels = np.array(labels)
    preds  = np.array(preds)
    n_cls  = len(class_names)
    cls_labels = list(range(n_cls))

    metrics = {
        "accuracy":          accuracy_score(labels, preds),
        "f1_micro":          f1_score(labels, preds, average="micro",    zero_division=0),
        "f1_macro":          f1_score(labels, preds, average="macro",    zero_division=0),
        "f1_weighted":       f1_score(labels, preds, average="weighted", zero_division=0),
        "recall_micro":      recall_score(labels, preds, average="micro",    zero_division=0),
        "recall_macro":      recall_score(labels, preds, average="macro",    zero_division=0),
        "recall_weighted":   recall_score(labels, preds, average="weighted", zero_division=0),
        "precision_micro":   precision_score(labels, preds, average="micro",    zero_division=0),
        "precision_macro":   precision_score(labels, preds, average="macro",    zero_division=0),
        "precision_weighted":precision_score(labels, preds, average="weighted", zero_division=0),
        "mcc":               matthews_corrcoef(labels, preds),
    }

    # AUC (requires probabilities)
    if probs is not None:
        probs = np.array(probs)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                per_auc = np.array(roc_auc_score(labels, probs, multi_class="ovr",
                                                  average=None, labels=cls_labels),
                                   dtype=float)
        except ValueError:
            per_auc = np.full(n_cls, float("nan"))

        # Derive aggregates from per-class values, skipping NaN (missing classes)
        counts = np.array([np.sum(labels == i) for i in cls_labels], dtype=float)
        valid  = ~np.isnan(per_auc)
        if valid.any():
            metrics["auc_macro"]    = float(np.mean(per_auc[valid]))
            w = counts[valid] / counts[valid].sum()
            metrics["auc_weighted"] = float(np.dot(per_auc[valid], w))
        else:
            metrics["auc_macro"]    = float("nan")
            metrics["auc_weighted"] = float("nan")
    else:
        per_auc = None

    # Per-class F1, recall, precision
    per_f1        = f1_score(labels, preds, average=None, zero_division=0, labels=cls_labels)
    per_recall    = recall_score(labels, preds, average=None, zero_division=0, labels=cls_labels)
    per_precision = precision_score(labels, preds, average=None, zero_division=0, labels=cls_labels)

    for i, name in enumerate(class_names):
        safe = name.replace(" ", "_").replace("-", "_").lower()
        metrics[f"f1_{safe}"]        = per_f1[i]
        metrics[f"recall_{safe}"]    = per_recall[i]
        metrics[f"precision_{safe}"] = per_precision[i]
        if per_auc is not None:
            metrics[f"auc_{safe}"]   = per_auc[i]

    return metrics


def print_report(labels, preds, class_names):
    print(classification_report(
        labels, preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    ))
