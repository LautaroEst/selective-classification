
import numpy as np
from sklearn.metrics import auc, roc_curve
from scipy.special import log_softmax


_MODEL_PERFORMANCE_METRICS = [
    "accuracy", 
    "norm_error_rate",
    "cross_entropy",
    "norm_cross_entropy",
]
_SELECTOR_PERFORMANCE_METRICS = [
    "roc_auc",
    "fpr95tpr",
    "aurc",
]

SUPPORTED_METRICS = _MODEL_PERFORMANCE_METRICS + _SELECTOR_PERFORMANCE_METRICS

def compute_priors(targets):
    priors = np.bincount(targets, minlength=len(np.unique(targets))) / len(targets)
    priors = np.tile(priors, (len(targets), 1))
    return priors

def accuracy(scores, targets, scores_type="logits"):
    return (targets == scores.argmax(axis=1)).mean()

def error_rate(scores, targets, scores_type="logits"):
    return 1 - accuracy(scores, targets)

def cross_entropy(scores, targets, scores_type="logits"):
    if scores_type == "logits" or scores_type == "logprobs":
        return -np.mean(log_softmax(scores, axis=1)[np.arange(len(targets)), targets])
    
    if scores_type == "probs":
        return -np.mean(np.log(scores[np.arange(len(targets)), targets]))
                        
    raise ValueError(f"Invalid scores_type {scores_type}")

def fpr_at_fixed_tpr(fprs, tprs, thresholds, tpr_level: float = 0.95):
    if all(tprs < tpr_level):
        raise ValueError(f"No threshold allows for TPR at least {tpr_level}.")
    idxs = [i for i, x in enumerate(tprs) if x >= tpr_level]
    if len(idxs) > 0:
        idx = min(idxs)
    else:
        idx = 0
    return fprs[idx], tprs[idx], thresholds[idx]

def hard_coverage(scores, thr: float):
    return (scores <= thr).mean()

def selective_net_risk(scores, pred, targets, thr: float):
    covered_idx = scores <= thr
    return np.sum(pred[covered_idx] != targets[covered_idx]) / np.sum(covered_idx)

def risks_coverages_selective_net(scores, pred, targets, sort=True):
    """
    Returns:

        risks, coverages, thrs
    """
    # this function is slow
    risks = []
    coverages = []
    thrs = []
    for thr in np.unique(scores):
        risks.append(selective_net_risk(scores, pred, targets, thr))
        coverages.append(hard_coverage(scores, thr))
        thrs.append(thr)
    risks = np.array(risks)
    coverages = np.array(coverages)
    thrs = np.array(thrs)

    # sort by coverages
    if sort:
        sorted_idx = np.argsort(coverages)
        risks = risks[sorted_idx]
        coverages = coverages[sorted_idx]
        thrs = thrs[sorted_idx]
    return risks, coverages, thrs

def compute_selector_metrics(logits, targets, scores):
    preds = logits.argmax(axis=1)
    train_labels = preds != targets
    fprs, tprs, thrs = roc_curve(train_labels, scores)
    
    roc_auc = auc(fprs, tprs)
    fpr, _, _ = fpr_at_fixed_tpr(fprs, tprs, thrs, 0.95)
    risks, coverages, _ = risks_coverages_selective_net(scores, preds, targets)
    aurc = auc(coverages, risks)
    return {
        "roc_auc": roc_auc,
        "fpr95tpr": fpr,
        "aurc": aurc,
    }

def compute_model_metric(logits, targets, metric, scores_type="logits"):
    if metric.startswith("norm_"):
        metric = metric[5:]
        priors = compute_priors(targets)
        norm_factor = compute_model_metric(priors, targets, metric, scores_type="probs")
    else:
        norm_factor = 1.0

    # Model performance metrics
    if metric == "accuracy":
        result = accuracy(logits, targets)
    elif metric == "error_rate":
        result = error_rate(logits, targets)
    elif metric == "cross_entropy":
        result = cross_entropy(logits, targets, scores_type="logits")
    else:
        raise ValueError(f"Invalid metric {metric}")
    
    return result / norm_factor

def compute_metrics(logits, targets, scores):
    metrics = {}
    for metric in _MODEL_PERFORMANCE_METRICS:
        metrics[metric] = compute_model_metric(logits, targets, metric, scores_type="logits")

    sel_metrics = compute_selector_metrics(logits, targets, scores)
    metrics.update(sel_metrics)

    return metrics
    
