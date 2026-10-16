"""Threshold-free metrics and operating points chosen on validation scores only."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Constant scores have no ranking, which is chance-level discrimination."""
    if len(np.unique(y_true)) < 2 or len(np.unique(scores)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, scores))


def safe_ap(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if len(np.unique(scores)) < 2:
        return float(np.mean(y_true))
    return float(average_precision_score(y_true, scores))


def classification_counts(y_true: np.ndarray, predicted: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "specificity": specificity,
    }


def metric_block(y_true: np.ndarray, scores: np.ndarray, threshold: float, probabilities: bool) -> dict:
    predicted = (scores >= threshold).astype(int)
    counts = classification_counts(y_true, predicted)
    block = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "roc_auc": safe_auc(y_true, scores),
        "pr_auc": safe_ap(y_true, scores),
        **counts,
    }
    if probabilities:
        clipped = np.clip(scores, 0.0, 1.0)
        block["brier"] = float(brier_score_loss(y_true, clipped))
    else:
        block["brier"] = None
    return block


def candidate_thresholds(scores: np.ndarray, probabilities: bool) -> np.ndarray:
    if probabilities:
        grid = np.linspace(0.05, 0.95, 37)
    else:
        grid = np.quantile(scores, np.linspace(0.02, 0.98, 49))
    return np.unique(np.round(grid, 6))


def choose_threshold(y_true: np.ndarray, scores: np.ndarray, mode: str, probabilities: bool) -> float:
    """Pick an operating point using validation labels only."""
    best_threshold = 0.5 if probabilities else float(np.median(scores))
    best_value = None
    for threshold in candidate_thresholds(scores, probabilities):
        predicted = (scores >= threshold).astype(int)
        if mode == "f1":
            value = f1_score(y_true, predicted, zero_division=0)
            better = best_value is None or value > best_value
        elif mode == "overbook":
            # Empty slots are costly, but a false no-show flag that triggers
            # overbooking puts two patients in one slot. Penalise that more.
            value = -_weighted_cost(y_true, predicted, false_positive=5.0, false_negative=1.0)
            better = best_value is None or value > best_value
        elif mode == "remind":
            # A reminder sent to someone who would have attended is cheap
            # relative to a missed visit that nobody followed up.
            value = -_weighted_cost(y_true, predicted, false_positive=1.0, false_negative=3.0)
            better = best_value is None or value > best_value
        else:
            raise ValueError(mode)
        if better:
            best_value = value
            best_threshold = float(threshold)
    return best_threshold


def _weighted_cost(y_true: np.ndarray, predicted: np.ndarray, false_positive: float, false_negative: float) -> float:
    counts = classification_counts(y_true, predicted)
    return false_positive * counts["fp"] + false_negative * counts["fn"]


def top_fraction_threshold(scores: np.ndarray, fraction: float = 0.20) -> float:
    """Flag the highest-scoring fraction of visits, matching a limited contact budget."""
    if fraction <= 0 or fraction >= 1:
        raise ValueError("fraction must be between 0 and 1")
    return float(np.quantile(scores, 1.0 - fraction))


def cost_per_thousand(y_true: np.ndarray, scores: np.ndarray, threshold: float, false_positive: float, false_negative: float) -> float:
    predicted = (scores >= threshold).astype(int)
    total = _weighted_cost(y_true, predicted, false_positive, false_negative)
    return float(1000.0 * total / max(len(y_true), 1))


def operating_points(y_true: np.ndarray, scores: np.ndarray, probabilities: bool) -> dict:
    """Several clinic policies. The F1 point is the default comparison point."""
    f1_threshold = choose_threshold(y_true, scores, "f1", probabilities)
    overbook_threshold = choose_threshold(y_true, scores, "overbook", probabilities)
    remind_threshold = choose_threshold(y_true, scores, "remind", probabilities)
    capacity_threshold = top_fraction_threshold(scores, 0.20)
    default_threshold = 0.5 if probabilities else f1_threshold
    points = {
        "f1": f1_threshold,
        "overbook": overbook_threshold,
        "remind": remind_threshold,
        "capacity20": capacity_threshold,
        "default": default_threshold,
    }
    return points
