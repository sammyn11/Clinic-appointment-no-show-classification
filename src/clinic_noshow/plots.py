"""Figures for the notebook and the written report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix

ATTENDED = "#1F4E79"
NOSHOW = "#C65D3B"
TRAIN_COLOR = "#1F4E79"
VAL_COLOR = "#C65D3B"
NEUTRAL = "#5C6B73"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def class_balance(profile: dict, path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    labels = ["Attended", "No-show"]
    values = [profile["n_attended"], profile["n_noshow"]]
    bars = ax.bar(labels, values, color=[ATTENDED, NOSHOW], width=0.62)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom")
    ax.set_ylabel("Appointments")
    ax.set_title("Cleaned cohort")
    ax.set_ylim(0, max(values) * 1.12)
    save(fig, path)


def noshow_by_lead(train: pd.DataFrame, path: Path) -> None:
    _style()
    bins = [-0.1, 0, 2, 7, 14, 30, 60, 10_000]
    labels = ["0", "1–2", "3–7", "8–14", "15–30", "31–60", "61+"]
    lead = pd.cut(train["lead_days"], bins=bins, labels=labels)
    rate = train.groupby(lead, observed=True)["noshow"].mean()
    counts = train.groupby(lead, observed=True)["noshow"].size()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(rate.index.astype(str), rate.values, color=NOSHOW, width=0.72)
    ax.set_ylabel("No-show rate")
    ax.set_xlabel("Days from booking to appointment (training patients)")
    ax.set_ylim(0, max(0.45, float(rate.max()) * 1.25))
    ax.set_title("Longer booking lead time, higher non-attendance")
    for label, value, count in zip(rate.index.astype(str), rate.values, counts.values):
        ax.text(label, value + 0.01, f"n={count:,}", ha="center", va="bottom", fontsize=8, color=NEUTRAL)
    save(fig, path)


def sms_confound(train: pd.DataFrame, path: Path) -> None:
    """Show that SMS is entangled with lead time, so a raw SMS effect is not causal."""
    _style()
    bins = [-0.1, 0, 2, 7, 14, 30, 10_000]
    labels = ["0", "1–2", "3–7", "8–14", "15–30", "31+"]
    work = train.copy()
    work["lead_bin"] = pd.cut(work["lead_days"], bins=bins, labels=labels)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    width = 0.38
    positions = np.arange(len(labels))
    for shift, sms_value, color, name in (
        (-width / 2, 0, ATTENDED, "No SMS"),
        (width / 2, 1, NOSHOW, "SMS sent"),
    ):
        subset = work.loc[work["sms_received"] == sms_value]
        rate = subset.groupby("lead_bin", observed=False)["noshow"].mean().reindex(labels)
        ax.bar(positions + shift, rate.fillna(0).to_numpy(), width=width, color=color, label=name)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Lead time in days (training patients)")
    ax.set_ylabel("No-show rate")
    ax.set_title("SMS was not sent for same-day visits")
    ax.legend(frameon=False)
    save(fig, path)


def noshow_by_age(train: pd.DataFrame, path: Path) -> None:
    _style()
    bins = [0, 18, 30, 45, 60, 111]
    labels = ["0–17", "18–29", "30–44", "45–59", "60+"]
    age = pd.cut(train["age"], bins=bins, labels=labels, right=False)
    rate = train.groupby(age, observed=True)["noshow"].mean()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(rate.index.astype(str), rate.values, color=ATTENDED, width=0.7)
    ax.set_ylabel("No-show rate")
    ax.set_xlabel("Age (training patients)")
    ax.set_title("Younger adults miss visits more often")
    ax.set_ylim(0, max(0.4, float(rate.max()) * 1.2))
    save(fig, path)


def learning_curve(sizes: np.ndarray, train_scores: np.ndarray, val_scores: np.ndarray, path: Path, title: str) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_std = val_scores.std(axis=1)
    ax.plot(sizes, train_mean, color=TRAIN_COLOR, marker="o", label="Training ROC-AUC")
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, color=TRAIN_COLOR, alpha=0.15)
    ax.plot(sizes, val_mean, color=VAL_COLOR, marker="o", label="Cross-validation ROC-AUC")
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, color=VAL_COLOR, alpha=0.15)
    ax.set_xlabel("Training appointments")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(title)
    ax.set_ylim(0.5, 1.0)
    ax.legend(frameon=False)
    save(fig, path)


def neural_history(histories: dict[str, dict], path: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    for name, history in histories.items():
        epochs = np.arange(1, len(history["loss"]) + 1)
        axes[0].plot(epochs, history["loss"], label=f"{name} train")
        axes[0].plot(epochs, history["val_loss"], linestyle="--", label=f"{name} val")
        axes[1].plot(epochs, history["val_auc"], label=name)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary cross-entropy")
    axes[0].set_title("Loss")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation ROC-AUC")
    axes[1].set_title("Ranking quality while training")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, path)


def roc_overlay(curves: list[tuple[str, np.ndarray, np.ndarray]], path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for name, y_true, scores in curves:
        RocCurveDisplay.from_predictions(y_true, scores, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], color=NEUTRAL, linestyle=":", label="Chance")
    ax.set_title("Test-set ROC curves")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, path)


def pr_overlay(curves: list[tuple[str, np.ndarray, np.ndarray]], base_rate: float, path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for name, y_true, scores in curves:
        PrecisionRecallDisplay.from_predictions(y_true, scores, name=name, ax=ax)
    ax.axhline(base_rate, color=NEUTRAL, linestyle=":", label=f"Prevalence ({base_rate:.2f})")
    ax.set_title("Test-set precision–recall curves")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, path)


def confusion(y_true: np.ndarray, scores: np.ndarray, threshold: float, path: Path, title: str) -> None:
    _style()
    predicted = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred. attend", "Pred. no-show"])
    ax.set_yticks([0, 1], ["Attended", "No-show"])
    ax.set_title(title)
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{matrix[row, col]:,}", ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046)
    save(fig, path)


def coefficient_bars(names: list[str], values: np.ndarray, path: Path, title: str) -> None:
    _style()
    order = np.argsort(np.abs(values))
    names = [names[i] for i in order]
    values = values[order]
    colors = [NOSHOW if value > 0 else ATTENDED for value in values]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.barh(names, values, color=colors)
    ax.axvline(0, color=NEUTRAL, linewidth=0.8)
    ax.set_xlabel("Standardized log-odds coefficient")
    ax.set_title(title)
    save(fig, path)


def importance_bars(names: list[str], values: np.ndarray, path: Path, title: str) -> None:
    _style()
    order = np.argsort(values)
    names = [names[i] for i in order]
    values = values[order]
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.barh(names, values, color=ATTENDED)
    ax.set_xlabel("Drop in validation ROC-AUC when shuffled")
    ax.set_title(title)
    save(fig, path)


def calibration(curves: list[tuple[str, np.ndarray, np.ndarray]], path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.plot([0, 1], [0, 1], linestyle=":", color=NEUTRAL, label="Perfect")
    for name, y_true, scores in curves:
        fraction, mean_pred = calibration_curve(y_true, np.clip(scores, 0, 1), n_bins=10, strategy="quantile")
        ax.plot(mean_pred, fraction, marker="o", label=name)
    ax.set_xlabel("Predicted no-show probability")
    ax.set_ylabel("Observed no-show rate")
    ax.set_title("Reliability on the test set")
    ax.legend(frameon=False, fontsize=8)
    save(fig, path)


def threshold_tradeoff(y_true: np.ndarray, scores: np.ndarray, path: Path, title: str) -> None:
    _style()
    thresholds = np.linspace(0.05, 0.95, 37)
    precision, recall = [], []
    for threshold in thresholds:
        predicted = (scores >= threshold).astype(int)
        tp = np.sum((predicted == 1) & (y_true == 1))
        fp = np.sum((predicted == 1) & (y_true == 0))
        fn = np.sum((predicted == 0) & (y_true == 1))
        precision.append(tp / (tp + fp) if (tp + fp) else 0.0)
        recall.append(tp / (tp + fn) if (tp + fn) else 0.0)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(thresholds, precision, color=ATTENDED, label="Precision")
    ax.plot(thresholds, recall, color=NOSHOW, label="Recall")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    save(fig, path)


def slice_recall(table: pd.DataFrame, path: Path, title: str) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    labels = table["slice"].tolist()
    ax.barh(labels, table["recall"], color=NOSHOW)
    ax.set_xlabel("Recall of no-shows")
    ax.set_xlim(0, 1)
    ax.set_title(title)
    save(fig, path)
