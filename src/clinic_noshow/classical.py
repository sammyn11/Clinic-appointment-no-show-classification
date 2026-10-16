"""Scikit-learn baselines, trees, boosting, and support-vector models."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from clinic_noshow.config import SEED


def logistic_pipeline(C: float, class_weight) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    max_iter=400,
                    solver="lbfgs",
                    random_state=SEED,
                ),
            ),
        ]
    )


def random_forest(max_depth, min_samples_leaf: int, n_estimators: int, max_samples) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_samples=max_samples,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=2,
        random_state=SEED,
    )


def hist_gradient(class_weight) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_depth=6,
        max_iter=250,
        min_samples_leaf=30,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=15,
        class_weight=class_weight,
        random_state=SEED,
    )


def linear_svm(C: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LinearSVC(
                    C=C,
                    class_weight="balanced",
                    dual="auto",
                    max_iter=4000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def rbf_svm() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    cache_size=800,
                    random_state=SEED,
                ),
            ),
        ]
    )


def positive_scores(model, features: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return a ranking score and whether it is a probability."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(features)[:, 1], True
        except Exception:
            pass
    return model.decision_function(features), False


def stratified_subsample(features: np.ndarray, labels: np.ndarray, size: int, seed: int = SEED):
    """Draw a class-stratified subset. Used only because RBF SVM does not scale to the full train set."""
    rng = np.random.default_rng(seed)
    chosen = []
    for label in (0, 1):
        index = np.flatnonzero(labels == label)
        take = max(1, int(round(size * (len(index) / len(labels)))))
        take = min(take, len(index))
        chosen.append(rng.choice(index, size=take, replace=False))
    selected = np.concatenate(chosen)
    rng.shuffle(selected)
    return features[selected], labels[selected]
