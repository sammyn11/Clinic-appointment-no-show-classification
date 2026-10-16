"""Run the classical and neural experiments and write metrics, figures, and models.

Validation scores choose hyperparameters and decision thresholds. The test set is
scored after those choices are fixed.
"""

from __future__ import annotations

import json
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve

from clinic_noshow.classical import (
    hist_gradient,
    linear_svm,
    logistic_pipeline,
    positive_scores,
    random_forest,
    rbf_svm,
    stratified_subsample,
)
from clinic_noshow.config import DATA_CSV, DELIVERABLES, FIG_DIR, METRIC_DIR, MODEL_DIR, SEED, WANDB_DIR
from clinic_noshow.evaluate import (
    cost_per_thousand,
    metric_block,
    operating_points,
    safe_auc,
)
from clinic_noshow.plots import (
    calibration,
    class_balance,
    coefficient_bars,
    confusion,
    importance_bars,
    learning_curve as plot_learning_curve,
    neural_history,
    noshow_by_age,
    noshow_by_lead,
    pr_overlay,
    roc_overlay,
    slice_recall,
    sms_confound,
    threshold_tradeoff,
)
from clinic_noshow.preprocess import labels, matrix, prepare_dataset


def _jsonify(value):
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    return value


def _log_wandb(result: dict) -> None:
    import os

    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_SILENT", "true")
    try:
        import wandb

        wandb.init(
            project="clinic-appointment-noshow",
            name=result["id"],
            group=result["paradigm"],
            mode="offline",
            dir=str(WANDB_DIR),
            config={
                "model": result["model"],
                "features": result["features"],
                "hyperparameters": result["hyperparameters"],
                "intent": result["intent"],
            },
            reinit="finish_previous",
        )
        payload = {"fit_seconds": result["fit_seconds"], "auc_gap_train_minus_val": result["auc_gap"]}
        for split in ("train", "val", "test"):
            for key, value in result[split].items():
                if isinstance(value, (int, float)):
                    payload[f"{split}_{key}"] = value
        wandb.log(payload)
        wandb.finish()
    except Exception as exc:  # noqa: BLE001 - logging must not discard a finished run
        result["wandb_error"] = str(exc)


def _pack(
    *,
    experiment_id: str,
    model: str,
    paradigm: str,
    intent: str,
    feature_set: str,
    hyperparameters: dict,
    y_train,
    train_scores,
    y_val,
    val_scores,
    y_test,
    test_scores,
    probabilities: bool,
    fit_seconds: float,
    fixed_threshold: float | None = None,
) -> dict:
    if fixed_threshold is None:
        points = operating_points(y_val, val_scores, probabilities)
        threshold = points["f1"]
    else:
        threshold = fixed_threshold
        points = {name: fixed_threshold for name in ("f1", "overbook", "remind", "capacity20", "default")}
    train_metrics = metric_block(y_train, train_scores, threshold, probabilities)
    val_metrics = metric_block(y_val, val_scores, threshold, probabilities)
    test_metrics = metric_block(y_test, test_scores, threshold, probabilities)
    gap = float(train_metrics["roc_auc"] - val_metrics["roc_auc"])
    result = {
        "id": experiment_id,
        "model": model,
        "paradigm": paradigm,
        "intent": intent,
        "features": feature_set,
        "hyperparameters": hyperparameters,
        "probabilities": probabilities,
        "threshold_source": "fixed" if fixed_threshold is not None else "validation F1",
        "thresholds_from_validation": points,
        "fit_seconds": round(fit_seconds, 2),
        "auc_gap": round(gap, 4),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    print(
        f"{experiment_id:4} {model:28} val AUC {val_metrics['roc_auc']:.3f}  "
        f"PR {val_metrics['pr_auc']:.3f}  F1 {val_metrics['f1']:.3f}  gap {gap:.3f}"
    )
    return result


def _store(results: list, bank: dict, result: dict, scores: dict) -> None:
    _log_wandb(result)
    results.append(result)
    bank[result["id"]] = scores
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    (METRIC_DIR / "results.json").write_text(json.dumps(_jsonify(results), indent=2))


def _lead_rate_table(train: pd.DataFrame) -> None:
    bins = [-0.1, 0, 2, 7, 14, 30, 60, 10_000]
    labels = ["0", "1–2", "3–7", "8–14", "15–30", "31–60", "61+"]
    lead = pd.cut(train["lead_days"], bins=bins, labels=labels)
    table = (
        train.groupby(lead, observed=True)["noshow"]
        .agg(n="size", noshow_rate="mean")
        .reset_index()
        .rename(columns={"lead_days": "lead_bin"})
    )
    table.to_csv(METRIC_DIR / "lead_time_rates_train.csv", index=False)


def _slice_table(frame: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    y = labels(frame)
    rows = []

    def add(name: str, mask: np.ndarray) -> None:
        if int(mask.sum()) < 40 or len(np.unique(y[mask])) < 2:
            return
        block = metric_block(y[mask], scores[mask], threshold, True)
        rows.append(
            {
                "slice": name,
                "n": int(mask.sum()),
                "noshow_rate": float(y[mask].mean()),
                "recall": block["recall"],
                "precision": block["precision"],
                "roc_auc": block["roc_auc"],
            }
        )

    lead = frame["lead_days"].to_numpy()
    age = frame["age"].to_numpy()
    add("Lead 0 days", lead == 0)
    add("Lead 1–7 days", (lead >= 1) & (lead <= 7))
    add("Lead 8–30 days", (lead >= 8) & (lead <= 30))
    add("Lead 31+ days", lead >= 31)
    add("Age 0–17", age < 18)
    add("Age 18–39", (age >= 18) & (age <= 39))
    add("Age 40–59", (age >= 40) & (age <= 59))
    add("Age 60+", age >= 60)
    add("Women", frame["gender_male"].to_numpy() == 0)
    add("Men", frame["gender_male"].to_numpy() == 1)
    add("Bolsa Família", frame["scholarship"].to_numpy() == 1)
    add("No scholarship", frame["scholarship"].to_numpy() == 0)
    add("SMS sent", frame["sms_received"].to_numpy() == 1)
    add("No SMS", frame["sms_received"].to_numpy() == 0)
    add("First visit in extract", frame["is_first_appt"].to_numpy() == 1)
    add("Return visit", frame["is_first_appt"].to_numpy() == 0)
    return pd.DataFrame(rows)


def _pretty_feature(name: str) -> str:
    days = {
        "dow_1": "Tuesday",
        "dow_2": "Wednesday",
        "dow_3": "Thursday",
        "dow_4": "Friday",
        "dow_5": "Saturday",
        "dow_6": "Sunday",
    }
    names = {
        "gender_male": "Male",
        "hipertension": "Hypertension",
        "handcap": "Handicap level",
        "sms_received": "SMS received",
        "lead_days": "Lead time (days)",
        "lead_days_log1p": "Log lead time",
        "same_day": "Same-day booking",
        "sched_hour": "Booking hour",
        "prior_appt_count": "Prior visits",
        "prior_noshow_rate": "Prior no-show rate",
        "days_since_prev": "Days since previous visit",
        "is_first_appt": "First visit in extract",
        "neighbourhood_freq": "Neighbourhood frequency",
        "neighbourhood_te": "Neighbourhood no-show rate",
        "scholarship": "Scholarship",
        "diabetes": "Diabetes",
        "alcoholism": "Alcoholism",
        "age": "Age",
    }
    return days.get(name, names.get(name, name))


def run_all() -> list[dict]:
    """Fit every experiment. Safe to re-run; it overwrites artifacts from this pipeline."""
    np.random.seed(SEED)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    WANDB_DIR.mkdir(parents=True, exist_ok=True)

    data = prepare_dataset()
    train, val, test = data.train, data.val, data.test
    y_train, y_val, y_test = labels(train), labels(val), labels(test)
    sets = data.feature_sets
    class_balance(data.profile, FIG_DIR / "class_balance.png")
    noshow_by_lead(train, FIG_DIR / "noshow_by_lead.png")
    sms_confound(train, FIG_DIR / "sms_by_lead.png")
    noshow_by_age(train, FIG_DIR / "noshow_by_age.png")
    _lead_rate_table(train)

    results: list[dict] = []
    bank: dict[str, dict] = {}
    fitted: dict = {}

    def remember(experiment_id: str, estimator, train_scores, val_scores, test_scores, probabilities: bool) -> None:
        fitted[experiment_id] = estimator
        bank_scores = {
            "train": train_scores,
            "val": val_scores,
            "test": test_scores,
            "probabilities": probabilities,
        }
        return bank_scores

    # E0 — majority baseline. Scores are constant so ranking is chance.
    t0 = time.perf_counter()
    zeros_train = np.zeros(len(y_train))
    zeros_val = np.zeros(len(y_val))
    zeros_test = np.zeros(len(y_test))
    result = _pack(
        experiment_id="E0",
        model="Majority class (always attend)",
        paradigm="baseline",
        intent="Show that accuracy near the attendance rate can hide a model that never detects a no-show.",
        feature_set="none",
        hyperparameters={"rule": "predict attended for every visit"},
        y_train=y_train,
        train_scores=zeros_train,
        y_val=y_val,
        val_scores=zeros_val,
        y_test=y_test,
        test_scores=zeros_test,
        probabilities=True,
        fit_seconds=time.perf_counter() - t0,
        fixed_threshold=0.5,
    )
    _store(results, bank, result, remember("E0", None, zeros_train, zeros_val, zeros_test, True))

    # E1 — transparent lead-time rule.
    t0 = time.perf_counter()
    lead_train = train["lead_days"].to_numpy(dtype=float)
    lead_val = val["lead_days"].to_numpy(dtype=float)
    lead_test = test["lead_days"].to_numpy(dtype=float)
    result = _pack(
        experiment_id="E1",
        model="Lead-time rule",
        paradigm="baseline",
        intent="Test whether a single booking-gap cutoff already captures most of the separable signal.",
        feature_set="lead_days",
        hyperparameters={"score": "lead_days", "threshold": "validation F1"},
        y_train=y_train,
        train_scores=lead_train,
        y_val=y_val,
        val_scores=lead_val,
        y_test=y_test,
        test_scores=lead_test,
        probabilities=False,
        fit_seconds=time.perf_counter() - t0,
    )
    _store(results, bank, result, remember("E1", None, lead_train, lead_val, lead_test, False))

    def run_estimator(experiment_id, model_name, paradigm, intent, feature_key, hyper, builder):
        columns = sets[feature_key]
        x_train = matrix(train, columns)
        x_val = matrix(val, columns)
        x_test = matrix(test, columns)
        t0 = time.perf_counter()
        estimator = builder()
        estimator.fit(x_train, y_train)
        seconds = time.perf_counter() - t0
        train_scores, probabilities = positive_scores(estimator, x_train)
        val_scores, _ = positive_scores(estimator, x_val)
        test_scores, _ = positive_scores(estimator, x_test)
        result = _pack(
            experiment_id=experiment_id,
            model=model_name,
            paradigm=paradigm,
            intent=intent,
            feature_set=feature_key,
            hyperparameters={**hyper, "n_features": len(columns)},
            y_train=y_train,
            train_scores=train_scores,
            y_val=y_val,
            val_scores=val_scores,
            y_test=y_test,
            test_scores=test_scores,
            probabilities=probabilities,
            fit_seconds=seconds,
        )
        _store(results, bank, result, remember(experiment_id, estimator, train_scores, val_scores, test_scores, probabilities))
        return estimator

    run_estimator(
        "E2",
        "Logistic regression",
        "traditional",
        "Demographic and clinical flags only, before any booking-process features.",
        "demographic",
        {"C": 1.0, "class_weight": None, "penalty": "l2"},
        lambda: logistic_pipeline(1.0, None),
    )
    run_estimator(
        "E3",
        "Logistic regression",
        "traditional",
        "Add lead time, booking hour, weekday, same-day flag, and SMS. Balance classes so the rare no-show is not ignored.",
        "scheduling",
        {"C": 1.0, "class_weight": "balanced", "penalty": "l2"},
        lambda: logistic_pipeline(1.0, "balanced"),
    )
    run_estimator(
        "E4",
        "Logistic regression",
        "traditional",
        "Add personal history and neighbourhood encodings on top of the scheduling features.",
        "full",
        {"C": 1.0, "class_weight": "balanced", "penalty": "l2"},
        lambda: logistic_pipeline(1.0, "balanced"),
    )

    # E5 — C chosen on validation ROC-AUC.
    columns = sets["full"]
    x_train, x_val, x_test = matrix(train, columns), matrix(val, columns), matrix(test, columns)
    grid = {}
    best = None
    t0 = time.perf_counter()
    for C in (0.01, 0.1, 1.0, 10.0):
        estimator = logistic_pipeline(C, "balanced")
        estimator.fit(x_train, y_train)
        val_scores, _ = positive_scores(estimator, x_val)
        auc = safe_auc(y_val, val_scores)
        grid[str(C)] = round(auc, 4)
        if best is None or auc > best[0]:
            best = (auc, C, estimator)
    _, chosen_c, chosen_lr = best
    train_scores, probabilities = positive_scores(chosen_lr, x_train)
    val_scores, _ = positive_scores(chosen_lr, x_val)
    test_scores, _ = positive_scores(chosen_lr, x_test)
    result = _pack(
        experiment_id="E5",
        model="Logistic regression",
        paradigm="traditional",
        intent="Sweep inverse regularisation on the full feature set and keep the value with the best validation ROC-AUC.",
        feature_set="full",
        hyperparameters={"C_grid_val_auc": grid, "C_selected": chosen_c, "class_weight": "balanced"},
        y_train=y_train,
        train_scores=train_scores,
        y_val=y_val,
        val_scores=val_scores,
        y_test=y_test,
        test_scores=test_scores,
        probabilities=probabilities,
        fit_seconds=time.perf_counter() - t0,
    )
    _store(results, bank, result, remember("E5", chosen_lr, train_scores, val_scores, test_scores, probabilities))

    run_estimator(
        "E6",
        "Random forest",
        "traditional",
        "Shallow forest. Depth and leaf size are constrained so the model cannot memorise individual visits.",
        "full",
        {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 20, "max_features": "sqrt", "class_weight": "balanced_subsample"},
        lambda: random_forest(8, 20, 200, None),
    )
    run_estimator(
        "E7",
        "Random forest",
        "traditional",
        "Deliberately high-variance forest used to expose the train–validation gap when depth is unlimited.",
        "full",
        {"n_estimators": 80, "max_depth": None, "min_samples_leaf": 1, "max_samples": 0.45, "class_weight": "balanced_subsample"},
        lambda: random_forest(None, 1, 80, 0.45),
    )
    run_estimator(
        "E8",
        "Random forest",
        "traditional",
        "Intermediate depth and leaf size, placed between the constrained forest and the unrestricted one.",
        "full",
        {"n_estimators": 250, "max_depth": 16, "min_samples_leaf": 5, "class_weight": "balanced_subsample"},
        lambda: random_forest(16, 5, 250, None),
    )
    run_estimator(
        "E9",
        "HistGradientBoosting",
        "traditional",
        "Boosted trees with class weights, shrinkage, and early stopping on an internal training slice.",
        "full",
        {
            "learning_rate": 0.08,
            "max_depth": 6,
            "max_iter": 250,
            "min_samples_leaf": 30,
            "l2_regularization": 0.1,
            "class_weight": "balanced",
            "early_stopping": True,
        },
        lambda: hist_gradient("balanced"),
    )
    run_estimator(
        "E10",
        "HistGradientBoosting",
        "traditional",
        "Same booster without class weights, to separate ranking quality from probability distortion.",
        "full",
        {
            "learning_rate": 0.08,
            "max_depth": 6,
            "max_iter": 250,
            "min_samples_leaf": 30,
            "l2_regularization": 0.1,
            "class_weight": None,
            "early_stopping": True,
        },
        lambda: hist_gradient(None),
    )

    # E11 — linear SVM, C chosen on validation ROC-AUC. Scores are margins, not probabilities.
    svm_grid = {}
    best_svm = None
    t0 = time.perf_counter()
    for C in (0.1, 1.0, 10.0):
        estimator = linear_svm(C)
        estimator.fit(x_train, y_train)
        val_scores, _ = positive_scores(estimator, x_val)
        auc = safe_auc(y_val, val_scores)
        svm_grid[str(C)] = round(auc, 4)
        if best_svm is None or auc > best_svm[0]:
            best_svm = (auc, C, estimator)
    _, chosen_svm_c, chosen_svm = best_svm
    train_scores, probabilities = positive_scores(chosen_svm, x_train)
    val_scores, _ = positive_scores(chosen_svm, x_val)
    test_scores, _ = positive_scores(chosen_svm, x_test)
    result = _pack(
        experiment_id="E11",
        model="Linear SVM",
        paradigm="traditional",
        intent="Max-margin linear separator on the same scaled features as logistic regression. Full data, linear kernel only.",
        feature_set="full",
        hyperparameters={"C_grid_val_auc": svm_grid, "C_selected": chosen_svm_c, "class_weight": "balanced", "kernel": "linear"},
        y_train=y_train,
        train_scores=train_scores,
        y_val=y_val,
        val_scores=val_scores,
        y_test=y_test,
        test_scores=test_scores,
        probabilities=probabilities,
        fit_seconds=time.perf_counter() - t0,
    )
    _store(results, bank, result, remember("E11", chosen_svm, train_scores, val_scores, test_scores, probabilities))

    # E12 — RBF SVM on a stratified training subsample.
    t0 = time.perf_counter()
    sample_x, sample_y = stratified_subsample(x_train, y_train, size=8000)
    rbf = rbf_svm()
    rbf.fit(sample_x, sample_y)
    # Train metrics refer to the 8,000 rows the kernel actually saw.
    train_scores, probabilities = positive_scores(rbf, sample_x)
    val_scores, _ = positive_scores(rbf, x_val)
    test_scores, _ = positive_scores(rbf, x_test)
    result = _pack(
        experiment_id="E12",
        model="RBF SVM",
        paradigm="traditional",
        intent="Nonlinear margin on 8,000 stratified training visits. The full table is too large for a kernel SVM on this machine.",
        feature_set="full",
        hyperparameters={"kernel": "rbf", "C": 1.0, "gamma": "scale", "train_subsample": 8000, "class_weight": "balanced"},
        y_train=sample_y,
        train_scores=train_scores,
        y_val=y_val,
        val_scores=val_scores,
        y_test=y_test,
        test_scores=test_scores,
        probabilities=probabilities,
        fit_seconds=time.perf_counter() - t0,
    )
    _store(results, bank, result, remember("E12", rbf, train_scores, val_scores, test_scores, probabilities))

    _run_neural(data, results, bank, fitted)
    _diagnostics(data, results, bank, fitted)
    _export_models(results, fitted)
    print(f"Finished {len(results)} experiments.")
    return results


def _run_neural(data, results: list, bank: dict, fitted: dict) -> None:
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf

    from clinic_noshow import neural

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    neural.set_seeds(SEED)

    train, val, test = data.train, data.val, data.test
    columns = data.feature_sets["full"]
    y_train, y_val, y_test = labels(train), labels(val), labels(test)
    x_train, x_val, x_test, scaler = neural.scale_features(
        matrix(train, columns), matrix(val, columns), matrix(test, columns)
    )
    weights = neural.class_weight_map(y_train)
    histories = {}

    def train_flat(experiment_id, model_name, intent, builder, class_weight, epochs, patience):
        neural.set_seeds(SEED)
        model = builder(x_train.shape[1])
        train_ds = neural.make_flat_dataset(x_train, y_train, batch_size=512, shuffle=True)
        val_ds = neural.make_flat_dataset(x_val, y_val, batch_size=512, shuffle=False)
        callback = neural.ValidationAUC(
            lambda: (neural.predict_flat(model, x_val), y_val),
            patience=patience,
        )
        t0 = time.perf_counter()
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            class_weight=class_weight,
            callbacks=[callback],
            verbose=0,
            shuffle=False,
        )
        seconds = time.perf_counter() - t0
        train_scores = neural.predict_flat(model, x_train)
        val_scores = neural.predict_flat(model, x_val)
        test_scores = neural.predict_flat(model, x_test)
        result = _pack(
            experiment_id=experiment_id,
            model=model_name,
            paradigm="deep",
            intent=intent,
            feature_set="full",
            hyperparameters={
                "architecture": model_name,
                "epochs_ran": len(callback.history["loss"]),
                "epochs_cap": epochs,
                "batch_size": 512,
                "class_weight": class_weight,
                "early_stopping": "validation ROC-AUC",
                "patience": patience,
                "input": "tf.data",
            },
            y_train=y_train,
            train_scores=train_scores,
            y_val=y_val,
            val_scores=val_scores,
            y_test=y_test,
            test_scores=test_scores,
            probabilities=True,
            fit_seconds=seconds,
        )
        _store(results, bank, result, {"train": train_scores, "val": val_scores, "test": test_scores, "probabilities": True})
        fitted[experiment_id] = {"model": model, "kind": "keras-flat", "scaler": scaler, "columns": columns}
        histories[experiment_id] = callback.history
        return model

    train_flat(
        "E13",
        "MLP shallow",
        "Sequential network on the same flat features as the linear model, without dropout or class weights.",
        neural.build_shallow,
        None,
        epochs=20,
        patience=4,
    )
    train_flat(
        "E14",
        "MLP regularized",
        "Deeper Sequential network with batch normalisation, dropout, L2, a lower learning rate, and class weights.",
        neural.build_regularized,
        weights,
        epochs=25,
        patience=5,
    )

    # E15 — functional model with a neighbourhood embedding.
    embed_columns = [column for column in columns if column not in {"neighbourhood_freq", "neighbourhood_te"}]
    e_train, e_val, e_test, embed_scaler = neural.scale_features(
        matrix(train, embed_columns), matrix(val, embed_columns), matrix(test, embed_columns)
    )
    n_ids = int(max(data.neighbourhood_to_id.values()) + 1)
    n_train = train["neighbourhood_id"].to_numpy()
    n_val = val["neighbourhood_id"].to_numpy()
    n_test = test["neighbourhood_id"].to_numpy()
    neural.set_seeds(SEED)
    embedding_model = neural.build_embedding(e_train.shape[1], n_ids)
    train_ds = neural.make_embedding_dataset(e_train, n_train, y_train, 512, True)
    val_ds = neural.make_embedding_dataset(e_val, n_val, y_val, 512, False)
    callback = neural.ValidationAUC(
        lambda: (neural.predict_embedding(embedding_model, e_val, n_val), y_val),
        patience=5,
    )
    t0 = time.perf_counter()
    embedding_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=25,
        class_weight=weights,
        callbacks=[callback],
        verbose=0,
        shuffle=False,
    )
    seconds = time.perf_counter() - t0
    train_scores = neural.predict_embedding(embedding_model, e_train, n_train)
    val_scores = neural.predict_embedding(embedding_model, e_val, n_val)
    test_scores = neural.predict_embedding(embedding_model, e_test, n_test)
    result = _pack(
        experiment_id="E15",
        model="MLP neighbourhood embedding",
        paradigm="deep",
        intent="Functional API: numeric tower plus a learned neighbourhood embedding instead of the hand-built location rates.",
        feature_set="full_without_handbuilt_location",
        hyperparameters={
            "embedding_dim": 8,
            "n_neighbourhood_ids": n_ids,
            "dropped_handbuilt_features": ["neighbourhood_freq", "neighbourhood_te"],
            "epochs_ran": len(callback.history["loss"]),
            "class_weight": weights,
            "learning_rate": 1e-3,
            "input": "tf.data",
        },
        y_train=y_train,
        train_scores=train_scores,
        y_val=y_val,
        val_scores=val_scores,
        y_test=y_test,
        test_scores=test_scores,
        probabilities=True,
        fit_seconds=seconds,
    )
    _store(
        results,
        bank,
        result,
        {"train": train_scores, "val": val_scores, "test": test_scores, "probabilities": True},
    )
    fitted["E15"] = {
        "model": embedding_model,
        "kind": "keras-embedding",
        "scaler": embed_scaler,
        "columns": embed_columns,
        "neighbourhood_to_id": data.neighbourhood_to_id,
    }
    histories["E15"] = callback.history
    (METRIC_DIR / "dl_histories.json").write_text(json.dumps(histories, indent=2))
    neural_history(histories, FIG_DIR / "dl_learning_curves.png")


def _diagnostics(data, results: list, bank: dict, fitted: dict) -> None:
    train, val, test = data.train, data.val, data.test
    y_train, y_val, y_test = labels(train), labels(val), labels(test)
    columns = data.feature_sets["full"]
    x_train = matrix(train, columns)

    # Learning curve for the tuned linear model and the constrained forest.
    lr_estimator = logistic_pipeline(float(fitted["E5"].named_steps["clf"].C), "balanced")
    sizes, train_scores, val_scores = learning_curve(
        lr_estimator,
        x_train,
        y_train,
        train_sizes=np.linspace(0.15, 1.0, 5),
        cv=3,
        scoring="roc_auc",
        n_jobs=2,
        shuffle=True,
        random_state=SEED,
    )
    plot_learning_curve(
        sizes,
        train_scores,
        val_scores,
        FIG_DIR / "learning_curve_logreg.png",
        "Logistic regression learning curve",
    )
    rf_estimator = random_forest(8, 20, 80, None)
    sizes, train_scores, val_scores = learning_curve(
        rf_estimator,
        x_train,
        y_train,
        train_sizes=np.linspace(0.2, 1.0, 4),
        cv=3,
        scoring="roc_auc",
        n_jobs=2,
        shuffle=True,
        random_state=SEED,
    )
    plot_learning_curve(
        sizes,
        train_scores,
        val_scores,
        FIG_DIR / "learning_curve_forest.png",
        "Constrained random forest learning curve",
    )

    traditional = [row for row in results if row["paradigm"] == "traditional"]
    deep = [row for row in results if row["paradigm"] == "deep"]
    best_traditional = max(traditional, key=lambda row: row["val"]["roc_auc"])
    best_deep = max(deep, key=lambda row: row["val"]["roc_auc"])
    (METRIC_DIR / "selection.json").write_text(
        json.dumps(
            {
                "rule": "Highest validation ROC-AUC within each paradigm. Thresholds were already locked on validation F1.",
                "best_traditional": best_traditional["id"],
                "best_deep": best_deep["id"],
            },
            indent=2,
        )
    )

    overlay_ids = ["E1", "E5", "E8", "E10", "E11", best_deep["id"]]
    # Keep the order stable and drop duplicates if the best deep id is already listed.
    seen = []
    for experiment_id in overlay_ids:
        if experiment_id not in seen and experiment_id in bank:
            seen.append(experiment_id)
    roc_curves = []
    pr_curves = []
    names = {row["id"]: f"{row['id']} {row['model']}" for row in results}
    for experiment_id in seen:
        roc_curves.append((names[experiment_id], y_test, bank[experiment_id]["test"]))
        pr_curves.append((names[experiment_id], y_test, bank[experiment_id]["test"]))
    roc_overlay(roc_curves, FIG_DIR / "roc_test.png")
    pr_overlay(pr_curves, float(y_test.mean()), FIG_DIR / "pr_test.png")

    for experiment_id, filename in (
        (best_traditional["id"], "confusion_best_traditional.png"),
        (best_deep["id"], "confusion_best_deep.png"),
    ):
        row = next(item for item in results if item["id"] == experiment_id)
        confusion(
            y_test,
            bank[experiment_id]["test"],
            row["thresholds_from_validation"]["f1"],
            FIG_DIR / filename,
            f"{experiment_id} test set, F1 threshold",
        )

    # Coefficients from the tuned logistic model (standardized features).
    lr_model = fitted["E5"]
    coefficients = lr_model.named_steps["clf"].coef_.ravel()
    pretty = [_pretty_feature(column) for column in columns]
    coefficient_bars(pretty, coefficients, FIG_DIR / "logreg_coefficients.png", "Tuned logistic regression")

    # Permutation importance on a validation subsample of the selected traditional model.
    best_estimator = fitted[best_traditional["id"]]
    clf = getattr(best_estimator, "named_steps", {}).get("clf", best_estimator)
    kernel_svm = type(clf).__name__ == "SVC"
    rng = np.random.default_rng(SEED)
    take = 600 if kernel_svm else min(4000, len(val))
    choice = rng.choice(len(val), size=take, replace=False)
    val_sample = matrix(val.iloc[choice], columns)
    y_sample = y_val[choice]
    importance = permutation_importance(
        best_estimator,
        val_sample,
        y_sample,
        scoring="roc_auc",
        n_repeats=4,
        random_state=SEED,
        n_jobs=2,
    )
    order = np.argsort(importance.importances_mean)[-12:]
    importance_bars(
        [_pretty_feature(columns[i]) for i in order],
        importance.importances_mean[order],
        FIG_DIR / "permutation_importance.png",
        f"Permutation importance, {best_traditional['id']}",
    )

    calibration_curves = []
    for experiment_id in ("E5", "E10", best_deep["id"]):
        if bank[experiment_id]["probabilities"]:
            calibration_curves.append((names[experiment_id], y_test, bank[experiment_id]["test"]))
    if calibration_curves:
        calibration(calibration_curves, FIG_DIR / "calibration_test.png")

    best_prob_id = best_traditional["id"] if bank[best_traditional["id"]]["probabilities"] else "E10"
    threshold_tradeoff(
        y_val,
        bank[best_prob_id]["val"],
        FIG_DIR / "threshold_tradeoff_val.png",
        f"{best_prob_id} validation precision and recall",
    )

    slices = _slice_table(test, bank[best_traditional["id"]]["test"], best_traditional["thresholds_from_validation"]["f1"])
    slices.to_csv(METRIC_DIR / "error_slices_test.csv", index=False)
    slice_recall(slices, FIG_DIR / "slice_recall_test.png", f"{best_traditional['id']} no-show recall by slice")

    policy_rows = []
    scores = bank[best_prob_id]["test"]
    thresholds = next(row for row in results if row["id"] == best_prob_id)["thresholds_from_validation"]
    y_ref = y_test
    for policy, threshold in thresholds.items():
        block = metric_block(y_ref, scores, threshold, True)
        policy_rows.append(
            {
                "model": best_prob_id,
                "policy": policy,
                "threshold_locked_on_validation": threshold,
                "precision": block["precision"],
                "recall": block["recall"],
                "f1": block["f1"],
                "overbook_cost_per_1000": cost_per_thousand(y_ref, scores, threshold, 5.0, 1.0),
                "remind_cost_per_1000": cost_per_thousand(y_ref, scores, threshold, 1.0, 3.0),
            }
        )
    pd.DataFrame(policy_rows).to_csv(METRIC_DIR / "policy_comparison_test.csv", index=False)

    table = []
    for row in results:
        table.append(
            {
                "id": row["id"],
                "model": row["model"],
                "paradigm": row["paradigm"],
                "features": row["features"],
                "hyperparameters": json.dumps(row["hyperparameters"]),
                "n_train": row["n_train"],
                "n_val": row["n_val"],
                "n_test": row["n_test"],
                "threshold": row["val"]["threshold"],
                "train_roc_auc": row["train"]["roc_auc"],
                "val_roc_auc": row["val"]["roc_auc"],
                "val_pr_auc": row["val"]["pr_auc"],
                "val_f1": row["val"]["f1"],
                "val_precision": row["val"]["precision"],
                "val_recall": row["val"]["recall"],
                "test_roc_auc": row["test"]["roc_auc"],
                "test_pr_auc": row["test"]["pr_auc"],
                "test_f1": row["test"]["f1"],
                "test_precision": row["test"]["precision"],
                "test_recall": row["test"]["recall"],
                "test_accuracy": row["test"]["accuracy"],
                "auc_gap": row["auc_gap"],
                "fit_seconds": row["fit_seconds"],
                "intent": row["intent"],
            }
        )
    pd.DataFrame(table).to_csv(METRIC_DIR / "experiment_table.csv", index=False)


def _export_models(results: list, fitted: dict) -> None:
    import joblib

    traditional = [row for row in results if row["paradigm"] == "traditional"]
    deep = [row for row in results if row["paradigm"] == "deep"]
    best_traditional = max(traditional, key=lambda row: row["val"]["roc_auc"])
    best_deep = max(deep, key=lambda row: row["val"]["roc_auc"])
    traditional_path = MODEL_DIR / "best_traditional.joblib"
    joblib.dump(
        {
            "experiment_id": best_traditional["id"],
            "model": fitted[best_traditional["id"]],
            "threshold": best_traditional["thresholds_from_validation"]["f1"],
            "metrics_test": best_traditional["test"],
        },
        traditional_path,
    )
    deep_bundle = fitted[best_deep["id"]]
    deep_path = MODEL_DIR / "best_deep.keras"
    deep_bundle["model"].save(deep_path)
    (MODEL_DIR / "best_deep_metadata.json").write_text(
        json.dumps(
            _jsonify(
                {
                    "experiment_id": best_deep["id"],
                    "kind": deep_bundle["kind"],
                    "columns": deep_bundle["columns"],
                    "threshold": best_deep["thresholds_from_validation"]["f1"],
                    "neighbourhood_to_id": deep_bundle.get("neighbourhood_to_id", {}),
                    "metrics_test": best_deep["test"],
                }
            ),
            indent=2,
        )
    )
    joblib.dump(deep_bundle["scaler"], MODEL_DIR / "best_deep_scaler.joblib")

    # Logistic coefficients are small and useful even when a tree wins.
    joblib.dump(fitted["E5"], MODEL_DIR / "logistic_regression_E5.joblib")

    DELIVERABLES.mkdir(parents=True, exist_ok=True)
    model_out = DELIVERABLES / "models"
    if model_out.exists():
        shutil.rmtree(model_out)
    model_out.mkdir(parents=True)
    for path in (
        traditional_path,
        deep_path,
        MODEL_DIR / "best_deep_metadata.json",
        MODEL_DIR / "best_deep_scaler.joblib",
        MODEL_DIR / "logistic_regression_E5.joblib",
    ):
        shutil.copy(path, model_out / path.name)
    zip_path = DELIVERABLES / "noshowappointments-raw.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(DATA_CSV, arcname=DATA_CSV.name)
    (DELIVERABLES / "UPLOAD.txt").write_text(
        "Upload this folder to Google Drive and set the link to "
        "'Anyone with the link can view'. Put that URL on the first page of the report.\n"
        "Contents: the raw dataset zip, the best traditional model (.joblib), "
        "and the best deep model (.keras) with its scaler and metadata.\n"
    )


def load_results() -> list[dict]:
    return json.loads((METRIC_DIR / "results.json").read_text())


if __name__ == "__main__":
    run_all()
