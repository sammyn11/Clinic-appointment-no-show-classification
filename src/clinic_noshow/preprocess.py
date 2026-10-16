"""Load, clean, and encode the Vitória appointment table without label leakage."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

from clinic_noshow.config import (
    AGE_MAX,
    AGE_MIN,
    DATA_CSV,
    DATA_URLS,
    LOCAL_UTC_OFFSET_HOURS,
    METRIC_DIR,
    SEED,
    TEST_SIZE,
    VAL_FRACTION_OF_ALL,
)

DEMOGRAPHIC_FEATURES = [
    "age",
    "gender_male",
    "scholarship",
    "hipertension",
    "diabetes",
    "alcoholism",
    "handcap",
]
SCHEDULING_CORE = [
    "lead_days",
    "lead_days_log1p",
    "same_day",
    "sched_hour",
    "sms_received",
]
HISTORY_FEATURES = [
    "prior_appt_count",
    "prior_noshow_rate",
    "days_since_prev",
    "is_first_appt",
    "neighbourhood_freq",
    "neighbourhood_te",
]


@dataclass
class SplitData:
    """Train, validation, and test frames that share one feature schema."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    feature_sets: dict
    neighbourhood_to_id: dict
    profile: dict

    def frame(self, name: str) -> pd.DataFrame:
        return {"train": self.train, "val": self.val, "test": self.test}[name]


def download_raw(destination: Path = DATA_CSV) -> Path:
    """Download the raw CSV if it is not already on disk."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1_000_000:
        return destination
    last_error = None
    for url in DATA_URLS:
        try:
            urllib.request.urlretrieve(url, destination)
            if destination.stat().st_size > 1_000_000:
                return destination
        except Exception as exc:  # noqa: BLE001 - try the next public mirror
            last_error = exc
    raise RuntimeError(f"Could not download the appointment file: {last_error}")


def load_raw(path: Path = DATA_CSV) -> pd.DataFrame:
    """Read the source file. PatientId is an identifier, not a measurement."""
    download_raw(path)
    frame = pd.read_csv(path)
    if len(frame) < 20_000:
        raise RuntimeError(f"Raw file has only {len(frame)} rows; expected at least 20,000.")
    return frame


def _local_booking_clock(scheduled: pd.Series) -> pd.Series:
    """Shift UTC booking timestamps to Vitória civil time (UTC−3)."""
    clock = pd.to_datetime(scheduled, utc=True)
    return clock - pd.Timedelta(hours=LOCAL_UTC_OFFSET_HOURS)


def clean_appointments(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop impossible ages and booking dates, and define the no-show label.

    In the source file, No-show = Yes means the patient did not attend.
    """
    frame = raw.copy()
    n_raw = int(len(frame))
    frame["PatientId"] = pd.to_numeric(frame["PatientId"], errors="coerce").round().astype("int64")
    frame["AppointmentID"] = pd.to_numeric(frame["AppointmentID"], errors="coerce").astype("int64")
    frame["ScheduledDay"] = pd.to_datetime(frame["ScheduledDay"], utc=True)
    frame["AppointmentDay"] = pd.to_datetime(frame["AppointmentDay"], utc=True)
    frame["noshow"] = (frame["No-show"].astype(str).str.strip().str.lower() == "yes").astype(int)

    scheduled_local = _local_booking_clock(frame["ScheduledDay"])
    # AppointmentDay is a date label stored at UTC midnight, not an instant.
    appointment_date = frame["AppointmentDay"].dt.floor("D")
    booking_date = scheduled_local.dt.floor("D")
    frame["lead_days"] = (appointment_date - booking_date).dt.days.astype(int)
    frame["sched_hour"] = scheduled_local.dt.hour.astype(int)
    frame["appt_dow"] = frame["AppointmentDay"].dt.dayofweek.astype(int)
    frame["appt_month"] = frame["AppointmentDay"].dt.month.astype(int)

    n_bad_age = int((~frame["Age"].between(AGE_MIN, AGE_MAX)).sum())
    n_neg_lead = int((frame["lead_days"] < 0).sum())
    n_dup = int(frame["AppointmentID"].duplicated().sum())

    frame = frame.loc[frame["Age"].between(AGE_MIN, AGE_MAX)].copy()
    frame = frame.loc[frame["lead_days"] >= 0].copy()
    frame = frame.drop_duplicates(subset=["AppointmentID"], keep="first")
    frame = frame.sort_values(
        ["PatientId", "AppointmentDay", "ScheduledDay", "AppointmentID"]
    ).reset_index(drop=True)

    frame["age"] = frame["Age"].astype(int)
    frame["gender_male"] = (frame["Gender"].astype(str).str.upper() == "M").astype(int)
    frame["scholarship"] = frame["Scholarship"].astype(int)
    frame["hipertension"] = frame["Hipertension"].astype(int)
    frame["diabetes"] = frame["Diabetes"].astype(int)
    frame["alcoholism"] = frame["Alcoholism"].astype(int)
    frame["handcap"] = frame["Handcap"].astype(int)
    frame["sms_received"] = frame["SMS_received"].astype(int)
    frame["neighbourhood"] = frame["Neighbourhood"].astype(str).str.strip()
    frame["same_day"] = (frame["lead_days"] == 0).astype(int)
    frame["lead_days_log1p"] = np.log1p(frame["lead_days"])

    quality = {
        "n_raw": n_raw,
        "n_clean": int(len(frame)),
        "n_removed_age": n_bad_age,
        "n_removed_negative_lead": n_neg_lead,
        "n_duplicate_appointment_id": n_dup,
        "n_patients": int(frame["PatientId"].nunique()),
        "noshow_rate": float(frame["noshow"].mean()),
        "n_noshow": int(frame["noshow"].sum()),
        "n_attended": int((frame["noshow"] == 0).sum()),
        "appointment_min": str(frame["AppointmentDay"].min().date()),
        "appointment_max": str(frame["AppointmentDay"].max().date()),
        "n_neighbourhoods": int(frame["neighbourhood"].nunique()),
        "sms_rate_same_day": float(frame.loc[frame["same_day"] == 1, "sms_received"].mean()),
        "sms_rate_later": float(frame.loc[frame["same_day"] == 0, "sms_received"].mean()),
        "handcap_max": int(frame["handcap"].max()),
        "female_rate": float((frame["gender_male"] == 0).mean()),
        "scholarship_rate": float(frame["scholarship"].mean()),
        "hypertension_rate": float(frame["hipertension"].mean()),
        "diabetes_rate": float(frame["diabetes"].mean()),
        "alcoholism_rate": float(frame["alcoholism"].mean()),
        "same_day_rate": float(frame["same_day"].mean()),
        "median_lead_days": float(frame["lead_days"].median()),
        "scheduled_min": str(frame["ScheduledDay"].min().date()),
        "scheduled_max": str(frame["ScheduledDay"].max().date()),
        "columns_source": list(raw.columns),
    }
    return frame, quality


def add_patient_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Prior attendance uses only earlier visits of the same patient.

    The current label is removed from the running sum, so a row cannot see its
    own outcome. Later visits are not used either, because the frame is sorted
    in time before the cumulative count.
    """
    out = frame.copy()
    grouped = out.groupby("PatientId", sort=False)
    out["prior_appt_count"] = grouped.cumcount().astype(int)
    out["prior_noshow_count"] = (grouped["noshow"].cumsum() - out["noshow"]).astype(int)
    out["is_first_appt"] = (out["prior_appt_count"] == 0).astype(int)
    rate = np.full(len(out), np.nan, dtype=float)
    has_history = out["prior_appt_count"].to_numpy() > 0
    rate[has_history] = (
        out.loc[has_history, "prior_noshow_count"].to_numpy()
        / out.loc[has_history, "prior_appt_count"].to_numpy()
    )
    out["prior_noshow_rate"] = rate
    previous_day = grouped["AppointmentDay"].shift(1)
    gap = (out["AppointmentDay"] - previous_day).dt.days
    out["days_since_prev"] = gap
    if (out["prior_noshow_count"] < 0).any():
        raise RuntimeError("History feature included a future or current no-show.")
    return out


def _patient_strata(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per patient, stratified by that patient's observed no-show rate."""
    rates = frame.groupby("PatientId")["noshow"].mean()
    # 0 never missed, 1 mixed, 2 missed at least half of recorded visits.
    strata = np.where(rates == 0, 0, np.where(rates < 0.5, 1, 2))
    return pd.DataFrame({"PatientId": rates.index.to_numpy(), "stratum": strata})


def split_by_patient(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep every visit of a patient inside a single split.

    A row-wise random split would let a later training visit use an earlier
    test visit's outcome as history, which leaks the test label.
    """
    patients = _patient_strata(frame)
    ids = patients["PatientId"].to_numpy()
    strata = patients["stratum"].to_numpy()
    outer = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    trainval_idx, test_idx = next(outer.split(ids, strata))
    inner_size = VAL_FRACTION_OF_ALL / (1.0 - TEST_SIZE)
    inner = StratifiedShuffleSplit(n_splits=1, test_size=inner_size, random_state=SEED)
    train_idx, val_idx = next(inner.split(ids[trainval_idx], strata[trainval_idx]))

    train_ids = set(ids[trainval_idx][train_idx])
    val_ids = set(ids[trainval_idx][val_idx])
    test_ids = set(ids[test_idx])
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Patient identifiers overlap across splits.")

    train = frame.loc[frame["PatientId"].isin(train_ids)].copy()
    val = frame.loc[frame["PatientId"].isin(val_ids)].copy()
    test = frame.loc[frame["PatientId"].isin(test_ids)].copy()
    return train, val, test


def _smooth_means(keys: pd.Series, target: pd.Series, smooth: float) -> tuple[pd.Series, float]:
    global_mean = float(target.mean())
    stats = pd.DataFrame({"key": keys.to_numpy(), "y": target.to_numpy()}).groupby("key")["y"].agg(["sum", "count"])
    encoded = (stats["sum"] + smooth * global_mean) / (stats["count"] + smooth)
    return encoded, global_mean


def _apply_encode(keys: pd.Series, mapping: pd.Series, default: float) -> np.ndarray:
    return keys.map(mapping).fillna(default).to_numpy(dtype=float)


def add_train_only_encodings(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    smooth: float = 30.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Frequency and target encodings are fit on training patients only.

    Target encoding inside the training split is out-of-fold. Using the same
    rows to compute the neighbourhood mean and to fit the classifier would let
    each row help predict itself.
    """
    train = train.copy()
    val = val.copy()
    test = test.copy()
    global_rate = float(train["noshow"].mean())

    freq = train["neighbourhood"].value_counts(normalize=True)
    for part in (train, val, test):
        part["neighbourhood_freq"] = part["neighbourhood"].map(freq).fillna(0.0).astype(float)

    oof = np.full(len(train), global_rate, dtype=float)
    # Group by patient so a visit does not help estimate the code applied to itself.
    try:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    except TypeError:
        splitter = StratifiedGroupKFold(n_splits=5)
    for tr_idx, va_idx in splitter.split(train, train["noshow"].to_numpy(), train["PatientId"].to_numpy()):
        mapping, mean = _smooth_means(
            train.iloc[tr_idx]["neighbourhood"],
            train.iloc[tr_idx]["noshow"],
            smooth,
        )
        oof[va_idx] = _apply_encode(train.iloc[va_idx]["neighbourhood"], mapping, mean)
    train["neighbourhood_te"] = oof

    full_mapping, full_mean = _smooth_means(train["neighbourhood"], train["noshow"], smooth)
    val["neighbourhood_te"] = _apply_encode(val["neighbourhood"], full_mapping, full_mean)
    test["neighbourhood_te"] = _apply_encode(test["neighbourhood"], full_mapping, full_mean)

    # First visits have no personal history. Fill the rate with the training
    # base rate and let is_first_appt tell the model that the rate is a prior,
    # not an observation. Days since the previous visit use the training median.
    gap_median = float(train.loc[train["is_first_appt"] == 0, "days_since_prev"].median())
    for part in (train, val, test):
        part["prior_noshow_rate"] = part["prior_noshow_rate"].fillna(global_rate)
        part["days_since_prev"] = part["days_since_prev"].fillna(gap_median)

    neighbourhoods = sorted(train["neighbourhood"].unique())
    neighbourhood_to_id = {name: i + 1 for i, name in enumerate(neighbourhoods)}
    for part in (train, val, test):
        part["neighbourhood_id"] = part["neighbourhood"].map(neighbourhood_to_id).fillna(0).astype(int)

    dow_columns = [f"dow_{day}" for day in range(7)]
    for part in (train, val, test):
        for day, column in enumerate(dow_columns):
            part[column] = (part["appt_dow"] == day).astype(int)

    # Monday is the reference level for linear models.
    dow_features = [f"dow_{day}" for day in range(1, 7)]
    feature_sets = {
        "demographic": DEMOGRAPHIC_FEATURES,
        "scheduling": DEMOGRAPHIC_FEATURES + SCHEDULING_CORE + dow_features,
        "full": DEMOGRAPHIC_FEATURES + SCHEDULING_CORE + dow_features + HISTORY_FEATURES,
        "dow": dow_features,
    }
    meta = {
        "neighbourhood_to_id": neighbourhood_to_id,
        "train_noshow_rate": global_rate,
        "days_since_prev_median": gap_median,
        "n_neighbourhood_ids": len(neighbourhood_to_id) + 1,
    }
    return train, val, test, {"feature_sets": feature_sets, **meta}


def prepare_dataset(path: Path = DATA_CSV) -> SplitData:
    """Full reproducible path from the raw CSV to model matrices."""
    raw = load_raw(path)
    cleaned, quality = clean_appointments(raw)
    featured = add_patient_history(cleaned)
    train, val, test = split_by_patient(featured)
    train, val, test, meta = add_train_only_encodings(train, val, test)
    profile = {
        **quality,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "n_train_patients": int(train["PatientId"].nunique()),
        "n_val_patients": int(val["PatientId"].nunique()),
        "n_test_patients": int(test["PatientId"].nunique()),
        "train_noshow_rate": float(train["noshow"].mean()),
        "val_noshow_rate": float(val["noshow"].mean()),
        "test_noshow_rate": float(test["noshow"].mean()),
        "train_noshow_rate_encoding": meta["train_noshow_rate"],
        "days_since_prev_median": meta["days_since_prev_median"],
        "feature_sets": meta["feature_sets"],
    }
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    (METRIC_DIR / "data_profile.json").write_text(json.dumps(profile, indent=2))
    return SplitData(
        train=train,
        val=val,
        test=test,
        feature_sets=meta["feature_sets"],
        neighbourhood_to_id=meta["neighbourhood_to_id"],
        profile=profile,
    )


def matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return frame.loc[:, columns].to_numpy(dtype=np.float64)


def labels(frame: pd.DataFrame) -> np.ndarray:
    return frame["noshow"].to_numpy(dtype=np.int64)
