"""Checks for label leakage and the patient-level split. Not a modeling dataset."""

import numpy as np
import pandas as pd

from clinic_noshow.preprocess import add_patient_history, clean_appointments, split_by_patient


def _raw_rows():
    return pd.DataFrame(
        {
            "PatientId": [1, 1, 1, 2, 2],
            "AppointmentID": [10, 11, 12, 20, 21],
            "Gender": ["F", "F", "F", "M", "M"],
            "ScheduledDay": [
                "2016-04-01T15:00:00Z",
                "2016-04-10T15:00:00Z",
                "2016-04-20T18:00:00Z",
                "2016-05-01T12:00:00Z",
                "2016-05-02T12:00:00Z",
            ],
            "AppointmentDay": [
                "2016-04-01T00:00:00Z",
                "2016-04-20T00:00:00Z",
                "2016-04-20T00:00:00Z",
                "2016-05-01T00:00:00Z",
                "2016-05-08T00:00:00Z",
            ],
            "Age": [30, 30, 30, 50, -1],
            "Neighbourhood": ["A", "A", "A", "B", "B"],
            "Scholarship": [0, 0, 0, 1, 1],
            "Hipertension": [0, 0, 0, 1, 1],
            "Diabetes": [0, 0, 0, 0, 0],
            "Alcoholism": [0, 0, 0, 0, 0],
            "Handcap": [0, 0, 0, 2, 2],
            "SMS_received": [0, 1, 0, 0, 1],
            "No-show": ["No", "Yes", "No", "No", "Yes"],
        }
    )


def test_history_does_not_include_the_current_label():
    cleaned, quality = clean_appointments(_raw_rows())
    assert quality["n_removed_age"] == 1
    featured = add_patient_history(cleaned)
    patient = featured.loc[featured["PatientId"] == 1].sort_values("AppointmentID")
    assert patient["prior_noshow_count"].tolist() == [0, 0, 1]
    assert patient["noshow"].tolist() == [0, 1, 0]
    # Same-calendar-day afternoon booking is lead time 0, not a negative day.
    same_day = patient.loc[patient["AppointmentID"] == 12, "lead_days"].iloc[0]
    assert same_day == 0


def test_patient_splits_are_disjoint():
    rng = np.random.default_rng(0)
    n = 400
    patients = rng.integers(1, 80, size=n)
    frame = pd.DataFrame(
        {
            "PatientId": patients,
            "noshow": rng.integers(0, 2, size=n),
            "AppointmentDay": pd.date_range("2016-04-01", periods=n, tz="UTC"),
        }
    )
    train, val, test = split_by_patient(frame)
    ids = [
        set(train["PatientId"]),
        set(val["PatientId"]),
        set(test["PatientId"]),
    ]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    assert len(train) + len(val) + len(test) == n


if __name__ == "__main__":
    test_history_does_not_include_the_current_label()
    test_patient_splits_are_disjoint()
    print("ok")
