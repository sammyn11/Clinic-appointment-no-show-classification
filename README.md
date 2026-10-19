# Clinic appointment no-show classification

This project predicts whether a booked outpatient visit will be missed, so a clinic can decide who should get a reminder and which slots are safe to double-book. The question follows a mission to digitise appointment management in African medical institutions: cut idle clinician time without filling the waiting room.

The data are not from an African facility. Open extracts at this size were not available, so the model is trained on the public Vitória (Brazil) appointment file: 110,527 visits, binary no-show label. The report treats that geography as a limit on what the thresholds mean.

## What is in the repository

- `src/clinic_noshow/` — download, cleaning, patient-level split, classical models, Keras models, evaluation, figures.
- `notebooks/appointment_noshow_classification.ipynb` — the narrative notebook. It loads the saved experiment table and figures, and it can refit everything.
- `report/main.tex` and `report/main.pdf` — IEEE-style report.
- `artifacts/metrics/` — experiment table, slices, policies, data profile.
- `artifacts/figures/` — learning curves, ROC, precision–recall, confusion matrices, calibration.
- `artifacts/wandb/` — Weights & Biases offline runs, one per experiment.
- `artifacts/models/` — selected traditional model (`.joblib`) and selected Keras model (`.keras`).
- `presentation/speaker_script.md` — timed notes for the oral recording.
- `tests/test_preprocess.py` — checks that history features do not include the current label, and that patient splits do not overlap.

The raw CSV is not committed.

## Run it

Python 3.11 or 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
python tests/test_preprocess.py
python -m clinic_noshow.experiments
```

The experiment module downloads the public CSV into `data/raw/` if it is missing, then fits all sixteen runs. On a CPU laptop that pass is a few minutes. It overwrites `artifacts/`.

Open the notebook from the repository root:

```bash
jupyter notebook notebooks/appointment_noshow_classification.ipynb
```

Leave `RETRAIN = False` to read the saved metrics. Set it to `True` to fit again.

The report and notebook commentary were drafted with an assistant from the measured results. Rewrite them in your own words, and be ready to defend every number in the experiment table, before you hand the work in.
