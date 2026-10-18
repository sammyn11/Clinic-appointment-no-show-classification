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

## Google Colab

Colab receives the notebook file and nothing else. The training code is `src/clinic_noshow` in this repository, so opening the `.ipynb` on its own raises `ModuleNotFoundError: clinic_noshow`.

Before you submit, paste your public GitHub URL into `GITHUB_REPO_URL` in the first code cell. The facilitator runs that cell and Colab clones the repository, including `src/`, the saved figures, and the experiment table. They then set `RETRAIN = True` and choose **Runtime → Run all** when they want to train rather than read the saved results.

Google Drive can supply the same files. It does not happen through **Mount Drive** alone: that shows the facilitator’s own My Drive, and a folder shared with them stays invisible until they add a shortcut or copy it. Two Drive options are in the same cell:

1. Zip the repository (`git archive -o clinic-noshow-code.zip HEAD` from this folder). Upload that zip to the shared Drive folder, set it to “Anyone with the link can view”, and paste the file link into `DRIVE_CODE_ZIP_URL`. The cell downloads the zip. The raw CSV is still fetched by the training code, so leave `data/raw/` out of the zip.
2. If the project folder already sits in the facilitator’s My Drive, set `DRIVE_PROJECT_FOLDER` to that folder name. The cell mounts Drive and uses it.

Fill in at least `GITHUB_REPO_URL`. The Drive lines can stay empty.

## Result in one paragraph

An unweighted histogram gradient booster is the selected traditional model (test ROC-AUC 0.748). A Keras network with a neighbourhood embedding is the selected deep model (test ROC-AUC 0.743). Lead time dominates. An unrestricted random forest memorises (training ROC-AUC 0.999) and then loses on new patients. Class weights hurt calibration. The F1 threshold is a reminder operating point, not an overbooking rule.

## Before you submit

Three things are not done, because they need your accounts and your face.

1. **GitHub.** Put the three links on the first page of `report/main.tex` (they are marked `REPLACE_WITH_...`), recompile with `pdflatex`, and push this project to the repository you want marked.
2. **Google Drive.** Zip up is already at `deliverables/noshowappointments-raw.zip`. Model files are in `deliverables/models/`. Upload that folder and set sharing to “Anyone with the link can view.” Paste the link into the report.
3. **Video.** Record 5–10 minutes with the camera on, using `presentation/speaker_script.md`. Explain the problem, the Vitória file and why it is not an African extract, the patient split, why the unrestricted forest overfits, and why the F1 threshold is the wrong rule for overbooking.

The course also caps AI-detected prose. The report and notebook commentary were drafted with an assistant from the measured results. Rewrite them in your own words, and be ready to defend every number in the experiment table, before you hand the work in.
