"""Paths, seeds, and split sizes shared by the notebook and the runner."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
DATA_CSV = DATA_DIR / "noshowappointments-kagglev2-may-2016.csv"
FIG_DIR = ROOT / "artifacts" / "figures"
MODEL_DIR = ROOT / "artifacts" / "models"
METRIC_DIR = ROOT / "artifacts" / "metrics"
WANDB_DIR = ROOT / "artifacts" / "wandb"
DELIVERABLES = ROOT / "deliverables"

# Public mirror of the Kaggle file released by Joni Hoppen (May 2016).
# The raw CSV is downloaded locally and is not committed.
DATA_URLS = [
    "https://raw.githubusercontent.com/CS-LEE2022/Investigate_the_No-show_appointments_Dataset/master/noshowappointments-kagglev2-may-2016.csv",
    "https://media.githubusercontent.com/media/CS-LEE2022/Investigate_the_No-show_appointments_Dataset/master/noshowappointments-kagglev2-may-2016.csv",
]

SEED = 42
TEST_SIZE = 0.15
VAL_FRACTION_OF_ALL = 0.15

# Vitória, Espírito Santo did not observe daylight saving time.
# Booking timestamps are UTC; local civil time is UTC−3.
LOCAL_UTC_OFFSET_HOURS = 3

AGE_MIN = 0
AGE_MAX = 110
