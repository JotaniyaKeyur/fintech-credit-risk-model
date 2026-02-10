from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model paths
MODELS_DIR = BASE_DIR / "models"

# Reports
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Data file
DATA_FILE = "give_me_some_credit.csv"

# Random seed
RANDOM_SEED = 42

# Target column
TARGET_COLUMN = "SeriousDlqin2yrs"

# xgboost tuned parameters
BEST_PARAMS = {
    "booster": "gbtree",
    "scale_pos_weight": 1.0844731956681208,
    "lambda": 0.007863744044452415,
    "alpha": 1.3708881631313705,
    "subsample": 0.7792604678854678,
    "colsample_bytree": 0.8536059725156923,
    "colsample_bynode": 0.7889610964707088,
    "max_depth": 7,
    "min_child_weight": 5,
    "gamma": 0.059826987594721145,
    "eta": 0.19998789278709772,
}
