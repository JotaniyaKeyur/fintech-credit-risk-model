import xgboost as xgb
from src.config import RANDOM_SEED, BEST_PARAMS

def get_model():
    params = {
        **BEST_PARAMS,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": RANDOM_SEED,
        "tree_method": "hist"
    }

    return xgb.XGBClassifier(**params)
