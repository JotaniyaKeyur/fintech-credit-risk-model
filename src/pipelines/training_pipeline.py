from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.features.feature_engineering import FeatureEngineer
from src.models.xgb_model import get_model
from src.config import RANDOM_SEED

def build_pipeline(feature_columns):
    pipeline = ImbPipeline(steps=[
        ("imputer", IterativeImputer(
            random_state=RANDOM_SEED,
            max_iter=20,
            initial_strategy="median",
            skip_complete=True
        )),
        ("feature_engineering", FeatureEngineer(columns=feature_columns)),
        ("smote", SMOTE(random_state=RANDOM_SEED)),
        ("model", get_model())
    ])
    return pipeline
