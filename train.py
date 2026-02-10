import joblib
from pathlib import Path

from src.logging_config import setup_logging
from src.data.loader import load_data
from src.data.split import split_data
from src.pipelines.training_pipeline import build_pipeline
from src.evaluation.metrics import evaluate
from src.config import MODELS_DIR

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def main():

    logger = setup_logging()
    logger.info("Starting training pipeline")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    X_train, X_val, y_train, y_val = split_data(df)

    pipeline = build_pipeline(feature_columns=X_train.columns)

    logger.info("Training model...")
    pipeline.fit(X_train, y_train)

    logger.info("Evaluating model...")
    results = evaluate(pipeline, X_val, y_val)

    logger.info(f"Validation Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Validation ROC-AUC: {results['roc_auc']:.4f}")

    model_path = MODELS_DIR / "loan_default_pipeline.pkl"
    joblib.dump(pipeline, model_path)

    logger.info(f"Model saved to {model_path}")
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()
