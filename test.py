import joblib
import pandas as pd
from src.config import MODELS_DIR

def main():
    model_path = MODELS_DIR / "loan_default_pipeline.pkl"
    pipeline = joblib.load(model_path)

    print("Model loaded successfully.\n")

    data = {
        "RevolvingUtilizationOfUnsecuredLines": 0.957151,
        "age": 49,
        "NumberOfTime30-59DaysPastDueNotWorse": 2,
        "DebtRatio": 0.802982,
        "MonthlyIncome": 63588.0,
        "NumberOfOpenCreditLinesAndLoans": 5,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2.0
    }

    # convert to DataFrame 
    input_df = pd.DataFrame([data])

    # predict probability
    proba = pipeline.predict_proba(input_df)[0][1]

    # custom threshold
    threshold = 0.469
    prediction = int(proba >= threshold)

    print("Default Probability:", round(proba, 4))
    print("Predicted Default:", prediction)

    if prediction == 1:
        print("High Risk: Likely to Default")
    else:
        print("Low Risk: Unlikely to Default")

if __name__ == "__main__":
    main()
