# 🏦 Loan Default Prediction System (Production ML Project)

A **Loan Default Risk Prediction System** built using a
**production-level Machine Learning pipeline**.

This project predicts whether a borrower is likely to default within 2
years using structured financial data from the **Give Me Some Credit
(Kaggle)** dataset.

> Built as a real-world ML engineering practice project to demonstrate
> proper pipeline design, data leakage prevention, imbalanced learning,
> and model deployment readiness.

------------------------------------------------------------------------

## 📌 Features

-   Production-level ML folder structure
-   MICE (IterativeImputer) for advanced missing value handling
-   Custom feature engineering transformer
-   SMOTE for class imbalance handling
-   XGBoost classifier
-   Proper Train/Validation split (no data leakage)
-   ROC-AUC evaluation metric
-   Logging system
-   Model saving using Joblib
-   Batch prediction support
-   Single customer prediction support

------------------------------------------------------------------------

## 🧠 Target Variable

`SeriousDlqin2yrs`

-   0 → No Default\
-   1 → Default

------------------------------------------------------------------------

## 🛠 Tech Stack

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Imbalanced-learn (SMOTE)
-   XGBoost
-   Joblib
-   Logging

> No AutoML\
> No Black-box wrappers\
> Fully controlled ML pipeline

------------------------------------------------------------------------

## 📂 Project Structure

    loan_default_prediction/
    │
    ├── data/
    │   └── raw/
    │       └── cs-training.csv
    │
    ├── models/
    │   └── loan_default_pipeline.pkl
    │
    ├── src/
    │   ├── config.py
    │   ├── logging_config.py
    │
    │   ├── data/
    │   │   ├── loader.py
    │   │   └── split.py
    │
    │   ├── features/
    │   │   └── feature_engineering.py
    │
    │   ├── pipelines/
    │   │   └── training_pipeline.py
    │
    │   └── evaluation/
    │       └── metrics.py
    │
    ├── train.py
    ├── test.py
    ├── predict_single.py
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## 🔄 ML Pipeline Flow

    IterativeImputer (MICE)
            ↓
    Feature Engineering
            ↓
    SMOTE (training only)
            ↓
    XGBoost Classifier

✔ Train/Validation split happens BEFORE fitting\
✔ No data leakage\
✔ Full preprocessing inside pipeline\
✔ Production-ready model artifact

------------------------------------------------------------------------

## 🚀 How to Run This Project

### 1️⃣ Create Virtual Environment

``` bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate    # Windows
```

### 2️⃣ Install Dependencies

``` bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model

``` bash
python train.py
```

This will: - Load dataset - Split train & validation data - Train full
ML pipeline - Evaluate performance - Save model to:

    models/loan_default_pipeline.pkl

------------------------------------------------------------------------

### 4️⃣ Run Batch Predictions

``` bash
python test.py
```

------------------------------------------------------------------------

### 5️⃣ Run Single Customer Prediction

``` bash
python predict_single.py
```

Modify the input dictionary inside the script to test individual
customer risk.

------------------------------------------------------------------------

## 📊 Model Evaluation Metrics

-   Accuracy
-   ROC-AUC (Primary Metric)
-   Precision
-   Recall
-   F1 Score

ROC-AUC is prioritized due to class imbalance.

------------------------------------------------------------------------

## 📈 Why This Project Is Valuable

-   Demonstrates real-world ML system design
-   Shows understanding of imbalanced classification
-   Implements advanced imputation (MICE)
-   Prevents data leakage properly
-   Clean modular architecture
-   Deployment-ready structure

This reflects how ML systems are built in fintech environments.

------------------------------------------------------------------------

## 🔮 Future Improvements

-   Cross-validation support
-   Hyperparameter tuning (Optuna)
-   MLflow experiment tracking
-   FastAPI model deployment
-   Dockerization
-   CI/CD integration

------------------------------------------------------------------------

## 📚 Dataset

**Give Me Some Credit -- Kaggle (2011)**\
Binary classification problem for consumer credit risk modeling.

------------------------------------------------------------------------

## 👨‍💻 Author

Built as a portfolio project to demonstrate:

**Machine Learning • Feature Engineering • Imbalanced Learning •
Production ML Design**
