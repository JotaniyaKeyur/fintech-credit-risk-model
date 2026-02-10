import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert numpy array back to DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns)

        X = X.copy()

        X['TotalPastDue'] = (X['NumberOfTime30-59DaysPastDueNotWorse'] + X['NumberOfTime60-89DaysPastDueNotWorse'] + X['NumberOfTimes90DaysLate'])
        X['DebtRatioPerDependent'] = X['DebtRatio'] / (X['NumberOfDependents'] + 1)
        X['UtilizationPerLine'] = (X['RevolvingUtilizationOfUnsecuredLines'] / (X['NumberOfOpenCreditLinesAndLoans'] + 1))
        X['IncomeDebtRatio'] = X['MonthlyIncome'] / (X['DebtRatio'] + 1)
        X['HasDependents'] = (X['NumberOfDependents'] > 0).astype(int)
        X['HighDebtRatio'] = (X['DebtRatio'] > 1).astype(int)
        X['HighUtilization'] = (X['RevolvingUtilizationOfUnsecuredLines'] > 0.8).astype(int)
        return X
