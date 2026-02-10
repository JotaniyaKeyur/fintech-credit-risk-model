from sklearn.model_selection import train_test_split
from src.config import RANDOM_SEED

def split_data(df):
    X = df.drop("SeriousDlqin2yrs", axis=1)
    y = df["SeriousDlqin2yrs"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    return X_train, X_val, y_train, y_val
