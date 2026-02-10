import pandas as pd
from src.config import RAW_DATA_DIR, DATA_FILE

def load_data():
    file_path = RAW_DATA_DIR / DATA_FILE

    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found")

    df = pd.read_csv(file_path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    return df
