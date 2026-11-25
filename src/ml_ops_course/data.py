from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw/bank-full.csv")
TARGET = "y"
EXPECTED_COLUMNS = [
    "age", "job", "marital", "education", "default", "balance",
    "housing", "loan", "contact", "day", "month", "duration",
    "campaign", "pdays", "previous", "poutcome", "y",
]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"В данных отсутствуют колонки: {sorted(missing)}")
    unexpected_target = set(df[TARGET].unique()) - {"yes", "no"}
    if unexpected_target:
        raise ValueError(f"Неожиданные значения в {TARGET}: {unexpected_target}")
    return df


def split_data(df: pd.DataFrame, random_state: int = 42):
    X = df.drop(columns=[TARGET])
    y = (df[TARGET] == "yes").astype(int)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=random_state
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
