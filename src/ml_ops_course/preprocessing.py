import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

CATEGORICAL = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "poutcome",
]
NUMERIC_V1 = ["age", "balance", "day", "campaign", "pdays", "previous"]
NUMERIC_V2 = ["age", "balance", "day", "campaign", "previous", "previously_contacted"]


def drop_leakage(X: pd.DataFrame) -> pd.DataFrame:
    # duration известна только после звонка, в модели её использовать нельзя
    return X.drop(columns=["duration"])


def add_previously_contacted(X: pd.DataFrame) -> pd.DataFrame:
    # pdays = -1 означает, что клиента ранее не звонили
    X = X.assign(previously_contacted=(X["pdays"] != -1).astype(int))
    return X.drop(columns=["pdays"])


def _build(numeric: list[str], feature_fn) -> Pipeline:
    column_tf = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline(
        steps=[
            ("features", FunctionTransformer(feature_fn)),
            ("columns", column_tf),
        ]
    )


def make_preprocessor(version: str) -> Pipeline:
    if version == "v1":
        return _build(NUMERIC_V1, drop_leakage)
    if version == "v2":
        def v2_features(X):
            return add_previously_contacted(drop_leakage(X))
        return _build(NUMERIC_V2, v2_features)
    raise ValueError(f"Неизвестная версия препроцессинга: {version}")
