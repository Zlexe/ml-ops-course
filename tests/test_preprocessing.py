import pandas as pd

from ml_ops_course.preprocessing import add_previously_contacted, drop_leakage, make_preprocessor


def _sample() -> pd.DataFrame:
    return pd.DataFrame([
        {"age": 30, "job": "admin.", "marital": "single", "education": "secondary",
         "default": "no", "balance": 100, "housing": "no", "loan": "no",
         "contact": "cellular", "day": 3, "month": "jun", "duration": 200,
         "campaign": 1, "pdays": -1, "previous": 0, "poutcome": "unknown"},
        {"age": 50, "job": "services", "marital": "married", "education": "tertiary",
         "default": "no", "balance": 900, "housing": "yes", "loan": "yes",
         "contact": "unknown", "day": 20, "month": "may", "duration": 80,
         "campaign": 2, "pdays": 120, "previous": 3, "poutcome": "success"},
    ])


def test_drop_leakage_removes_duration():
    assert "duration" not in drop_leakage(_sample()).columns


def test_drop_leakage_tolerates_missing_duration():
    X = _sample().drop(columns=["duration"])
    assert "duration" not in drop_leakage(X).columns


def test_previously_contacted_flag():
    out = add_previously_contacted(_sample())
    assert list(out["previously_contacted"]) == [0, 1]
    assert "pdays" not in out.columns


def test_preprocessor_outputs_fixed_width():
    X = _sample()
    y = pd.Series([0, 1])
    for version in ["v1", "v2"]:
        Xt = make_preprocessor(version).fit(X, y).transform(X)
        assert Xt.shape[0] == 2
