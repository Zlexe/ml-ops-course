import subprocess

import mlflow
import yaml
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from ml_ops_course.data import load_raw, split_data
from ml_ops_course.preprocessing import make_preprocessor

# (название, версия препроцессинга, модель)
EXPERIMENTS = [
    ("logreg_v1", "v1", "logreg"),
    ("rf_v1", "v1", "rf"),
    ("lgbm_v2", "v2", "lgbm"),
]


def build_classifier(name: str):
    if name == "logreg":
        return LogisticRegression(max_iter=1000, random_state=42)
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    if name == "lgbm":
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)
    raise ValueError(f"Неизвестная модель: {name}")


def get_git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def get_data_md5() -> str:
    with open("data/raw/bank-full.csv.dvc") as f:
        return yaml.safe_load(f)["outs"][0]["md5"]


def evaluate(model, X, y, threshold: float = 0.5) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "f1": f1_score(y, pred),
    }


def run_experiments() -> None:
    df = load_raw()
    (X_tr, y_tr), (X_va, y_va), _ = split_data(df)
    mlflow.set_experiment("bank-marketing")

    for run_name, prep_version, clf_name in EXPERIMENTS:
        with mlflow.start_run(run_name=run_name):
            model = Pipeline([
                ("prep", make_preprocessor(prep_version)),
                ("clf", build_classifier(clf_name)),
            ])
            model.fit(X_tr, y_tr)
            metrics = evaluate(model, X_va, y_va)

            mlflow.log_params({"preprocessing": prep_version, "model": clf_name})
            mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
            mlflow.set_tags({"git_sha": get_git_sha(), "data_md5": get_data_md5()})

            print(run_name, {k: round(v, 4) for k, v in metrics.items()})


if __name__ == "__main__":
    run_experiments()
