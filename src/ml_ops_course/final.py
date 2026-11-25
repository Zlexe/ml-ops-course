from pathlib import Path

import joblib
import mlflow
from sklearn.pipeline import Pipeline

from ml_ops_course.data import load_raw, split_data
from ml_ops_course.preprocessing import make_preprocessor
from ml_ops_course.train import build_classifier, evaluate, get_data_md5, get_git_sha

# Выбрано по validation ROC-AUC и PR-AUC из эксперимента lgbm_v2
BEST = {"prep": "v2", "clf": "lgbm"}
MODEL_VERSION = "v1.0.0"
MODEL_PATH = Path("models/model.joblib")


def main() -> None:
    df = load_raw()
    (X_tr, y_tr), _, (X_te, y_te) = split_data(df)

    model = Pipeline([
        ("prep", make_preprocessor(BEST["prep"])),
        ("clf", build_classifier(BEST["clf"])),
    ])
    model.fit(X_tr, y_tr)

    # test используется только здесь, один раз
    test_metrics = evaluate(model, X_te, y_te)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    mlflow.set_experiment("bank-marketing")
    with mlflow.start_run(run_name="final_lgbm_v2"):
        mlflow.log_params({"preprocessing": BEST["prep"], "model": BEST["clf"]})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.set_tags({
            "git_sha": get_git_sha(),
            "data_md5": get_data_md5(),
            "model_version": MODEL_VERSION,
        })

    print("test", {k: round(v, 4) for k, v in test_metrics.items()})
    print("saved", MODEL_PATH)


if __name__ == "__main__":
    main()
