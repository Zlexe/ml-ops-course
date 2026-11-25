import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("MODEL_PATH", "models/model.joblib")
os.environ.setdefault("MODEL_VERSION", "v1.0.0")

from ml_ops_course.main import app  # noqa: E402

VALID = {
    "age": 41, "job": "technician", "marital": "married", "education": "secondary",
    "default": "no", "balance": 1270, "housing": "yes", "loan": "no",
    "contact": "unknown", "day": 5, "month": "may", "campaign": 1,
    "pdays": -1, "previous": 0, "poutcome": "unknown",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_valid(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["label"] in {"yes", "no"}


def test_predict_wrong_type_returns_422(client):
    bad = {**VALID, "age": "сорок"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_missing_field_returns_422(client):
    incomplete = {k: v for k, v in VALID.items() if k != "job"}
    r = client.post("/predict", json=incomplete)
    assert r.status_code == 422


def test_model_version_in_response(client):
    r = client.post("/predict", json=VALID)
    assert r.json()["model_version"] == "v1.0.0"
    assert client.get("/health").json()["model_version"] == "v1.0.0"
