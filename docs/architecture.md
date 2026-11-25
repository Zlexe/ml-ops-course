# Архитектура

Схема ML-процесса от данных до инференса.

```mermaid
flowchart LR
  A[UCI bank-full.csv] -->|dvc add / dvc push| B[(DVC remote)]
  A --> C[data.py: загрузка и split 60/20/20]
  C --> D[preprocessing.py: v1 / v2]
  D --> E[train.py: 3 эксперимента]
  E -->|параметры, метрики, git SHA, data md5| F[(MLflow)]
  E --> G[final.py: lgbm_v2, оценка на test]
  G --> H[models/model.joblib]
  H -->|dvc add / dvc push| B
  H --> I[FastAPI: /health, /predict]
  I --> J[Docker image ml-ops-course:1.0.0]
```

## Компоненты

- **Данные:** `data/raw/bank-full.csv` версионируется DVC, в Git хранится только `.dvc`-файл с md5.
- **Эксперименты:** MLflow, каждый запуск логирует параметры, метрики validation, git SHA и md5 датасета.
- **Модель:** `models/model.joblib` содержит весь `Pipeline` (препроцессинг и классификатор), поэтому сервис применяет те же преобразования, что и при обучении.
- **Сервис:** FastAPI, модель загружается один раз при старте через `lifespan`, путь и версия берутся из переменных окружения `MODEL_PATH` и `MODEL_VERSION`.
- **Контейнер:** `python:3.12.14-slim-bookworm`, пользователь `appuser`, зависимости из `uv.lock`.
