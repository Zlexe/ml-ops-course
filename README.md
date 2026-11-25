# Bank Marketing: воспроизводимый ML-сервис

Курсовой проект по дисциплине «Операционализация моделей машинного обучения».
Задача: прогноз согласия клиента банка на срочный депозит (бинарная классификация).

## Требования

- macOS / Linux, Python 3.12 (управляется через `uv`)
- [uv](https://docs.astral.sh/uv/) 0.13+
- Docker 29+
- Git, DVC (ставится через `uv`)

## Восстановление окружения

```bash
git clone https://github.com/Zlexe/ml-ops-course.git
cd ml-ops-course
git checkout v1.0.0
uv sync --frozen
```

## Данные и модель

Датасет и модель версионируются через DVC. Удалённое хранилище настроено в `.dvc/config`.

```bash
uv run dvc pull
```

Проверка контрольной суммы датасета: файл `data/raw/bank-full.csv.dvc` содержит md5 `e6b0ca77f3f200ec5428e04dd104da53`.

## Тесты

```bash
uv run pytest --cov=src/ml_ops_course
```

## Обучение и итоговая модель

Три эксперимента на validation (логирование в MLflow):

```bash
uv run python -m ml_ops_course.train
uv run mlflow ui   # интерфейс на http://127.0.0.1:5000
```

Итоговая модель (лучший эксперимент `lgbm_v2`) обучается и оценивается на test один раз:

```bash
uv run python -m ml_ops_course.final
```

## Запуск сервиса локально

```bash
cp .env.example .env
export MODEL_PATH=models/model.joblib MODEL_VERSION=v1.0.0
uv run uvicorn ml_ops_course.main:app --reload
```

Документация API: http://127.0.0.1:8000/docs

## Docker

```bash
docker build -t ml-ops-course:1.0.0 .
docker run --rm -p 8000:8000 ml-ops-course:1.0.0
```

Контейнер работает от пользователя `appuser` (uid 1000).

## Примеры запросов

Проверка готовности:

```bash
curl -s localhost:8000/health
```

Ответ: `{"status":"ok","model_version":"v1.0.0","model_loaded":true}`

Предсказание:

```bash
curl -s -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":41,"job":"technician","marital":"married","education":"secondary","default":"no","balance":1270,"housing":"yes","loan":"no","contact":"unknown","day":5,"month":"may","campaign":1,"pdays":-1,"previous":0,"poutcome":"unknown"}'
```

Ответ: `{"probability":0.0261,"label":"no","model_version":"v1.0.0"}`

Некорректный вход возвращает `422 Unprocessable Entity` с описанием ошибок, а не `500`.

## Структура репозитория

configs/ конфигурация (зарезервировано)
data/raw/ датасет (только через DVC, в Git лежит .dvc-файл)
models/ модель (только через DVC)
src/ml_ops_course/
data.py загрузка и разбиение данных
preprocessing.py препроцессинг, версии v1 и v2
train.py три эксперимента с MLflow
final.py итоговая модель и оценка на test
schemas.py схемы запроса и ответа
main.py FastAPI: /health, /predict
tests/ тесты API и препроцессинга
Dockerfile, .dockerignore
## Примечания к сдаче

- **Модель в Git.** Файл `models/model.joblib` (около 1 МБ) закоммичен напрямую, а не через DVC, чтобы CI на GitHub Actions мог запускать тесты и сборку без доступа к локальному хранилищу DVC. Датасет по-прежнему версионируется через DVC.
- **Покрытие тестами.** Порог 80% считается по модулям, которые проверяются автоматически (`main.py`, `preprocessing.py`, `schemas.py`). Обучающие скрипты (`data.py`, `train.py`, `final.py`) и UI исключены из измерения: они запускаются вручную, а результаты фиксируются в MLflow.
- **CI.** Workflow `.github/workflows/ci.yml`: lint (`ruff`), тесты с порогом покрытия, сборка Docker-образа и smoke-проверка `/health`.
