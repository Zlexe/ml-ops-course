#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8001}"

echo "Проверка /health..."
health=$(curl -sf "$BASE_URL/health")
echo "$health" | grep -q '"model_loaded":true' || { echo "FAIL: модель не загружена"; exit 1; }

echo "Проверка /predict..."
payload='{"age":41,"job":"technician","marital":"married","education":"secondary","default":"no","balance":1270,"housing":"yes","loan":"no","contact":"unknown","day":5,"month":"may","campaign":1,"pdays":-1,"previous":0,"poutcome":"unknown"}'
pred=$(curl -sf -X POST "$BASE_URL/predict" -H "Content-Type: application/json" -d "$payload")
echo "$pred" | grep -q '"label"' || { echo "FAIL: нет поля label в ответе"; exit 1; }

echo "Проверка контролируемой ошибки на неверном вводе..."
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/predict" -H "Content-Type: application/json" -d '{"age":"x"}')
[ "$code" = "422" ] || { echo "FAIL: ожидался 422, получен $code"; exit 1; }

echo "OK: smoke-проверка пройдена"
