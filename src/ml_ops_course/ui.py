import json
import os
import urllib.error
import urllib.request

import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Bank Marketing: прогноз депозита", page_icon="🏦")
st.title("Прогноз согласия на срочный депозит")
st.caption("Интерфейс обращается к FastAPI-сервису по /predict. Модель внутри UI не используется.")

with st.form("client"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Возраст", 18, 100, 41)
        job = st.selectbox("Род занятий", [
            "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
            "retired", "self-employed", "services", "student", "technician",
            "unemployed", "unknown"], index=9)
        marital = st.selectbox("Семейное положение", ["married", "single", "divorced"])
        education = st.selectbox("Образование", ["primary", "secondary", "tertiary", "unknown"], index=1)
        default = st.selectbox("Просроченный кредит", ["no", "yes"])
        balance = st.number_input("Баланс, евро", -10000, 200000, 1270)
    with col2:
        housing = st.selectbox("Ипотека", ["yes", "no"])
        loan = st.selectbox("Персональный кредит", ["no", "yes"])
        contact = st.selectbox("Тип связи", ["unknown", "cellular", "telephone"])
        day = st.number_input("День последнего контакта", 1, 31, 5)
        month = st.selectbox("Месяц", ["jan", "feb", "mar", "apr", "may", "jun",
                                        "jul", "aug", "sep", "oct", "nov", "dec"], index=4)
        campaign = st.number_input("Контактов в этой кампании", 1, 100, 1)
        pdays = st.number_input("Дней с прошлой кампании (-1 — не звонили)", -1, 999, -1)
        previous = st.number_input("Контактов до кампании", 0, 100, 0)
        poutcome = st.selectbox("Итог прошлой кампании", ["unknown", "failure", "success", "other"])
    submitted = st.form_submit_button("Получить прогноз")

if submitted:
    payload = {
        "age": int(age), "job": job, "marital": marital, "education": education,
        "default": default, "balance": int(balance), "housing": housing, "loan": loan,
        "contact": contact, "day": int(day), "month": month, "campaign": int(campaign),
        "pdays": int(pdays), "previous": int(previous), "poutcome": poutcome,
    }
    req = urllib.request.Request(
        f"{API_URL}/predict",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
        st.metric("Вероятность согласия", f"{result['probability']:.1%}")
        if result["label"] == "yes":
            st.success("Прогноз: клиент, скорее всего, согласится")
        else:
            st.info("Прогноз: клиент, скорее всего, откажется")
        st.caption(f"Версия модели: {result['model_version']}")
    except urllib.error.URLError:
        st.error(f"Сервис недоступен по адресу {API_URL}. Запустите API.")
    except urllib.error.HTTPError as e:
        st.error(f"Сервис вернул ошибку {e.code}. Проверьте введённые данные.")
