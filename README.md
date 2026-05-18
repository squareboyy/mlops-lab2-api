# Лабораторна робота №2: CI/CD та ML API

[![CI Pipeline](https://github.com/squareboyy/mlops-lab2-api/actions/workflows/ci.yml/badge.svg)](https://github.com/squareboyy/mlops-lab2-api/actions/workflows/ci.yml)

## Опис проєкту
Цей проєкт реалізує наскрізний MLOps-конвеєр для класифікації ірисів. Він включає навчання моделі, створення REST API на FastAPI, автоматичне тестування через GitHub Actions та розгортання в хмарі за допомогою Docker.

## Стек технологій
* **Мова**: Python 3.11
* **Framework**: FastAPI
* **ML**: Scikit-learn, Joblib
* **Тестування**: Pytest, HTTPX
* **CI/CD**: GitHub Actions
* **Контейнеризація**: Docker
* **Деплой**: Render

## Як запустити локально
1. Склонуйте репозиторій: `git clone https://github.com/squareboyy/mlops-lab2-api.git`
2. Створіть віртуальне оточення: `python -m venv .venv`
3. Активуйте його та встановіть залежності: `pip install -r requirements.txt`
4. Навчіть моделі: `python -m ml.train`
5. Запустіть API: `uvicorn app.main:app --reload`

## Запуск через Docker
1. Зберіть образ: `docker build -t ml-api:lab2 .`
2. Запустіть контейнер: `docker run -p 8000:8000 ml-api:lab2`

## Як запустити тести
Виконайте команду:
```bash
pytest -q
```
Тести перевіряють як точність ML-моделі, так і коректність відповідей API.

## Як працює API
* GET `/health`: перевірка працездатності та завантаження моделі.
* POST `/predict`: приймає ознаки квітки (sepal/petal length/width) та повертає назву класу.

## Посилання на деплой
Сервіс розгорнуто на Render: https://mlops-lab2-api-b4pw.onrender.com

## Структура репозиторію
* `app/`: Код FastAPI застосунку та Pydantic-схем.
* `ml/`: Скрипти для навчання та підготовки моделі.
* `tests/`: Unit-тести для моделі та API.
* `Dockerfile`: Конфігурація для контейнеризації.
* `.github/workflows/ci.yml`: Налаштування CI/CD пайплайну.
