from fastapi.testclient import TestClient
from app.main import app, MODEL_PATH, REFERENCE_PATH
from ml.train import train_and_save

# Переконаємося, що артефакти існують для тестів
if not MODEL_PATH.exists() or not REFERENCE_PATH.exists():
    train_and_save(MODEL_PATH, REFERENCE_PATH)

client = TestClient(app)

def test_metrics_endpoint_available():
    """Перевірка доступності ендпоінта метрик."""
    response = client.get("/metrics")
    assert response.status_code == 200 
    assert "ml_predictions_total" in response.text 
    assert "ml_prediction_latency_seconds" in response.text

def test_predict_increments_counter():
    """Перевірка, що запит до /predict збільшує лічильник у Prometheus."""
    payload = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    
    # Отримуємо стан метрик до запиту
    before = client.get("/metrics").text 
    
    # Робимо два прогнози
    client.post("/predict", json=payload) 
    client.post("/predict", json=payload) 
    
    # Отримуємо стан після
    after = client.get("/metrics").text 
    
    # Перевіряємо наявність міток успіху та класу в метриках
    assert 'class_name="setosa",status="success"' in after
    assert before != after 