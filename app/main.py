import logging
import time
from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from .schemas import IrisFeatures, PredictionResponse, DriftRequest, DriftResponse
from .metrics import (
    REGISTRY, PREDICTION_COUNTER, PREDICTION_LATENCY, 
    PREDICTION_CONFIDENCE, ERROR_COUNTER, MODEL_LOADED, 
    DRIFT_CHECKS, DRIFT_DETECTED
)
from .drift import DriftDetector
from .logging_config import setup_logging

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model.joblib"
REFERENCE_PATH = BASE_DIR / "reference_stats.joblib"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

setup_logging()
logger = logging.getLogger("ml-api")

app = FastAPI(
    title="Iris ML API with Monitoring & Drift Detection",
    version="3.0.0"
)

model = None
drift_detector = None

@app.on_event("startup")
def startup_event():
    """Виконується при запуску сервера: завантаження ресурсів та встановлення метрик."""
    global model, drift_detector
    
    if not MODEL_PATH.exists():
        MODEL_LOADED.set(0)
        logger.error("model_file_not_found", extra={"path": str(MODEL_PATH)})
    else:
        try:
            model = joblib.load(MODEL_PATH)
            MODEL_LOADED.set(1)
            logger.info("model_loaded_successfully")
        except Exception as e:
            MODEL_LOADED.set(0)
            logger.error("model_load_failed", extra={"error": str(e)})

    if not REFERENCE_PATH.exists():
        logger.warning("reference_stats_not_found", extra={"path": str(REFERENCE_PATH)})
    else:
        try:
            ref_data = joblib.load(REFERENCE_PATH)
            drift_detector = DriftDetector(
                reference=ref_data["X"], 
                feature_names=ref_data["feature_names"]
            )
            logger.info("drift_detector_ready")
        except Exception as e:
            logger.error("drift_detector_init_failed", extra={"error": str(e)})

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware для вимірювання часу обробки запитів (latency)."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    
    if request.url.path == "/predict":
        PREDICTION_LATENCY.observe(process_time)
        
    return response

@app.get("/metrics")
def metrics():
    """Ендпоінт для збору метрик Prometheus (Pull-модель)."""
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health():
    """Перевірка стану сервісу та завантаження моделі."""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "drift_detector_ready": drift_detector is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    """Виконання інференсу моделі та запис бізнес-метрик."""
    if model is None:
        ERROR_COUNTER.labels(error_type="model_unavailable").inc()
        raise HTTPException(status_code=503, detail="ML model is not loaded")
    
    input_data = np.array([[
        features.sepal_length, 
        features.sepal_width, 
        features.petal_length, 
        features.petal_width
    ]])
    
    try:
        class_id = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        confidence = float(probabilities[class_id])
        class_name = CLASS_NAMES[class_id]
        
        PREDICTION_COUNTER.labels(class_name=class_name, status="success").inc()
        PREDICTION_CONFIDENCE.observe(confidence)
        
        logger.info("prediction_made", extra={
            "predicted_class": class_name,
            "confidence": round(confidence, 4)
        })
        
        return PredictionResponse(
            class_id=class_id, 
            class_name=class_name, 
            probability=round(confidence, 4)
        )
    except Exception as e:
        ERROR_COUNTER.labels(error_type="inference_error").inc()
        logger.exception("prediction_failed")
        raise HTTPException(status_code=500, detail="Internal server error during inference")

@app.post("/check-drift", response_model=DriftResponse)
def check_drift(payload: DriftRequest):
    """Ендпоінт для перевірки зсуву даних (Data Drift)."""
    if drift_detector is None:
        ERROR_COUNTER.labels(error_type="drift_detector_unavailable").inc()
        raise HTTPException(status_code=503, detail="Drift detector is not initialized")
    
    DRIFT_CHECKS.inc()
    
    try:
        current_data = np.array(payload.samples)
        result = drift_detector.detect(current_data, alpha=payload.alpha)
        
        for feature, info in result["per_feature"].items():
            if info["drift_detected"]:
                DRIFT_DETECTED.labels(feature=feature).inc()
                logger.warning("data_drift_detected", extra={
                    "feature": feature, 
                    "p_value": info["p_value"]
                })
                
        return DriftResponse(**result)
    except Exception as e:
        ERROR_COUNTER.labels(error_type="drift_calculation_error").inc()
        logger.error("drift_check_failed", extra={"error": str(e)})
        raise HTTPException(status_code=400, detail=str(e))
    
@app.get("/")
def root() -> dict:
    """Кореневий ендпоінт для перевірки працездатності."""
    return {"status": "ok", "service": "Iris ML API", "version": "3.0.0"}