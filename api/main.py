from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import pickle
import logging
import time
import psutil
import os
import sys
from typing import List, Literal
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
from contextlib import asynccontextmanager
from sklearn.pipeline import Pipeline
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Use custom logging utility
from src.logging_utils import setup_logging

# Path to API logging config
API_LOGS_CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs/api_logs_config.json'))
API_LOG_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs/api.log'))
os.makedirs(os.path.dirname(API_LOG_FILE_PATH), exist_ok=True)
setup_logging(log_path=API_LOG_FILE_PATH, logs_config_path=API_LOGS_CONFIG_PATH)
logger = logging.getLogger("api")


# Global state
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/best_model.pkl'))
model = None
metrics = {"requests": 0, "prediction_time": 0.0, "start_time": time.time()}


# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API Requests', ['method', 'endpoint', 'http_status'])
REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'API Request latency', ['endpoint'])
#CPU_USAGE = Gauge('api_cpu_usage_percent', 'API process CPU usage percent')
#MEMORY_USAGE = Gauge('api_memory_usage_mb', 'API process memory usage in MB')

class PassengerInput(BaseModel):
    """Passenger data input"""
    Pclass: int = Field(..., ge=1, le=3)
    Sex: Literal["male", "female"] = Field(...)
    Age: float = Field(..., ge=0, le=120)
    SibSp: int = Field(..., ge=0, le=10)
    Parch: int = Field(..., ge=0, le=10)
    Fare: float = Field(..., ge=0,le=10000)
    Embarked: Literal["C", "Q", "S"] = Field(...)
    
    from pydantic import field_validator

    @field_validator('Sex')
    @classmethod
    def validate_sex(cls, v):
        if v.lower() not in ['male', 'female']:
            raise ValueError('Sex must be "male" or "female"')
        return v.lower()

    @field_validator('Embarked')
    @classmethod
    def validate_embarked(cls, v):
        if v.upper() not in ['C', 'Q', 'S']:
            raise ValueError('Embarked must be C, Q, or S')
        return v.upper()

class PredictionResult(BaseModel):
    """Prediction result for a single passenger"""
    prediction_label: str = Field(..., description="Human-readable prediction label: 'survived' or 'not survive'")
    survival_prediction: int = Field(..., description="0 = did not survive, 1 = survived")
    survival_probability: float = Field(..., description="Probability of survival (0-1)")

class PredictResponse(BaseModel):
    """Response model for /predict endpoint"""
    predictions: List[PredictionResult]
    processing_time_ms: float
    model_info: dict

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    total_requests: int
    avg_prediction_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float

class MetricsResponse(BaseModel):
    requests: int
    avg_prediction_time_ms: float
    memory_mb: float
    cpu_percent: float
    model_loaded: bool

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup"""
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            print('aaaa')
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
        yield
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    finally:
        logger.info("API shutdown")

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predict passenger survival on the Titanic",
    version="1.0.0",
    lifespan=lifespan
)


@app.middleware("http")
async def logging_and_prometheus_middleware(request: Request, call_next):
    """Log requests, track performance, and update Prometheus metrics"""
    metrics["requests"] += 1
    start_time = time.time()
    logger.info(f"Request {metrics['requests']}: {request.method} {request.url}")
    # Update CPU and memory usage gauges on every request
    #process = psutil.Process()
    #MEMORY_USAGE.set(process.memory_info().rss / 1024 / 1024)
    #CPU_USAGE.set(psutil.cpu_percent())
    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        logger.info(f"Request completed in {process_time:.2f}ms")
        response.headers["X-Process-Time"] = str(process_time)
        # Prometheus metrics (seconds)
        endpoint = request.url.path
        REQUEST_COUNT.labels(request.method, endpoint, response.status_code).inc()
        REQUEST_LATENCY.labels(endpoint).observe(process_time / 1000.0)
        return response
    except Exception as e:
        logger.error(f"Request failed: {e}")
        # Optionally, you can increment an error counter here
        raise

@app.post("/predict", response_model=PredictResponse)
async def predict_survival(passengers: List[PassengerInput]):
    """Predict survival for passengers"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    start_time = time.time()
    try:
        # Convert to DataFrame and use directly for prediction
        data = [p.model_dump() for p in passengers]
        df = pd.DataFrame(data)
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)

        # Format results
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                "prediction_label": "survived" if pred == 1 else "not survive",
                "survival_prediction": int(pred),
                "survival_probability": float(prob[1]),
            })

        processing_time = (time.time() - start_time) * 1000
        metrics["prediction_time"] += processing_time

        logger.info(f"Processed {len(passengers)} passengers in {processing_time:.2f}ms")

        return {
            "predictions": results,
            "processing_time_ms": processing_time,
            "model_info": {"algorithm": type(model).__name__, "version": "1.0.0"}
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check with system metrics."""
    try:
        process = psutil.Process()
        uptime = time.time() - metrics["start_time"]
        avg_time = metrics["prediction_time"] / max(metrics["requests"], 1)
        return HealthResponse(
            status="healthy" if model is not None else "unhealthy",
            uptime_seconds=uptime,
            total_requests=metrics["requests"],
            avg_prediction_time_ms=avg_time,
            memory_usage_mb=process.memory_info().rss / 1024 / 1024,
            cpu_usage_percent=psutil.cpu_percent()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/metricas", response_model=MetricsResponse)
async def get_metrics():
    """Detailed metrics for monitoring."""
    try:
        process = psutil.Process()
        return MetricsResponse(
            requests=metrics["requests"],
            avg_prediction_time_ms=metrics["prediction_time"] / max(metrics["requests"], 1),
            memory_mb=process.memory_info().rss / 1024 / 1024,
            cpu_percent=psutil.cpu_percent(),
            model_loaded=model is not None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Metrics unavailable")


@app.get("/prometheus", include_in_schema=False)
def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
async def root():
    """API information"""
    return {
        "message": "Titanic Survival Prediction API",
        "version": "1.0.0",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)