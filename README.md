# Titanic ML Pipeline API

A machine learning pipeline and API for predicting passenger survival on the Titanic. This solution implements an automated training pipeline with profiling, logging, and a FastAPI service for inference.

## 🏗️ Project Structure

```
├── api/
│   └── main.py                 # FastAPI application for predictions
├── src/                        # Core ML pipeline components
│   ├── train_xgboost.py       # Main training pipeline with profiling
│   ├── column_transformers.py # Feature preprocessing transformers
│   ├── loaders.py             # Data loading utilities
│   ├── train_profiling.py     # CPU and RAM usage profiling
│   └── logging_utils.py       # Logging configuration
├── models/                     # Trained models and evaluation reports
│   ├── best_model.pkl         # XGBoost trained model
│   ├── rf_clf.pkl            # RandomForest model (used by API)
│   ├── eval_report.json      # Cross-validation results
│   └── feature_importance.json # Feature importance analysis
├── data/                       # Training datasets
├── notebooks/                  # Jupyter notebooks for exploration
├── configs/
│   └── logs_config.json       # Logging configuration
├── logs/                       # Application logs
├── conda-env.yml              # Conda environment specification
├── requirements.txt           # Python dependencies
└── Dockerfile                 # Docker container for training
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker (optional)

### Environment Setup and Training

**Option 1: Using Docker (Recommended)**
```bash
docker build -f Dockerfile.train -t titanic-train .
docker run --rm -v $(pwd):/app titanic-train

```

**Option 2: Using Conda**
```bash
conda env create -f conda-env.yml
conda activate titanic-ml
python src/train_xgboost.py
```

### API Deployment

**Option 1: Docker (Production Ready)**
```bash
# Build and run API
docker build -f Dockerfile.api -t titanic-api .
docker run -p 8000:8000 titanic-api

# Or using docker-compose
docker-compose up --build
```

**Option 2: Local Development**
```bash
# Install API dependencies
pip install fastapi uvicorn python-multipart psutil

# Run the API
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Testing

**Containerized Testing (Ensures Environment Consistency)**
```bash
# Run tests using docker-compose (recommended)
docker-compose --profile test run --rm test

# Or build and run test container directly
docker build -f Dockerfile.test -t titanic-api-test .
docker run --rm titanic-api-test

# Local testing (if needed)
pytest tests/ -v
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

##  Implemented Features

### 1. Binary Classifier 
- **XGBoost implementation** with hyperparameter optimization
- **Stratified cross-validation** for model evaluation
- **Feature cleaning and scaling** with proper preprocessing pipeline
- **Model comparison** between XGBoost, LogisticRegression, RandomForest, and LinearSVM

### 2. Automated Pipeline Execution 
- **Single command training**: `python src/train_xgboost.py`
- **Modular pipeline** with separate stages for data loading, preprocessing, training
- **Automated model saving** and evaluation report generation
- **Reproducible results** with fixed random seeds

### 3. Pipeline Profiling 
- **CPU and RAM monitoring** during each pipeline stage
- **Execution time tracking** for performance analysis
- **Resource usage logging** with detailed profiling information
- Located in `src/train_profiling.py`

### 4. Dockerized Solution 
- **Training pipeline containerization** with `Dockerfile`
- **Python 3.11-slim base image** for efficiency
- **Volume mounting** for model persistence
- **Environment variable** setup for reproducibility

### 5. Improved Classifier Implementation 
- **Enhanced from notebook**: Added proper cross-validation, hyperparameter tuning
- **Grid search optimization** for best parameters
- **Stratified sampling** to handle class imbalance
- **Multiple model comparison** with systematic evaluation

### 6. Logs Configuration 
- **Structured logging** with `src/logging_utils.py`
- **Configurable logging** via `configs/logs_config.json`
- **File and console output** with rotation and backup
- **Profiling logs** integrated into training pipeline

### 7. Virtual Environment Configuration 
- **Conda environment**: `conda-env.yml` with pinned versions
- **Pip requirements**: `requirements.txt` for alternative setup
- **Dependency management** with specific versions for reproducibility

## 📊 Model Performance

### XGBoost Classifier Results
- **F1 Score**: 0.767
- **Cross-validation**: 5-fold stratified
- **Hyperparameter optimization**: GridSearchCV with 7 parameters
- **Feature importance**: Exported to `models/feature_importance.json`

**Key Features by Importance:**
1. **Sex**: 48.5% (primary survival factor)
2. **Pclass**: 21.9% (passenger class/socioeconomic status)
3. **Embarked**: 8.8% (port of embarkation)
4. **SibSp**: 5.6% (siblings/spouses aboard)
5. **Age**: 5.4% (passenger age)

## 🔧 Technical Implementation

### Data Processing Pipeline
- **Custom transformer**: `TitanicInputTransformer` for feature cleaning
- **Missing value handling**: Age (mean=28), Fare (mean=32), Embarked ('S')
- **One-hot encoding**: For categorical variables (Sex, Pclass, Embarked)
- **Multicollinearity handling**: Dropping redundant dummy variables

### Profiling System
```python
# Example profiling output
[PROFILE] Starting stage: Model Training
[PROFILE] ⤷ Model Training completed in 45.23s
[PROFILE] ⤷ RAM used: 156.42 MB
[PROFILE] ⤷ CPU usage during stage: 87.50%
```

## 🌐 API Features

### Production-Ready FastAPI Service
- **Comprehensive validation**: Pydantic models with field constraints
- **Error handling**: Proper HTTP status codes and detailed error messages
- **Documentation**: Automatic OpenAPI/Swagger documentation at `/docs`
- **CORS support**: Configurable cross-origin resource sharing
- **Health monitoring**: `/health` endpoint with system status

### Logging & Monitoring
- **Structured logging**: Request/response tracking with correlation IDs
- **Performance metrics**: Response times, request counts, error rates
- **System monitoring**: CPU, RAM usage via `/metrics` endpoint
- **Log persistence**: File-based logging with rotation

### Input Processing
- **Single predictions**: Individual passenger data
- **Batch processing**: Multiple passengers in single request
- **Data validation**: Age (0-120), Fare (≥0), proper categorical values
- **Preprocessing pipeline**: Automatic feature transformation

### API Endpoints
```bash
POST /predict           # Single passenger prediction
POST /predict/batch     # Multiple passengers prediction
GET  /health           # Health check and system status
GET  /metrics          # Performance and system metrics
GET  /docs             # Interactive API documentation
```

### Example Usage
```python
import requests

# Single prediction
response = requests.post("http://localhost:8000/predict", json={
    "pclass": 3, "sex": "male", "age": 22.0, "sibsp": 1,
    "parch": 0, "fare": 7.25, "embarked": "S"
})
print(response.json())  # {"prediction": 0, "probability": 0.23}

# Batch prediction
passengers = [
    {"pclass": 1, "sex": "female", "age": 35, "sibsp": 0, "parch": 0, "fare": 80, "embarked": "C"},
    {"pclass": 3, "sex": "male", "age": 25, "sibsp": 0, "parch": 0, "fare": 8, "embarked": "S"}
]
response = requests.post("http://localhost:8000/predict/batch", json={"passengers": passengers})
```

## 🐳 Docker Deployment

### Testing in Containers
```bash
# Run tests in containerized environment (recommended for CI/CD)
docker build -f Dockerfile.test -t titanic-api-test .
docker run --rm titanic-api-test

# Or use docker-compose
docker-compose --profile test run --rm test
```

### Training Pipeline
```bash
docker build -t titanic-train .
docker run --rm -v $(pwd):/app titanic-train
```

### API Service
```bash
# Build with integrated testing (tests run during build)
docker build -f Dockerfile.api -t titanic-api .
docker run -p 8000:8000 titanic-api

# Production with docker-compose
docker-compose --profile production up
```

### Container Features
- **Security**: Non-root user execution
- **Health checks**: Built-in container health monitoring  
- **Log persistence**: Volume mounting for log retention

**Environment variables:**
- `PYTHONHASHSEED=42` for reproducible results
- `LOG_LEVEL=INFO` for logging configuration

## 📁 Key Files

| File | Purpose |
|------|---------|
| `src/train_xgboost.py` | Main training script with profiling |
| `src/train_profiling.py` | CPU/RAM monitoring utilities |
| `src/column_transformers.py` | Feature preprocessing pipeline |
| `src/logging_utils.py` | Configurable logging setup |
| `api/main.py` | FastAPI prediction service |
| `configs/logs_config.json` | Logging configuration |
| `conda-env.yml` | Environment specification |

## � Running the Complete Pipeline

1. **Setup environment**: `conda env create -f conda-env.yml`
2. **Train model**: `python src/train_xgboost.py`
3. **Start API**: `cd api && uvicorn main:app --reload`
4. **Make predictions**: Use the `/predict` endpoint

## 📈 Evaluation Results

The pipeline generates files with evaluation data:
- **Cross-validation metrics** saved to `models/eval_report.json`
- **Feature importance** analysis in `models/feature_importance.json`
- **Model artifacts** saved as pickle files
- **Detailed logs** with profiling information






---

This implementation focuses on the core requirements with practical, working solutions for binary classification, automation, profiling, and containerization.

