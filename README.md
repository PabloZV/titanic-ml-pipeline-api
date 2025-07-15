# Titanic ML Pipeline API

A machine learning pipeline and API for predicting passenger survival on the Titanic. This solution implements an automated training pipeline with profiling, logging, and a FastAPI service for inference.

## 🏗️ Project Structure

```
├── api/
│   ├── main.py                 # FastAPI application for predictions
│   ├── sample_client.py        # Sample client for testing API
│   └── sample_client_repeater.py # Load testing client
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
│   ├── train.csv              # Training data
│   └── test.csv               # Test data
├── tests/                      # Unit tests
│   ├── test_api.py            # API endpoint tests
│   └── test_model.py          # Model functionality tests
├── notebooks/                  # Jupyter notebooks for exploration
│   ├── 1_feature_cleaning.ipynb
│   ├── 2_comparing_models.ipynb
│   └── explorations-ml-challenge.ipynb
├── configs/
│   ├── logs_config.json       # Logging configuration
│   ├── api_logs_config.json   # API logging configuration
│   └── model_config.json      # Model configuration
├── grafana/                    # Grafana monitoring setup
│   └── provisioning/          # Dashboards and datasources
├── logs/                       # Application logs
├── conda-env.yml              # Conda environment specification
├── requirements.txt           # Python dependencies
├── data_preprocessing.py       # Standalone data preprocessing script
├── design_decisions.md         # Architecture and design documentation
├── prometheus.yml             # Prometheus configuration
├── docker-compose-api.yml     # Docker Compose for full stack
├── Dockerfile.api             # Docker container for API service
├── Dockerfile.train           # Docker container for training pipeline
├── data.dvc                   # DVC tracking for data files
└── models.dvc                 # DVC tracking for model files
```

# Quickstart

## 1. Prerequisite: .env-keys file and DVC tracked files
Create a `.env-keys` file in the project root with your AWS credentials:
```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

```bash
dvc pull --force
```

## 2. Run the Training Pipeline (Docker)
**Build:**
```bash

export $(cat .env-keys | xargs) && docker build --build-arg AWS_ACCESS_KEY_ID --build-arg AWS_SECRET_ACCESS_KEY -f Dockerfile.train -t titanic-train .
```
**Run:**
```bash
docker run --rm -v $(pwd):/app titanic-train
```

## 3. Run the API (Docker)
**Build:**
```bash
export $(cat .env-keys | xargs) && docker build --build-arg AWS_ACCESS_KEY_ID --build-arg AWS_SECRET_ACCESS_KEY -f Dockerfile.api -t titanic-api .
```
**Run:**
```bash
docker run --env-file .env-keys -p 8000:8000 titanic-api
```

## 4. Run the Full Stack (API + Monitoring) with Docker Compose
```bash
docker compose -f docker-compose-api.yml up --build -d
```
- The API will be available at: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Prometheus UI at: http://localhost:9090
- Grafana dashboards and Alerts: http://localhost:3000

To failitate stess CPU usage when seeing grafana dashboards ./api/sample_client_repeater.py or ./api/sample_client.py . One alert was set up as demsotration, it can be seen in the Grafana UI.



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
POST /predict          # Passenger prediction
GET  /health           # Health check and system status
GET  /prometheus       # Performance and system metrics meant to be consumed by prometheus
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

## Feature Importance Endpoint

The API provides a `/feature_importance` endpoint that returns the preprocessed feature importances from `/models/feature_importance.json`.

**Usage:**
- Send a GET request to `/feature_importance`
- Response will be a JSON object with the feature importances used by the model

Example response:
```json
{
  "feature_importance": {
    "feature1": 0.23,
    "feature2": 0.18,
    ...
  }
}
```

# Data & Model Versioning with DVC and S3

This project uses [DVC](https://dvc.org/) to version and manage large files in `/data` and `/models`.

## Remote Storage
- **Remote:** AWS S3 bucket: `titanic-ml-pipeline-api/ml-pipeline`
- DVC is configured to push and pull data/models from this S3 bucket.

## How to Use
1. **Install requirements:**
   - With pip: `pip install 'dvc[s3]'`
   - Or use the provided `conda-env.yml`.
2. **Set up AWS credentials:**
   - Use `aws configure` or set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`.
3. **Pull data and models:**
   ```bash
   dvc pull
   ```
4. **Push changes (if you have write access):**
   ```bash
   dvc push
   ```

## Notes
- The S3 bucket may be public for read-only access, but only authorized users can push.
- See `.dvcignore` for files/folders excluded from DVC tracking.

## DVC S3 Access for Reviewers

A `.env-keys` file containing AWS credentials will be provided for the revision of this challenge. This is required to allow reviewers to pull the DVC-tracked data and models from the private S3 bucket.

**Note:**
- A better solution for open review would be to make the S3 bucket public-read, so anyone can pull the DVC repo without credentials.
- As an improvement, consider configuring the S3 bucket for public read access and removing the need for secret keys in the future.

---

This implementation focuses on the core requirements with practical, working solutions for binary classification, automation, profiling, and containerization.

## API Documentation

### Overview
This API provides real-time predictions for Titanic passenger survival using a trained classifier. It is designed for batch predictions, monitoring, profiling, and production readiness.

### Features
- **Prediction Endpoint:**  Receives a list of passengers and returns survival predictions for each.
- **Error Handling:**  Returns appropriate HTTP status codes and error messages for invalid input, missing model, and unhandled exceptions.
- **Interactive Documentation:**  Built with FastAPI, providing automatic OpenAPI docs at `/docs` and `/redoc`.
- **Logging:**  All requests, responses, and errors are logged to file and console.
- **Monitoring & Alerts:**  Exposes Prometheus metrics for requests, latency, CPU, and RAM usage. Integrated with Grafana dashboards and alerting.
- **Profiling:**  Tracks CPU and RAM usage, request rates, and supports scaling via Docker and multi-worker deployment.
- **Unit Testing:**  Includes tests for the model’s predict function to ensure reliability.
- **Dockerized:**  All components (API, Prometheus, Grafana) are containerized for easy deployment.

### Main Endpoints
- `POST /predict`  Accepts a list of passenger objects and returns survival predictions.
- `GET /health`  Returns API health status and system metrics.
- `GET /prometheus`  Exposes metrics for Prometheus scraping.

### Usage Example
```json
POST /predict
[
  {
    "Pclass": 3,
    "Sex": "male",
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
  }
]
```

### Monitoring & Scaling
- Metrics are available in Grafana dashboards.
- Alerts are provisioned for high CPU usage and other conditions.
- Profiling data (CPU, RAM, request rates) is collected and visualized.
- The API is designed to scale horizontally using Docker and multiple workers.

### Error Handling
- Returns `400 Bad Request` for invalid input.
- Returns `503 Service Unavailable` if the model is not loaded.
- Returns `500 Internal Server Error` for unhandled exceptions.

### Testing
- Unit tests for the model’s predict function are included in the `tests/` directory.

### Deployment
- All services are defined in Docker Compose for reproducible deployment.
- Prometheus and Grafana are auto-provisioned for monitoring and alerting.



---

## Notes
- The API container now requires the `.env-keys` file at runtime (see `--env-file .env-keys` above or the `env_file` section in docker-compose).
- The API will automatically run `dvc pull --force` at startup to fetch models/data from S3.
- The training pipeline still requires credentials at build time for DVC pull.

