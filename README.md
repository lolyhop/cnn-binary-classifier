# CNN Binary Classifier: Cats vs Dogs

An MLOps pipeline for binary image classification using PyTorch ResNet-18 to classify cats and dogs images. The project includes automated data processing, model training, evaluation, deployment, and a web interface.

## 🏗️ Architecture

```
├── code/
│   ├── datasets/           
│   ├── deployment/        # API and Web App
│   │   ├── api/           # FastAPI backend
│   │   └── app/           # Streamlit frontend
│   └── models/            # Training and evaluation scripts
├── data/
│   ├── raw/               # Original datasets
│   └── processed/         # Cleaned and split data
├── models/                # Trained model artifacts
└── services/
    └── airflow/           # Airflow DAGs and configuration
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.12+
- Git

### Clone Repository
```bash
git clone https://github.com/lolyhop/cnn-binary-classifier.git
cd cnn-binary-classifier
```

## 1. 🐳 Deployment

Deploy the trained model with FastAPI backend and Streamlit frontend.

```bash
docker-compose up -d
```

### Access Services
- **Web Application**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/status

## 2. ✈️ Airflow Pipelines

Automated MLOps pipelines for data processing and model training.

### Setup Airflow
```bash
cd services/airflow

# Start Airflow with Python 3.12
docker-compose up -d

# Access Airflow UI
# URL: http://localhost:8080
# Username: admin
# Password could be found in docker container logs
```

### 2.1 ETL Pipeline

**DAG**: `etl_pipeline`
**Description**: Automated data ingestion, cleaning, and preprocessing

**Pipeline Steps**:
1. **Data Ingestion**: Download/load raw images
2. **Data Cleaning**: Remove corrupted files, standardize formats
3. **Data Splitting**: Train/validation/test split (70/15/15)
4. **Data Validation**: Check data quality and distribution


### 2.2 Model Training Pipeline

**DAG**: `model_training_pipeline`
**Description**: Complete ML pipeline from training to deployment

**Pipeline Steps**:

1. **Start TensorBoard**: Launch monitoring dashboard
2. **Model Training**: Train ResNet-18 on processed data and save model weights
3. **Model Evaluation**: Calculate performance metrics

## 📊 Monitoring & Logs

### TensorBoard
```bash
# Access training metrics
http://localhost:6006
```

### Airflow Monitoring
```bash
# View all DAGs
http://localhost:8080

# Check task execution
http://localhost:8080/admin/airflow/graph?dag_id=model_training_pipeline
```

### Model Artifacts
- **Trained Model**: `models/cats_dogs_model.pth`
- **Training History**: `models/training_history.json`
- **Evaluation Metrics**: `models/metrics.csv`
- **Model Packages**: `models/cats_dogs_model_v{timestamp}.zip`
