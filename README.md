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
# Password: admin
```

### 2.1 ETL Pipeline

**DAG**: `etl_pipeline`
**Schedule**: Every 5 minutes
**Description**: Automated data ingestion, cleaning, and preprocessing

**Pipeline Steps**:
1. **Data Ingestion**: Download/load raw images
2. **Data Cleaning**: Remove corrupted files, standardize formats
3. **Data Splitting**: Train/validation/test split (70/15/15)
4. **Data Validation**: Check data quality and distribution

**Configuration**:
```bash
# Environment variables
export DATA_SOURCE="path/to/raw/data"
export PROCESSED_DATA_PATH="data/processed"
export TRAIN_SPLIT_RATIO="0.7"
export VAL_SPLIT_RATIO="0.15"
```

**Manual Trigger**:
```bash
# Trigger ETL pipeline
docker exec airflow-airflow-1 airflow dags trigger etl_pipeline
```

### 2.2 Model Training Pipeline

**DAG**: `model_training_pipeline`
**Schedule**: Every 5 minutes (after data is ready)
**Description**: Complete ML pipeline from training to deployment

**Pipeline Steps**:

1. **Start TensorBoard**: Launch monitoring dashboard
2. **Model Training**: Train ResNet-18 on processed data
3. **Model Evaluation**: Calculate performance metrics
4. **Model Packaging**: Package model with metadata

**Configuration**:
```json
// code/models/config.json
{
    "data": {
        "data_path": "data/processed",
        "batch_size": 32,
        "num_workers": 0
    },
    "model": {
        "num_classes": 2,
        "backbone": "resnet18"
    },
    "training": {
        "learning_rate": 1e-3,
        "num_epochs": 3,
        "step_size": 7,
        "gamma": 0.1
    },
    "paths": {
        "model_save_path": "models",
        "tensorboard_log_dir": "/opt/airflow/runs",
        "metrics_save_path": "models"
    },
    "device": "auto"
}
```

**Pipeline Tasks**:

#### Task 1: Start TensorBoard
```bash
tensorboard --logdir=/opt/airflow/runs --host=0.0.0.0 --port=6006
```
**Access**: http://localhost:6006

#### Task 2: Train Model
- Uses PyTorch ResNet-18 with transfer learning
- Logs metrics to TensorBoard in real-time
- Saves model checkpoints and training history

#### Task 3: Calculate Metrics
- Evaluates model on test set
- Generates comprehensive metrics report
- Saves results to `models/metrics.csv`

#### Task 4: Package Model
- Creates deployment-ready model package
- Includes model weights, metadata, and deployment script
- Generates timestamped archive

**Manual Operations**:
```bash
# Trigger training pipeline
docker exec airflow-airflow-1 airflow dags trigger model_training_pipeline

# View pipeline status
docker exec airflow-airflow-1 airflow dags state model_training_pipeline

# Check task logs
docker exec airflow-airflow-1 airflow tasks logs model_training_pipeline train_model
```

**Environment Variables**:
```bash
export DATA_PATH="/opt/airflow/data/processed"
export MODEL_SAVE_PATH="/opt/airflow/models"
export CONFIG_PATH="/opt/airflow/code/models/config.json"
export TENSORBOARD_LOG_DIR="/opt/airflow/runs"
```

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

## 🛠️ Development

### Local Training
```bash
# Train model locally
python -m code.models.train

# Calculate metrics
python -m code.models.calculate_metrics

# Package model
python -m code.models.package_model
```

### Configuration
Modify `code/models/config.json` for:
- Training hyperparameters
- Data paths
- Model architecture
- Device selection (CPU/CUDA/MPS)

### API Development
```bash
# Test API endpoints
curl http://localhost:8000/status
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"image": "base64_image_string"}'
```

## 📈 Performance

### Model Metrics
- **Architecture**: ResNet-18 with transfer learning
- **Dataset**: Cats vs Dogs binary classification
- **Input Size**: 224x224 RGB images
- **Training Time**: ~3 epochs on CPU
- **Inference Time**: <1 second per image

### System Requirements
- **CPU**: 2+ cores
- **RAM**: 4GB+ (8GB recommended)
- **Storage**: 2GB+ for Docker images and data
- **Network**: Required for downloading pretrained weights

## 🔧 Troubleshooting

### Common Issues

**Airflow Connection Issues**:
```bash
# Restart Airflow
cd services/airflow
docker-compose restart
```

**Model Loading Errors**:
```bash
# Check model file exists
ls -la models/cats_dogs_model.pth

# Verify model path in API
docker-compose logs api
```

**TensorBoard Not Loading**:
```bash
# Check logs directory
ls -la runs/

# Restart TensorBoard
docker exec airflow-airflow-1 pkill -f tensorboard
```

**Port Conflicts**:
```bash
# Check occupied ports
lsof -i :8000
lsof -i :8501
lsof -i :8080
lsof -i :6006
```

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request