import os
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

default_args = {"owner": "chrnegor", "start_date": datetime(2025, 1, 1)}

dag = DAG(
    "model_training_pipeline",
    default_args=default_args,
    description="Model training pipeline",
    catchup=False,
)

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/app")
env_vars = {
    "PYTHONPATH": PROJECT_ROOT,
    "DATA_PATH": os.path.join(PROJECT_ROOT, "data/processed"),
    "MODEL_SAVE_PATH": os.path.join(PROJECT_ROOT, "models"),
    "CONFIG_PATH": os.path.join(PROJECT_ROOT, "code/models/config.json"),
}

start_tensorboard = BashOperator(
    task_id="start_tensorboard",
    bash_command="cd /opt/airflow && tensorboard --logdir=/opt/airflow/runs --host=0.0.0.0 --port=6006 --bind_all > /opt/airflow/logs/tensorboard.log 2>&1 &",
    dag=dag,
)


train_model = BashOperator(
    task_id="train_model",
    bash_command="cd /opt/airflow && python -m code.models.train",
    env=env_vars,
    dag=dag,
)

calculate_metrics = BashOperator(
    task_id="calculate_metrics",
    bash_command="cd /opt/airflow && python -m code.models.calculate_metrics",
    env=env_vars,
    dag=dag,
)

start_tensorboard >> train_model >> calculate_metrics
