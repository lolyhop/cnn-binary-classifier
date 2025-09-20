from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import random


dag = DAG(
    "example_pipeline",
    start_date=datetime(2025, 1, 1),
    description="Simple data pipeline",
    catchup=False,
)


def extract_data(**context):
    data = [random.randint(1, 100) for _ in range(10)]
    print(f"1. Extracted: {data}")
    return data


def transform_data(**context):
    raw_data = context["task_instance"].xcom_pull(task_ids="extract")
    transformed = [x * 2 for x in raw_data if x > 50]
    print(f"2. Transformed: {transformed}")
    return transformed


def load_data(**context):
    data = context["task_instance"].xcom_pull(task_ids="transform")
    print(f"3. Loaded {len(data)} records: {data}")
    return f"Success: {len(data)} records"


extract = PythonOperator(task_id="extract", python_callable=extract_data, dag=dag)
transform = PythonOperator(task_id="transform", python_callable=transform_data, dag=dag)
load = PythonOperator(task_id="load", python_callable=load_data, dag=dag)

extract >> transform >> load
