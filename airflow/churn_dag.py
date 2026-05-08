from airflow import DAG 
from airflow.operators.python import PythonOperator
from datetime import datetime
import os

def data_pipeline():
    os.system(python src/data_pipeline.py)

def evaluation():
    os.system(python src/evaluation.py)

def monitoring():
    os.system(python src/monitoring.py)

def retraining():
    os.system(python src/train.py)

dag = DAG("mlops_churn", start_date=datetime(2024,1,1), schedule_interval="@daily")

t1 = PythonOperator(task_id="data", python_callable=data_pipeline, dag=dag)
t2 = PythonOperator(task_id="train", python_callable=train, dag=dag)
t3 = PythonOperator(task_id="monitor", python_callable=monitor, dag=dag)
t4 = PythonOperator(task_id="retrain", python_callable=retrain, dag=dag)

t1 >> t2 >> t3 >> t4