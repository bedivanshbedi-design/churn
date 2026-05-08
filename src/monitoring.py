# src/monitoring.py

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def generate_drift_report():
    ref = pd.read_csv("data/raw/data.csv")

    # Simulating current data (you can replace with new_data.csv later)
    current = ref.copy()
    current["age"] += 5  

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=current)

    report.save_html("drift_report.html")