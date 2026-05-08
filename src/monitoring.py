import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import time

def generate_drift_report():
    df = pd.read_csv("data/raw/data.csv")

    split = int(0.7 * len(df))

    reference = df.iloc[:split]
    current = df.iloc[split:]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    report_path = f"drift_report_{int(time.time())}.html"
    report.save_html(report_path)

    return report_path