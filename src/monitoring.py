import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

ref = pd.read_csv("data/raw/data.csv")

current = ref.copy()
current["age"] += 10

report =Report(metrics =[DataDriftPreset()])
report.run(reference_data=ref, current_data=current)

report.save_html("drift_report.html")

print("Drift report generated")