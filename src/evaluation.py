import pandas as pd
import joblib

df = pd.read_csv("data/raw/data.csv")

X= df.drop("churn", axis=1)
y= df["churn"]

model = joblib.load("model.pkl")

acc=model.score(X,y)

print("Final Accuracy:", acc)