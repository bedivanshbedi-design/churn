import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def train_model(data_path="data/raw/data.csv"):

    df = pd.read_csv("data/raw/data.csv")

    X= df.drop("churn", axis=1)
    y= df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

    mlflow.set_experiment("churn-mlops")

    with mlflow.start_run():
        model = RandomForestClassifier()
        model.fit(X_train,y_train)

        acc = model.score(X_test, y_test)

        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

    joblib.dump(model,'model.pkl')

    print("Model trained , accuracy:", acc)

    return model, acc