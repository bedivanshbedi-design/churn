import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -----------------------------
# OPTIONAL MLFLOW IMPORT
# -----------------------------
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_ENABLED = True
except:
    MLFLOW_ENABLED = False


def train_model(data_path="data/raw/data.csv"):

    # Load data
    df = pd.read_csv(data_path)

    X = df.drop("churn", axis=1)
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()

    # -----------------------------
    # TRAIN WITH / WITHOUT MLFLOW
    # -----------------------------
    if MLFLOW_ENABLED:
        mlflow.set_experiment("churn-mlops")

        with mlflow.start_run(run_name="retrain_run"):
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)

            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, "model")

    else:
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)

    # Save model
    joblib.dump(model, "model.pkl")

    print("Model trained, accuracy:", acc)

    return model, acc
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

    joblib.dump(model,'model.pkl')

    print("Model trained , accuracy:", acc)

    return model, acc