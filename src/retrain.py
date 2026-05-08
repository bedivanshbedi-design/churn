from src.train import train_model

def check_drift():
    return True

if check_drift():
    print("Drift detected -> Retraining..")
    model, acc = train_model()
    print("New accuracy:", acc)
else:
    print("No drift")