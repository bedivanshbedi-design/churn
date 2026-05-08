import os

def check_drift():
    return True 


if check_drift():
    print("Drift detected -> Retraining..")
    os.system("python src/train.py")

else:
    print("No drift")