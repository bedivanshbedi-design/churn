import pandas as pd
import numpy as np
import os

os.makedirs("data/raw", exist_ok=True)

def load_data():
    df=pd.DataFrame({
        "age": np.random.randint(18,60,1000),
        "salary": np.random.randint(20000,100000,1000),
        "balance": np.random.randint(0,50000,1000),
        "tenure": np.random.randint(2,10,1000),
        "churn": np.random.randint(0,2,1000)
    })
    return df

if __name__ == "__main__":
    df=load_data()
    df.to_csv("data/raw/data.csv", index=False)
    print("data_created")