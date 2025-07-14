import pandas as pd
import requests
import random
import sys
import os
import time

API_URL = "http://localhost:8000/predict"
CSV_PATH = "../data/test.csv"

script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def send_request(df):
    n_passengers = random.randint(1, 5)
    sampled = df.sample(n_passengers)
    payload = []
    for _, row in sampled.iterrows():
        row = row.replace([float('inf'), float('-inf')], pd.NA)
        row = row.fillna(0)
        row_dict = row.to_dict()
        filtered_row = {}
        filtered_row["Pclass"] = int(row_dict.get("Pclass", 0))
        filtered_row["Sex"] = str(row_dict.get("Sex", "male")).lower()
        filtered_row["Age"] = float(row_dict.get("Age", 0))
        filtered_row["SibSp"] = int(row_dict.get("SibSp", 0))
        filtered_row["Parch"] = int(row_dict.get("Parch", 0))
        filtered_row["Fare"] = float(row_dict.get("Fare", 0))
        filtered_row["Embarked"] = str(row_dict.get("Embarked", "S")).upper()
        payload.append(filtered_row)
    print(f"Outgoing payload ({n_passengers} passengers):", payload)
    response = requests.post(API_URL, json=payload)
    print("Response:", response.status_code)
    print(response.json())

def main():
    csv_path = os.path.abspath(os.path.join(script_dir, CSV_PATH))
    df = pd.read_csv(csv_path)
    intervals = [2, 1, 0.5, 0.1, 0.5, 1,2]
    duration = 12  # seconds per interval
    for interval in intervals:
        print(f"--- Sending requests every {interval} seconds for {duration} seconds ---")
        start = time.time()
        while time.time() - start < duration:
            send_request(df)
            print(f"Waiting {interval} seconds before next request...")
            time.sleep(interval)

if __name__ == "__main__":
    main()
