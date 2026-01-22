import pandas as pd
from sklearn.datasets import load_breast_cancer
import os

def ingest_data():
    # 1. سحب البيانات من المصدر
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    # 2. التأكد من وجود المجلدات
    os.makedirs('data/raw', exist_ok=True)
    
    # 3. حفظ الملف كـ CSV
    raw_path = 'data/raw/breast_cancer.csv'
    df.to_csv(raw_path, index=False)
    print(f"✅ Data ingested successfully at: {raw_path}")

if __name__ == "__main__":
    ingest_data()