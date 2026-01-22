import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def start_data_transformation():
    # 1. قراءة البيانات اللي DVC جبها
    df = pd.read_csv('data/raw/breast_cancer.csv')
    
    # 2. تقسيم الداتا لـ Features و Target
    X = df.drop(columns=['target'])
    y = df['target']
    
    # 3. تقسيم Train و Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. عمل Scaling (مهم جداً للـ Decision Trees في بعض الحالات والـ KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. حفظ البيانات المعالجة
    os.makedirs('data/processed', exist_ok=True)
    
    # هنحولهم لـ DataFrames عشان نحفظهم بسهولة
    pd.DataFrame(X_train_scaled).to_csv('data/processed/X_train.csv', index=False)
    pd.DataFrame(X_test_scaled).to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    print("✅ Data Transformation Complete: Train/Test sets saved in data/processed")

if __name__ == "__main__":
    start_data_transformation()