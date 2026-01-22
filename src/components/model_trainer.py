import pandas as pd
import joblib
import os
# السطر ده هو "الوصلة" بين كودك وبين عملية التدريب
from src.components.model import MyDecisionTree 

def train_model():
    print("--- Starting Model Training Sprint ---")
    
    # 1. تحميل البيانات اللي جهزناها في الـ Transformation
    X_train = pd.read_csv('data/processed/X_train.csv').values
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    
    # 2. استدعاء الموديل بتاعك (بتاع الـ Info Gain)
    # تقدر تغير الـ max_depth من هنا لو حابب
    model = MyDecisionTree(max_depth=3) 
    
    # 3. عملية التدريب (تنفيذ الـ fit اللي أنت كتبتها)
    print("Fitting the model with Information Gain...")
    model.fit(X_train, y_train)
    
    # 4. حفظ الموديل في فولدر models عشان الـ Evaluation يشوفه
    os.makedirs('models', exist_ok=True)
    model_path = 'models/breast_cancer_model.pkl'
    joblib.dump(model, model_path)
    
    print(f"✅ Success! Model is saved at: {model_path}")

if __name__ == "__main__":
    train_model()