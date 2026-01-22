import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score

def check_overfitting(model, X_train, y_train, X_test, y_test):
    # حساب الدقة على بيانات التدريب
    train_preds = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)

    # حساب الدقة على بيانات الاختبار
    test_preds = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)

    print(f"\n--- 📉 Overfitting Check ---")
    print(f"Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    
    diff = (train_acc - test_acc) * 100
    if diff > 10:
        print(f"⚠️ Warning: High Overfitting detected! Difference: {diff:.2f}%")
    else:
        print(f"✅ Model is generalizing well. Difference: {diff:.2f}%")

def evaluate_model():
    # 1. تحميل كل البيانات
    X_train = pd.read_csv('data/processed/X_train.csv').values
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    X_test = pd.read_csv('data/processed/X_test.csv').values
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

    # 2. تحميل الموديل
    model = joblib.load('models/breast_cancer_model.pkl')

    # 3. حساب الأداء العام والـ Overfitting
    check_overfitting(model, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    evaluate_model()