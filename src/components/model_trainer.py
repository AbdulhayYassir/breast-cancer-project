import pandas as pd
import joblib
import os

from src.components.model import MyDecisionTree 

def train_model():
    print("--- Starting Model Training Sprint ---")
    
    X_train = pd.read_csv('data/processed/X_train.csv').values
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    
    model = MyDecisionTree(max_depth=3) 
    
    print("Fitting the model with Information Gain...")
    model.fit(X_train, y_train)
    
    os.makedirs('models', exist_ok=True)
    model_path = 'models/breast_cancer_model.pkl'
    joblib.dump(model, model_path)
    
    print(f"✅ Success! Model is saved at: {model_path}")

if __name__ == "__main__":
    train_model()