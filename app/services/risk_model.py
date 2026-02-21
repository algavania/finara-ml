import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import shap

def train_risk_model():
    print("Loading mock data...")
    try:
        df = pd.read_csv('data/mock_financial_data.csv')
    except FileNotFoundError:
        print("Error: Mock data not found. Please run mock_data_generator.py first.")
        return
        
    # Features and target
    features = [
        'monthly_income', 
        'total_debt_balance', 
        'total_minimum_payments',
        'monthly_expenses', 
        'savings', 
        'dti', 
        'savings_ratio', 
        'payment_history_score'
    ]
    
    X = df[features]
    y = df['is_default']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Gradient Boosting model...")
    model = GradientBoostingClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=4, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model AUC: {auc:.4f}, Accuracy: {acc:.4f}")
    
    # Save the model
    os.makedirs('trained_models', exist_ok=True)
    with open('trained_models/risk_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved to trained_models/risk_model.pkl")

    print("Building SHAP Explainer...")
    # Initialize the explainer using the trained model
    explainer = shap.TreeExplainer(model)
    
    # Save the explainer
    with open('trained_models/shap_explainer.pkl', 'wb') as f:
        pickle.dump(explainer, f)
    print("SHAP Explainer saved to trained_models/shap_explainer.pkl")

if __name__ == "__main__":
    train_risk_model()
