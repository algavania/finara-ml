import pickle
import numpy as np

class RiskExplainer:
    def __init__(self):
        self.explainer = None
        self.model = None
        self.load_models()

    def load_models(self):
        try:
            with open('trained_models/risk_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open('trained_models/shap_explainer.pkl', 'rb') as f:
                self.explainer = pickle.load(f)
        except Exception as e:
            print(f"Failed to load models: {e}")

    def explain(self, features_dict):
        """
        Calculates the Default Probability and SHAP values for the given features.
        """
        if self.model is None or self.explainer is None:
            self.load_models()
        
        if self.model is None or self.explainer is None:
             # return safe defaults if model fails to load
             return 0.0, []

        feature_names = [
            'monthly_income', 'total_debt_balance', 'total_minimum_payments',
            'monthly_expenses', 'savings', 'dti', 'savings_ratio', 'payment_history_score'
        ]

        # Convert dict to array
        X_array = np.array([[features_dict[name] for name in feature_names]])

        # Get probability
        prob = float(self.model.predict_proba(X_array)[0][1])

        # Get SHAP values
        shap_values_raw = self.explainer.shap_values(X_array)
        
        # Format the explanations
        explanations = []
        for i, name in enumerate(feature_names):
            explanations.append({
                "feature": name,
                "value": float(X_array[0][i]),
                "impact": float(shap_values_raw[0][i])
            })
            
        # Sort by impact absolute value (highest absolute impact first)
        explanations.sort(key=lambda x: abs(x["impact"]), reverse=True)
        
        return prob, explanations

risk_explainer = RiskExplainer()
