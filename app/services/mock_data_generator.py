import pandas as pd
import numpy as np
import random
import os

# Create mock data for training the risk assessment model.
# In a real scenario, this would come from Finara's PostgreSQL database.

def generate_mock_data(num_samples=5000):
    np.random.seed(42)
    random.seed(42)
    
    data = []
    
    for _ in range(num_samples):
        monthly_income = np.random.lognormal(mean=8.5, sigma=0.8) # ~$4,900 mean
        monthly_income = max(500, min(monthly_income, 25000))
        
        # Debts
        num_debts = random.randint(1, 5)
        total_debt_balance = 0
        total_minimum_payments = 0
        
        for _ in range(num_debts):
            b = np.random.lognormal(mean=7.0, sigma=1.0) # ~$1,000 mean
            total_debt_balance += b
            total_minimum_payments += b * random.uniform(0.01, 0.05)
            
        # Expenses
        expense_ratio = random.uniform(0.3, 0.9)
        monthly_expenses = monthly_income * expense_ratio
        
        # Savings
        savings = np.random.lognormal(mean=7.0, sigma=1.5)
        savings = max(0, min(savings, 100000))
        
        # Derived features
        dti = (total_minimum_payments + monthly_expenses) / (monthly_income + 1e-5)
        savings_ratio = savings / (monthly_expenses + 1e-5)
        
        # History (mock a score from 0-100 indicating good payment history)
        payment_history_score = random.randint(30, 100)
        
        # Rules for 'Default' label (target variable)
        # Probability goes up if DTI > 0.8, savings < 1 month, history < 60
        prob_default = 0.05 # base
        
        if dti > 0.6: prob_default += 0.2
        if dti > 0.8: prob_default += 0.3
        
        if savings_ratio < 1.0: prob_default += 0.2
        elif savings_ratio > 3.0: prob_default -= 0.1
        
        if payment_history_score < 50: prob_default += 0.25
        elif payment_history_score > 90: prob_default -= 0.1
        
        prob_default = max(0.01, min(prob_default, 0.99))
        
        # Generate target label based on probability
        is_default = 1 if random.random() < prob_default else 0
        
        data.append({
            'monthly_income': monthly_income,
            'total_debt_balance': total_debt_balance,
            'total_minimum_payments': total_minimum_payments,
            'monthly_expenses': monthly_expenses,
            'savings': savings,
            'dti': dti,
            'savings_ratio': savings_ratio,
            'payment_history_score': payment_history_score,
            'is_default': is_default
        })
        
    df = pd.DataFrame(data)
    
    # Save to file
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/mock_financial_data.csv', index=False)
    print(f"Generated {num_samples} mock records in data/mock_financial_data.csv")
    
if __name__ == "__main__":
    generate_mock_data()
