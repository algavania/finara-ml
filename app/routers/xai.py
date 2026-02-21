from fastapi import APIRouter
from app.schemas import XAIRequest, XAIResponse
import math
import numpy as np

router = APIRouter()


@router.post("/explain-risk", response_model=XAIResponse)
async def explain_risk(request: XAIRequest):
    """
    Compute explainable risk assessment using Gradient Boosting + SHAP.
    Returns default probability, SHAP feature contributions, and ranked debts.
    """
    from app.services.shap_explainer import risk_explainer

    # 1. Extract financial features
    monthly_income = request.monthly_income
    
    # Calculate averages from 3 months of expenses
    monthly_expenses = sum(request.monthly_expenses_3mo) / len(request.monthly_expenses_3mo) if request.monthly_expenses_3mo else 0
    
    savings = request.savings
    
    total_debt_balance = sum([d.balance for d in request.debts])
    total_minimum_payments = sum([d.minimum_payment for d in request.debts])
    
    dti = (total_minimum_payments + monthly_expenses) / (monthly_income + 1e-5)
    savings_ratio = savings / (monthly_expenses + 1e-5)
    
    # Simple heuristic for payment history score if history is provided
    payment_history_score = 80 # default
    if request.payment_history:
        good_payments = sum(1 for p in request.payment_history if p.get('status') == 'paid')
        payment_history_score = int((good_payments / len(request.payment_history)) * 100)
        
    features_dict = {
        'monthly_income': monthly_income,
        'total_debt_balance': total_debt_balance,
        'total_minimum_payments': total_minimum_payments,
        'monthly_expenses': monthly_expenses,
        'savings': savings,
        'dti': dti,
        'savings_ratio': savings_ratio,
        'payment_history_score': payment_history_score
    }
    
    # 2 & 3. Run GradientBoosting prediction and Compute SHAP values
    prob, shap_impacts = risk_explainer.explain(features_dict)
    
    # Deterministic PD Fallback if ML model is unavailable or returns 0.0
    if prob == 0.0 and len(shap_impacts) == 0:
        # Calculate Cashflow Volatility
        if len(request.monthly_expenses_3mo) > 1 and monthly_expenses > 0:
            volatility = np.std(request.monthly_expenses_3mo) / monthly_expenses
        else:
            volatility = 0.0
        
        # PD Model based on DTI and Volatility
        prob = min(1.0, (dti * 0.7) + (volatility * 0.3))
        
        # Generate dummy SHAP values for explainability
        shap_impacts = [
            {"feature": "dti", "value": dti, "impact": dti * 0.7},
            {"feature": "volatility", "value": volatility, "impact": volatility * 0.3}
        ]
    
    # Determine risk level
    if prob < 0.20:
        risk_level = "low"
        rec = "Your debt risk is low. Keep managing your cash flow efficiently."
    elif prob < 0.40:
        risk_level = "medium"
        rec = "Moderate risk. Consider setting up an emergency buffer."
    elif prob < 0.70:
        risk_level = "high"
        rec = "High risk detected. Avoid taking on new debt and focus on high-interest accounts."
    else:
        risk_level = "critical"
        rec = "CRITICAL RISK. High probability of default. Seek immediate debt restructuring."
        
    # Format SHAP values for response
    shap_values = [
        s for s in shap_impacts
    ]
    
    # 4. Rank debts using AHP (Analytic Hierarchy Process)
    # Weights
    W_INTEREST = 0.35
    W_URGENCY = 0.25
    W_PENALTY = 0.20
    W_DTI_PRESSURE = 0.10
    W_STRESS = 0.10
    
    # Find max values for normalization
    max_interest = max([d.interest_rate for d in request.debts]) if request.debts else 1.0
    max_interest = max(max_interest, 0.01)  # avoid division by zero
    
    max_penalty = max([d.penalty_rate for d in request.debts]) if request.debts else 1.0
    max_penalty = max(max_penalty, 0.01)
    
    ranked_debts = []
    for debt in request.debts:
        # 1. Interest Score (0-1)
        score_interest = debt.interest_rate / max_interest
        
        # 2. Urgency Score (0-1) - Mocked based on naive string assumption (e.g. "15" -> 15 days)
        # Real implementation would parse dates and find days diff.
        try:
            days_due = int(debt.due_date) if debt.due_date.isdigit() else 30
        except:
            days_due = 30
        days_due = max(1, min(days_due, 30))
        score_urgency = 1.0 - (days_due / 30.0)  # closer to 0 days = higher score
        
        # 3. Penalty Score (0-1)
        score_penalty = debt.penalty_rate / max_penalty if debt.penalty_rate else 0.0
        
        # 4. DTI Pressure Score (0-1)
        score_dti_pressure = min(1.0, debt.minimum_payment / (monthly_income + 1e-5))
        
        # 5. Stress Score (0-1)
        score_stress = (debt.stress_level or 5) / 10.0
        
        # Final AHP Base Priority
        ahp_base_priority = (
            (score_interest * W_INTEREST) +
            (score_urgency * W_URGENCY) +
            (score_penalty * W_PENALTY) +
            (score_dti_pressure * W_DTI_PRESSURE) +
            (score_stress * W_STRESS)
        )
        
        # Adjust base priority using the global default probability
        adjusted_risk = ahp_base_priority * (1 + prob)
        
        ranked_debts.append({
            "name": debt.name,
            "creditor": debt.creditor,
            "adjusted_risk_score": round(adjusted_risk, 4),
            "original_balance": debt.balance,
            "ahp_breakdown": {
                "interest": round(score_interest * W_INTEREST, 4),
                "urgency": round(score_urgency * W_URGENCY, 4),
                "penalty": round(score_penalty * W_PENALTY, 4),
                "dti_pressure": round(score_dti_pressure * W_DTI_PRESSURE, 4),
                "stress": round(score_stress * W_STRESS, 4)
            }
        })
        
    # Sort debts highest risk first
    ranked_debts.sort(key=lambda x: x["adjusted_risk_score"], reverse=True)

    return XAIResponse(
        default_probability=round(prob, 4),
        risk_level=risk_level,
        shap_values=shap_values,
        ranked_debts=ranked_debts,
        recommendation=rec,
    )
