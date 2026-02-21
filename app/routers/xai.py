from fastapi import APIRouter
from app.schemas import XAIRequest, XAIResponse

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
        # using dict destructuring so it aligns with SHAPValue pydantic model automatically 
        s for s in shap_impacts
    ]
    
    # 4. Rank debts by adjusted risk
    ranked_debts = []
    for debt in request.debts:
        # Heavily penalize high interest and high penalties
        base_priority = debt.interest_rate + (debt.penalty_rate * 2) + (debt.stress_level / 2.0)
        
        # Adjust base priority using the global default probability 
        # (If likely to default, high-penalty debts become exponentially more urgent)
        adjusted_risk = base_priority * (1 + prob)
        
        ranked_debts.append({
            "name": debt.name,
            "creditor": debt.creditor,
            "adjusted_risk_score": round(adjusted_risk, 2),
            "original_balance": debt.balance
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
