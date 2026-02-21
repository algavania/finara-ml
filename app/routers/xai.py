from fastapi import APIRouter
from app.schemas import XAIRequest, XAIResponse

router = APIRouter()


@router.post("/explain-risk", response_model=XAIResponse)
async def explain_risk(request: XAIRequest):
    """
    Compute explainable risk assessment using Gradient Boosting + SHAP.
    Returns default probability, SHAP feature contributions, and ranked debts.
    """
    # TODO: Implement in Phase 2
    # 1. Extract financial features (feature_eng.py)
    # 2. Run GradientBoosting prediction (risk_model.py)
    # 3. Compute SHAP values (shap_explainer.py)
    # 4. Rank debts by adjusted risk
    return XAIResponse(
        default_probability=0.0,
        risk_level="low",
        shap_values=[],
        ranked_debts=[],
        recommendation="Risk assessment not yet implemented",
    )
