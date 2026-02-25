from fastapi import APIRouter
from app.schemas import ProfilerRequest, ProfilerResponse
from app.services.feature_engineering import extract_features
from app.services.spending_profiler import classify_profile

router = APIRouter()


@router.post("/analyze", response_model=ProfilerResponse)
async def analyze(request: ProfilerRequest):
    """
    Cluster user into a spending archetype using K-Means on
    behavioral features extracted from 6 months of transactions.

    Returns:
    - Archetype name (e.g., "Weekend Splurger")
    - Defining traits
    - Risk impact description
    - Personalized actionable tips
    - Raw behavioral feature values (for radar chart)
    """
    # Convert Pydantic models to dicts
    transactions = [t.model_dump() for t in request.transactions_6mo]

    # Minimum data check
    if len(transactions) < 10:
        return ProfilerResponse(
            profile="Insufficient Data",
            traits=["Not enough transaction data to analyze"],
            risk_impact="N/A",
            actionable_tips=[
                "Add at least 3 months of transactions for accurate profiling",
                "Include both income and expense transactions",
            ],
            features={},
            cluster_distances={},
        )

    # 1. Extract behavioral features
    features = extract_features(transactions)

    # 2. Classify into archetype via K-Means nearest centroid
    result = classify_profile(features)

    return ProfilerResponse(
        profile=result["profile"],
        traits=result["traits"],
        risk_impact=result["risk_impact"],
        actionable_tips=result["actionable_tips"],
        features=result["features"],
        cluster_distances=result["cluster_distances"],
    )
