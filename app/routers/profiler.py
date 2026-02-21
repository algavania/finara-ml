from fastapi import APIRouter
from app.schemas import ProfilerRequest, ProfilerResponse

router = APIRouter()


@router.post("/analyze", response_model=ProfilerResponse)
async def analyze(request: ProfilerRequest):
    """
    Cluster user into a spending archetype using K-Means on
    behavioral features extracted from 6 months of transactions.
    """
    # TODO: Implement in Phase 5
    # 1. Extract behavioral features (feature_eng.py)
    # 2. Run K-Means clustering (clustering.py)
    # 3. Map cluster to archetype name + traits
    # 4. Generate actionable tips
    return ProfilerResponse(
        profile="Unknown",
        traits=["Not enough transaction data to analyze"],
        risk_impact="N/A",
        actionable_tips=["Add at least 3 months of transactions for profiling"],
    )
