from fastapi import APIRouter
from app.schemas import OptimizerRequest, OptimizerResponse

router = APIRouter()


@router.post("/recommend", response_model=OptimizerResponse)
async def recommend(request: OptimizerRequest):
    """
    Train a PPO RL agent on the user's financial data and return
    an optimal month-by-month debt payment plan.
    """
    # TODO: Implement in Phase 5
    # 1. Build Gymnasium environment from user's debts
    # 2. Train PPO agent (~10K episodes)
    # 3. Extract learned policy as monthly allocations
    # 4. Compare against snowball/avalanche baselines
    return OptimizerResponse(
        monthly_plan=[],
        metrics={
            "months_to_free": 0,
            "total_interest": 0,
            "vs_snowball": "N/A",
            "vs_avalanche": "N/A",
        },
        training_curve=[],
    )
