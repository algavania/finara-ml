import sys
import os

# Add app to path
sys.path.append(os.path.dirname(__file__))

from app.schemas import XAIRequest, DebtInput, OptimizerRequest
from app.routers.xai import explain_risk
from app.routers.optimizer import recommend
import asyncio

async def test_xai():
    print("Testing XAI Explain Risk...")
    req = XAIRequest(
        debts=[
            DebtInput(name="Credit Card", creditor="Bank A", balance=5000, interest_rate=24.0, minimum_payment=200, due_date="15", penalty_rate=5.0, stress_level=9),
            DebtInput(name="Student Loan", creditor="Gov", balance=15000, interest_rate=5.0, minimum_payment=150, due_date="30", stress_level=4)
        ],
        monthly_income=3000,
        monthly_expenses_3mo=[1500, 1600, 1400],
        savings=1000
    )
    res = await explain_risk(req)
    print("XAI Response Risk Level:", res.risk_level)
    print("XAI Ranked Debts:", [d['name'] for d in res.ranked_debts])
    print("---")

async def test_opt():
    print("Testing Optimizer...")
    req = OptimizerRequest(
        debts=[
            DebtInput(name="Credit Card", creditor="Bank A", balance=5000, interest_rate=24.0, minimum_payment=200, due_date="15", penalty_rate=5.0, stress_level=9),
            DebtInput(name="Student Loan", creditor="Gov", balance=15000, interest_rate=5.0, minimum_payment=150, due_date="30", stress_level=4)
        ],
        monthly_income=3000,
        monthly_expenses=1500,
        savings=1000,
        risk_tolerance="moderate"
    )
    res = await recommend(req)
    print("Optimizer Metrics:", res.metrics)
    if res.monthly_plan:
        print("First Month Allocations:", res.monthly_plan[0].allocations)
    print("---")

if __name__ == "__main__":
    asyncio.run(test_xai())
    asyncio.run(test_opt())
