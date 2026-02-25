import sys
import os

# Add app to path
sys.path.append(os.path.dirname(__file__))

from app.schemas import (
    XAIRequest, DebtInput, OptimizerRequest,
    RLOptimizerRequest, ProfilerRequest, TransactionInput,
)
from app.routers.xai import explain_risk
from app.routers.optimizer import recommend, rl_recommend
from app.routers.profiler import analyze
import asyncio
import random
from datetime import datetime, timedelta

# --- Test XAI (Phase 4) ---

async def test_xai():
    print("=" * 60)
    print("Testing XAI Explain Risk (Phase 4)...")
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
    print(f"  Risk Level: {res.risk_level}")
    print(f"  Default Probability: {res.default_probability}")
    print(f"  Ranked Debts: {[d['name'] for d in res.ranked_debts]}")
    print("  ✅ XAI test passed")
    print()

# --- Test Deterministic Optimizer (Phase 4) ---

async def test_optimizer():
    print("=" * 60)
    print("Testing Deterministic Optimizer (Phase 4)...")
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
    print(f"  Months to free: {res.metrics['months_to_free']}")
    print(f"  Worst case: {res.metrics['worst_case_months']}")
    print(f"  Total interest: {res.metrics['total_interest']}")
    if res.monthly_plan:
        print(f"  First month allocations: {res.monthly_plan[0].allocations}")
    print("  ✅ Deterministic optimizer test passed")
    print()

# --- Test Profiler (Phase 5) ---

def generate_mock_transactions(num=200):
    """Generate 6 months of mock transaction data."""
    categories = ["food", "transport", "entertainment", "shopping", "bills", "healthcare", "education"]
    transactions = []
    base_date = datetime(2025, 1, 1)
    
    for i in range(num):
        day_offset = random.randint(0, 179)  # 6 months
        date = base_date + timedelta(days=day_offset)

        if random.random() < 0.15:  # 15% are income
            transactions.append(TransactionInput(
                amount=round(random.uniform(3000, 6000), 2),
                category="salary",
                date=date.strftime("%Y-%m-%d"),
                type="income",
            ))
        else:
            transactions.append(TransactionInput(
                amount=round(random.uniform(5, 500), 2),
                category=random.choice(categories),
                date=date.strftime("%Y-%m-%d"),
                type="expense",
            ))
    return transactions

async def test_profiler():
    print("=" * 60)
    print("Testing Behavioral Profiler (Phase 5 — K-Means)...")
    transactions = generate_mock_transactions(200)
    req = ProfilerRequest(transactions_6mo=transactions)
    res = await analyze(req)
    
    valid_archetypes = [
        "Consistent Saver", "Weekend Splurger", "Subscription Hoarder",
        "Feast-or-Famine", "Debt Juggler", "Insufficient Data"
    ]
    assert res.profile in valid_archetypes, f"Invalid archetype: {res.profile}"
    
    print(f"  Archetype: {res.profile}")
    print(f"  Traits: {res.traits[:2]}...")
    print(f"  Risk Impact: {res.risk_impact[:60]}...")
    print(f"  Tips: {res.actionable_tips[0][:60]}...")
    if res.features:
        print(f"  Features: { {k: round(v, 3) for k, v in res.features.items()} }")
    print("  ✅ Profiler test passed")
    print()

# --- Test RL Optimizer (Phase 6) ---

async def test_rl_optimizer():
    print("=" * 60)
    print("Testing RL Optimizer (Phase 6 — PPO)...")
    print("  (Training agent with 10K timesteps — this may take a few seconds)")
    req = RLOptimizerRequest(
        debts=[
            DebtInput(name="Credit Card", creditor="Bank A", balance=5000, interest_rate=24.0, minimum_payment=200, due_date="15", penalty_rate=5.0, stress_level=9),
            DebtInput(name="Student Loan", creditor="Gov", balance=15000, interest_rate=5.0, minimum_payment=150, due_date="30", stress_level=4)
        ],
        monthly_income=3000,
        monthly_expenses=1500,
        savings=1000,
        risk_tolerance="moderate",
        training_timesteps=10000,  # Keep low for test speed
    )
    res = await rl_recommend(req)
    
    print(f"  RL Months to free: {res.metrics['months_to_free']}")
    print(f"  RL Total interest: {res.metrics['total_interest']}")
    print(f"  Training episodes: {len(res.training_reward_curve)}")
    if res.monthly_plan:
        print(f"  First month allocations: {res.monthly_plan[0].allocations}")
    print(f"  RL vs Deterministic:")
    print(f"    Months saved: {res.rl_vs_deterministic.get('months_saved', 'N/A')}")
    print(f"    Interest saved: {res.rl_vs_deterministic.get('interest_saved', 'N/A')}")
    print("  ✅ RL Optimizer test passed")
    print()


if __name__ == "__main__":
    print("\n🚀 Finara ML Engine — Endpoint Tests\n")
    
    asyncio.run(test_xai())
    asyncio.run(test_optimizer())
    asyncio.run(test_profiler())
    asyncio.run(test_rl_optimizer())
    
    print("=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
