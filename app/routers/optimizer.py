from fastapi import APIRouter
from app.schemas import (
    OptimizerRequest, OptimizerResponse, MonthlyAllocation,
    RLOptimizerRequest, RLOptimizerResponse,
)
import copy
import numpy as np

router = APIRouter()

def calculate_ahp_priority(debts):
    # Find max values for normalization
    max_interest = max([d.interest_rate for d in debts]) if debts else 1.0
    max_interest = max(max_interest, 0.01)
    
    max_penalty = max([d.penalty_rate for d in debts]) if debts else 1.0
    max_penalty = max(max_penalty, 0.01)
    
    ranked = {}
    for d in debts:
        score_int = d.interest_rate / max_interest
        try:
            days = int(d.due_date) if d.due_date.isdigit() else 30
        except:
            days = 30
        days = max(1, min(days, 30))
        score_urg = 1.0 - (days / 30.0)
        score_pen = (d.penalty_rate or 0) / max_penalty
        score_stress = (d.stress_level or 5) / 10.0
        
        # Simplified weights for optimizer internal ranking
        priority = (score_int * 0.4) + (score_urg * 0.2) + (score_pen * 0.3) + (score_stress * 0.1)
        ranked[d.name] = priority
        
    return ranked

def simulate_repayment(debts_sim, income, expenses, std_dev, is_deterministic=True):
    current_debts = copy.deepcopy(debts_sim)
    month = 0
    allocations = []
    total_interest_paid = 0.0
    
    # max 120 months (10 years)
    while sum(d['balance'] for d in current_debts) > 0 and month < 120:
        month += 1
        
        if not is_deterministic:
            m_income = max(0, np.random.normal(income, income * std_dev))
            m_expenses = max(0, np.random.normal(expenses, expenses * std_dev))
        else:
            m_income = income
            m_expenses = expenses
            
        surplus = m_income - m_expenses
        
        month_alloc = {d['name']: 0.0 for d in current_debts}
        
        # 1. Pay minimums
        remaining_surplus = surplus
        for d in current_debts:
            if d['balance'] > 0:
                payment = min(d['balance'], d['min_pay'])
                if remaining_surplus >= payment:
                    month_alloc[d['name']] += payment
                    d['balance'] -= payment
                    remaining_surplus -= payment
                else:
                    if remaining_surplus > 0:
                        pay = min(d['balance'], remaining_surplus)
                        month_alloc[d['name']] += pay
                        d['balance'] -= pay
                        remaining_surplus = 0
                        
        # 2. Allocate remaining surplus to highest priority debt
        while remaining_surplus > 0:
            active_debts = [d for d in current_debts if d['balance'] > 0]
            if not active_debts:
                break
            target_debt = max(active_debts, key=lambda x: x['priority'])
            pay = min(target_debt['balance'], remaining_surplus)
            month_alloc[target_debt['name']] += pay
            target_debt['balance'] -= pay
            remaining_surplus -= pay
                
        # 3. Add monthly interest
        for d in current_debts:
            if d['balance'] > 0:
                monthly_interest = d['balance'] * d['int_rate']
                d['balance'] += monthly_interest
                total_interest_paid += monthly_interest
                
        # Record exact allocations for the response
        allocations.append({
            "month": month,
            "allocations": {k: round(v, 2) for k, v in month_alloc.items()}
        })
        
    return month, allocations, total_interest_paid


@router.post("/recommend", response_model=OptimizerResponse)
async def recommend(request: OptimizerRequest):
    """
    Cashflow-constrained deterministic optimization + Monte Carlo Scenario Simulation
    """
    ahp_priorities = calculate_ahp_priority(request.debts)
    
    # Prepare simulation dicts
    debts_sim = []
    for d in request.debts:
        debts_sim.append({
            "name": d.name,
            "balance": d.balance,
            "min_pay": d.minimum_payment,
            "int_rate": d.interest_rate,
            "priority": ahp_priorities.get(d.name, 0.0)
        })
        
    # Standard deviation mapped by risk tolerance
    risk_tol = request.risk_tolerance.lower() if request.risk_tolerance else "moderate"
    if risk_tol == "conservative" or risk_tol == "low":
        std_dev = 0.05
    elif risk_tol == "aggressive" or risk_tol == "high":
        std_dev = 0.25
    else:
        std_dev = 0.15
        
    # Run 1 deterministic path for the monthly plan
    exp_months, monthly_allocations_raw, expected_interest = simulate_repayment(
        debts_sim, request.monthly_income, request.monthly_expenses, std_dev, is_deterministic=True
    )
    
    # Run Monte Carlo Lite (100 paths) for metrics
    sim_months = []
    sim_interests = []
    for _ in range(100):
        m, _, i = simulate_repayment(
            debts_sim, request.monthly_income, request.monthly_expenses, std_dev, is_deterministic=False
        )
        sim_months.append(m)
        sim_interests.append(i)
        
    worst_case_month = int(np.percentile(sim_months, 95))
    expected_month = int(np.mean(sim_months))
    
    metrics = {
        "months_to_free": expected_month,
        "worst_case_months": worst_case_month,
        "total_interest": round(expected_interest, 2),
        "deterministic_months": exp_months,
    }
    
    monthly_plan = []
    for alloc in monthly_allocations_raw:
        monthly_plan.append(MonthlyAllocation(**alloc))

    return OptimizerResponse(
        monthly_plan=monthly_plan,
        metrics=metrics,
        training_curve=[]
    )


@router.post("/rl-recommend", response_model=RLOptimizerResponse)
async def rl_recommend(request: RLOptimizerRequest):
    """
    PPO Reinforcement Learning optimizer that trains an agent on-the-fly
    to discover optimal debt allocation strategies.

    This complements the deterministic `/recommend` endpoint by using
    a learning-based approach that can discover counter-intuitive
    strategies (e.g., paying a lower-priority debt first to free up
    minimum payment obligations).

    Training is lightweight (~10-50K timesteps) and completes in seconds.
    """
    from app.services.rl_optimizer import train_agent, get_rl_plan

    # Prepare debt dicts for the environment
    debts_env = []
    for d in request.debts:
        try:
            days = int(d.due_date) if d.due_date.isdigit() else 30
        except:
            days = 30
        debts_env.append({
            "name": d.name,
            "balance": d.balance,
            "interest_rate": d.interest_rate,
            "minimum_payment": d.minimum_payment,
            "days_due": max(1, min(days, 30)),
        })

    risk_tol = request.risk_tolerance or "moderate"
    timesteps = request.training_timesteps or 20000

    # 1. Train PPO agent
    model, reward_curve, env = train_agent(
        debts=debts_env,
        monthly_income=request.monthly_income,
        monthly_expenses=request.monthly_expenses,
        savings=request.savings,
        risk_tolerance=risk_tol,
        training_timesteps=timesteps,
    )

    # 2. Run inference for the RL plan
    rl_allocations, rl_metrics = get_rl_plan(model, env)

    # 3. Run deterministic plan for comparison
    ahp_priorities = calculate_ahp_priority(request.debts)
    debts_sim = []
    for d in request.debts:
        debts_sim.append({
            "name": d.name,
            "balance": d.balance,
            "min_pay": d.minimum_payment,
            "int_rate": d.interest_rate,
            "priority": ahp_priorities.get(d.name, 0.0)
        })
    det_months, _, det_interest = simulate_repayment(
        debts_sim, request.monthly_income, request.monthly_expenses,
        std_dev=0.0, is_deterministic=True,
    )

    # 4. Build comparison
    rl_vs_deterministic = {
        "rl_months": rl_metrics["months_to_free"],
        "deterministic_months": det_months,
        "rl_interest": rl_metrics["total_interest"],
        "deterministic_interest": round(det_interest, 2),
        "months_saved": det_months - rl_metrics["months_to_free"],
        "interest_saved": round(det_interest - rl_metrics["total_interest"], 2),
    }

    monthly_plan = [MonthlyAllocation(**a) for a in rl_allocations]

    return RLOptimizerResponse(
        monthly_plan=monthly_plan,
        metrics=rl_metrics,
        training_reward_curve=reward_curve,
        rl_vs_deterministic=rl_vs_deterministic,
    )


from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Snowball / Avalanche / Finara strategy comparison
# ---------------------------------------------------------------------------

def simulate_strategy(
    debts_list: List[Dict[str, Any]], 
    income: float, 
    expenses: float, 
    strategy: str = "finara", 
    ahp_priorities: Optional[Dict[str, float]] = None, 
    std_dev: float = 0.15, 
    is_deterministic: bool = True
) -> Tuple[int, float, List[Dict[str, Any]]]:
    """
    Generic simulation that supports three strategies:
      - 'snowball'  : excess goes to smallest balance first
      - 'avalanche' : excess goes to highest interest rate first
      - 'finara'    : excess goes to highest AHP priority first (default)
    Returns (months, total_interest_paid, allocations).
    """
    current = copy.deepcopy(debts_list)
    month = 0
    total_interest = 0.0
    allocations = []

    while sum(d['balance'] for d in current) > 0 and month < 120:
        month += 1
        
        if not is_deterministic:
            m_income = max(0, np.random.normal(income, income * std_dev))
            m_expenses = max(0, np.random.normal(expenses, expenses * std_dev))
        else:
            m_income = income
            m_expenses = expenses
            
        surplus = m_income - m_expenses
        remaining = surplus
        
        month_alloc = {d['name']: 0.0 for d in current}

        for d in current:
            if d['balance'] > 0:
                payment = min(d['balance'], d['min_pay'])
                if remaining >= payment:
                    d['balance'] -= payment
                    remaining -= payment
                    month_alloc[d['name']] += payment
                else:
                    if remaining > 0:
                        pay = min(d['balance'], remaining)
                        d['balance'] -= pay
                        remaining = 0
                        month_alloc[d['name']] += pay

        # 2. Allocate excess based on strategy
        while remaining > 0:
            active = [d for d in current if d['balance'] > 0]
            if not active:
                break
            if strategy == "snowball":
                target = min(active, key=lambda x: x['balance'])
            elif strategy == "avalanche":
                target = max(active, key=lambda x: x['int_rate'])
            else:  # finara
                target = max(active, key=lambda x: x.get('priority', 0))
            pay = min(target['balance'], remaining)
            target['balance'] -= pay
            remaining -= pay
            month_alloc[target['name']] += pay

        # 3. Accrue monthly interest
        for d in current:
            if d['balance'] > 0:
                mi = d['balance'] * d['int_rate']
                d['balance'] += mi
                total_interest += mi

        if is_deterministic:
            allocations.append({
                "month": month,
                "allocations": {k: round(v, 2) for k, v in month_alloc.items()}
            })

    return month, round(total_interest, 2), allocations


from app.schemas import StrategyComparisonResponse, StrategyResult

@router.post("/compare-strategies", response_model=StrategyComparisonResponse)
async def compare_strategies(request: OptimizerRequest):
    """
    Compare Snowball, Avalanche, and Finara Optimized strategies side-by-side.
    Returns months_to_free and total_interest for each strategy.
    """
    ahp_priorities = calculate_ahp_priority(request.debts)

    base_debts = []
    for d in request.debts:
        base_debts.append({
            "name": d.name,
            "balance": d.balance,
            "min_pay": d.minimum_payment,
            "int_rate": d.interest_rate,
            "priority": ahp_priorities.get(d.name, 0.0),
        })

    risk_tol = request.risk_tolerance.lower() if request.risk_tolerance else "moderate"
    if risk_tol in ("conservative", "low"):
        std_dev = 0.05
    elif risk_tol in ("aggressive", "high"):
        std_dev = 0.25
    else:
        std_dev = 0.15

    results = {}
    for strategy in ["snowball", "avalanche", "finara"]:
        months, interest, allocs = simulate_strategy(
            base_debts,
            request.monthly_income,
            request.monthly_expenses,
            strategy=strategy,
            ahp_priorities=ahp_priorities,
            std_dev=std_dev,
            is_deterministic=True
        )
        
        # Monte Carlo for worst_case_months
        sim_months = []
        for _ in range(100):
            m, _, _ = simulate_strategy(
                base_debts,
                request.monthly_income,
                request.monthly_expenses,
                strategy=strategy,
                ahp_priorities=ahp_priorities,
                std_dev=std_dev,
                is_deterministic=False
            )
            sim_months.append(m)
        worst_case_months = int(np.percentile(sim_months, 95))
        
        monthly_plan = [MonthlyAllocation(**a) for a in allocs]

        results[strategy] = StrategyResult(
            months_to_free=months,
            total_interest=interest,
            worst_case_months=worst_case_months,
            monthly_plan=monthly_plan
        )

    recommendation = min(results.keys(), key=lambda k: results[k].total_interest)

    return StrategyComparisonResponse(
        strategies=results,
        recommendation=recommendation
    )
