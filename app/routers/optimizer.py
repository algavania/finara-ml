from fastapi import APIRouter
from app.schemas import OptimizerRequest, OptimizerResponse, MonthlyAllocation
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
        if remaining_surplus > 0:
            active_debts = [d for d in current_debts if d['balance'] > 0]
            if active_debts:
                target_debt = max(active_debts, key=lambda x: x['priority'])
                pay = min(target_debt['balance'], remaining_surplus)
                month_alloc[target_debt['name']] += pay
                target_debt['balance'] -= pay
                remaining_surplus -= pay
                
        # 3. Add monthly interest
        for d in current_debts:
            if d['balance'] > 0:
                monthly_interest = d['balance'] * (d['int_rate'] / 100.0 / 12.0)
                d['balance'] += monthly_interest
                total_interest_paid += monthly_interest
                
        # Calculate percentages for the response
        total_paid_this_month = sum(month_alloc.values())
        if total_paid_this_month > 0:
            percentages = {k: round(v / total_paid_this_month, 4) for k, v in month_alloc.items()}
        else:
            percentages = {k: 0.0 for k in month_alloc.keys()}
            
        allocations.append({
            "month": month,
            "allocations": percentages
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
        training_curve=[]  # Left blank as we moved to a deterministic simulation over RL in Phase 4
    )
