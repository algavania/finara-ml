from pydantic import BaseModel
from typing import Optional


# --- XAI Schemas ---

class DebtInput(BaseModel):
    name: str
    creditor: str
    balance: float
    interest_rate: float
    minimum_payment: float
    due_date: str
    penalty_rate: Optional[float] = 0.0
    stress_level: Optional[int] = 5  # 1-10

class XAIRequest(BaseModel):
    debts: list[DebtInput]
    monthly_income: float
    monthly_expenses_3mo: list[float]  # last 3 months
    savings: float
    payment_history: Optional[list[dict]] = []

class SHAPValue(BaseModel):
    feature: str
    value: float
    impact: float

class XAIResponse(BaseModel):
    default_probability: float
    risk_level: str  # low, medium, high, critical
    shap_values: list[SHAPValue]
    ranked_debts: list[dict]
    recommendation: str


# --- Optimizer Schemas ---

class OptimizerRequest(BaseModel):
    debts: list[DebtInput]
    monthly_income: float
    monthly_expenses: float
    savings: float
    risk_tolerance: Optional[str] = "moderate"  # conservative, moderate, aggressive

class MonthlyAllocation(BaseModel):
    month: int
    allocations: dict[str, float]  # debt_name -> percentage

class OptimizerResponse(BaseModel):
    monthly_plan: list[MonthlyAllocation]
    metrics: dict
    training_curve: Optional[list[float]] = []


# --- Profiler Schemas ---

class TransactionInput(BaseModel):
    amount: float
    category: str
    date: str
    type: str  # income or expense
    day_of_week: Optional[int] = None

class ProfilerRequest(BaseModel):
    transactions_6mo: list[TransactionInput]

class ProfilerResponse(BaseModel):
    profile: str
    traits: list[str]
    risk_impact: str
    actionable_tips: list[str]
