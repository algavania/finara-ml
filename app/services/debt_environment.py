"""
Phase 6 — Gymnasium Debt Repayment Environment

A custom Gymnasium environment that models multi-debt repayment
as a sequential decision problem for PPO training.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DebtPayoffEnv(gym.Env):
    """
    Gymnasium environment for debt repayment optimization.

    State: [balance_1..N, rate_1..N, days_due_1..N, surplus, months_elapsed, buffer_ratio]
    Action: Continuous allocation percentages [alloc_1..N], softmax-normalized to sum=1
    Reward: -interest - penalties + debt_cleared_bonus - time_penalty
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        debts: list[dict],
        monthly_income: float,
        monthly_expenses: float,
        savings: float,
        income_std: float = 0.15,
        max_months: int = 120,
    ):
        super().__init__()

        self.initial_debts = debts  # list of {name, balance, interest_rate, minimum_payment, due_date}
        self.monthly_income = monthly_income
        self.monthly_expenses = monthly_expenses
        self.savings = savings
        self.income_std = income_std
        self.max_months = max_months
        self.n_debts = len(debts)

        # Action space: continuous allocation per debt [0,1] — will be softmax-normalized
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_debts,), dtype=np.float32
        )

        # State space: balances + rates + days_due + surplus + months_elapsed + buffer_ratio
        # = 3*N + 3 dimensions
        state_dim = 3 * self.n_debts + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        self.reset()

    def _get_state(self) -> np.ndarray:
        """Build the observation vector."""
        balances = [d["balance"] for d in self.debts]
        rates = [d["interest_rate"] for d in self.debts]  # already in decimal form (e.g. 0.0295)
        days_due = [d.get("days_due", 30) / 30.0 for d in self.debts]  # normalize to 0-1

        surplus = self.monthly_income - self.monthly_expenses
        buffer_ratio = self.savings / (self.monthly_expenses + 1e-9)

        # Normalize balances by initial total to keep values reasonable
        total_initial = sum(d["balance"] for d in self.initial_debts) + 1e-9
        norm_balances = [b / total_initial for b in balances]

        state = (
            norm_balances
            + rates
            + days_due
            + [surplus / (self.monthly_income + 1e-9), self.months_elapsed / self.max_months, min(buffer_ratio, 5.0) / 5.0]
        )
        return np.array(state, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Deep copy initial debts
        self.debts = []
        for d in self.initial_debts:
            self.debts.append({
                "name": d["name"],
                "balance": float(d["balance"]),
                "interest_rate": float(d["interest_rate"]),
                "minimum_payment": float(d["minimum_payment"]),
                "days_due": int(d.get("days_due", 30)),
            })

        self.months_elapsed = 0
        self.total_interest_paid = 0.0
        self.debts_cleared = 0
        self.monthly_allocations = []

        return self._get_state(), {}

    def step(self, action: np.ndarray):
        self.months_elapsed += 1

        # Softmax normalize actions to get allocation percentages
        exp_action = np.exp(action - np.max(action))  # numerical stability
        alloc_pct = exp_action / (exp_action.sum() + 1e-9)

        # Stochastic income/expense perturbation during training
        m_income = max(0, np.random.normal(self.monthly_income, self.monthly_income * self.income_std))
        m_expenses = max(0, np.random.normal(self.monthly_expenses, self.monthly_expenses * self.income_std))
        surplus = max(0, m_income - m_expenses)

        reward = 0.0
        month_alloc = {}

        # 1. Pay minimum payments first
        remaining = surplus
        for d in self.debts:
            if d["balance"] > 0:
                min_pay = min(d["balance"], d["minimum_payment"])
                if remaining >= min_pay:
                    d["balance"] -= min_pay
                    remaining -= min_pay
                    month_alloc[d["name"]] = min_pay
                else:
                    # Can't meet minimum → penalty
                    paid = min(d["balance"], remaining)
                    d["balance"] -= paid
                    remaining -= paid
                    month_alloc[d["name"]] = paid
                    reward -= 5.0  # penalty for missed minimum payment
            else:
                month_alloc[d["name"]] = 0.0

        # 2. Allocate remaining surplus according to agent's policy
        if remaining > 0:
            active_mask = np.array([1.0 if d["balance"] > 0 else 0.0 for d in self.debts])
            masked_alloc = alloc_pct * active_mask
            total_masked = masked_alloc.sum()
            if total_masked > 0:
                masked_alloc = masked_alloc / total_masked

            for i, d in enumerate(self.debts):
                if d["balance"] > 0:
                    extra = remaining * masked_alloc[i]
                    payment = min(d["balance"], extra)
                    d["balance"] -= payment
                    month_alloc[d["name"]] = month_alloc.get(d["name"], 0) + payment

        # 3. Accrue interest on remaining balances
        month_interest = 0.0
        for d in self.debts:
            if d["balance"] > 0:
                interest = d["balance"] * d["interest_rate"]
                d["balance"] += interest
                month_interest += interest

        self.total_interest_paid += month_interest
        
        # Normalize interest penalty by income so it doesn't break across currencies (USD vs IDR)
        interest_penalty_ratio = (month_interest / (self.monthly_income + 1e-9))
        reward -= interest_penalty_ratio * 5.0  # penalize interest accrual relative to income

        # 4. Bonus for clearing a debt
        for d in self.debts:
            if d["balance"] <= 0.01 and d["balance"] >= 0:
                d["balance"] = 0.0

        newly_cleared = sum(1 for d in self.debts if d["balance"] == 0.0) - self.debts_cleared
        if newly_cleared > 0:
            reward += 10.0 * newly_cleared
            self.debts_cleared += newly_cleared

        # 5. Time penalty
        reward -= 0.1

        # Record exact allocations
        self.monthly_allocations.append({
            "month": self.months_elapsed, 
            "allocations": {k: round(v, 2) for k, v in month_alloc.items()}
        })

        # Check termination
        all_paid = all(d["balance"] <= 0 for d in self.debts)
        timed_out = self.months_elapsed >= self.max_months

        terminated = all_paid
        truncated = timed_out and not all_paid

        if all_paid:
            reward += 50.0  # big bonus for full payoff

        return self._get_state(), float(reward), terminated, truncated, {}
