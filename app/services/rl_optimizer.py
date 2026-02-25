"""
Phase 6 — PPO Reinforcement Learning Optimizer

Trains a PPO agent to discover optimal debt allocation strategies
using the DebtPayoffEnv Gymnasium environment.
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from app.services.debt_environment import DebtPayoffEnv


class RewardTracker(BaseCallback):
    """Callback to track episode rewards during training."""

    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._current_reward = 0.0

    def _on_step(self) -> bool:
        # Accumulate rewards
        if self.locals.get("rewards") is not None:
            self._current_reward += float(np.mean(self.locals["rewards"]))

        # Check for episode end
        if self.locals.get("dones") is not None:
            dones = self.locals["dones"]
            if any(dones):
                self.episode_rewards.append(round(self._current_reward, 2))
                self._current_reward = 0.0

        return True


def train_agent(
    debts: list[dict],
    monthly_income: float,
    monthly_expenses: float,
    savings: float,
    risk_tolerance: str = "moderate",
    training_timesteps: int = 20000,
) -> tuple[PPO, list[float], DebtPayoffEnv]:
    """
    Train a PPO agent on the debt repayment environment.

    Returns:
        (trained_model, reward_curve, environment)
    """
    # Map risk tolerance to income std dev
    std_map = {"conservative": 0.05, "low": 0.05, "moderate": 0.15, "aggressive": 0.25, "high": 0.25}
    income_std = std_map.get(risk_tolerance.lower(), 0.15)

    env = DebtPayoffEnv(
        debts=debts,
        monthly_income=monthly_income,
        monthly_expenses=monthly_expenses,
        savings=savings,
        income_std=income_std,
    )

    # PPO with MlpPolicy — lightweight for in-request training
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        verbose=0,
    )

    reward_tracker = RewardTracker()
    model.learn(total_timesteps=training_timesteps, callback=reward_tracker)

    return model, reward_tracker.episode_rewards, env


def get_rl_plan(
    model: PPO,
    env: DebtPayoffEnv,
) -> tuple[list[dict], dict]:
    """
    Run inference with a trained PPO model to produce a deterministic
    month-by-month allocation plan.

    Returns:
        (monthly_allocations, metrics)
    """
    obs, _ = env.reset()
    total_interest = 0.0
    allocations = []

    for _ in range(env.max_months):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    allocations = env.monthly_allocations
    months = env.months_elapsed
    total_interest = env.total_interest_paid

    metrics = {
        "months_to_free": months,
        "total_interest": round(total_interest, 2),
        "debts_cleared": env.debts_cleared,
        "all_paid": all(d["balance"] <= 0 for d in env.debts),
    }

    return allocations, metrics
