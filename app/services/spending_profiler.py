"""
Phase 5 — K-Means Behavioral Spending Profiler

Classifies users into spending archetypes based on behavioral features
using K-Means clustering with pre-defined archetype centroids.
"""

import numpy as np

# Pre-defined archetype centroids (K=5)
# Each centroid represents the "average" feature vector for that archetype.
# Order: [weekend_spend_ratio, category_entropy, income_stability,
#          expense_trend, savings_rate, recurring_ratio, impulse_score]
ARCHETYPE_CENTROIDS = np.array([
    # Consistent Saver: low weekend, moderate diversity, stable income,
    # flat/negative trend, high savings, moderate recurring, low impulse
    [0.20, 0.60, 0.10, -0.05, 0.35, 0.40, 0.10],

    # Weekend Splurger: high weekend, moderate diversity, moderate stability,
    # positive trend, low savings, low recurring, high impulse
    [0.55, 0.50, 0.20, 0.15, 0.05, 0.20, 0.45],

    # Subscription Hoarder: low weekend, low diversity, stable income,
    # flat trend, moderate savings, very high recurring, low impulse
    [0.15, 0.35, 0.15, 0.02, 0.15, 0.75, 0.08],

    # Feast-or-Famine: moderate weekend, high diversity, very high instability,
    # variable trend, negative savings, low recurring, moderate impulse
    [0.30, 0.70, 0.60, 0.10, -0.10, 0.15, 0.25],

    # Debt Juggler: moderate weekend, moderate diversity, moderate instability,
    # positive trend, very negative savings, moderate recurring, moderate impulse
    [0.25, 0.55, 0.35, 0.20, -0.20, 0.30, 0.30],
])

ARCHETYPES = [
    {
        "name": "Consistent Saver",
        "traits": [
            "High savings rate relative to income",
            "Stable and predictable spending patterns",
            "Low impulse purchase frequency",
            "Disciplined budget adherence",
        ],
        "risk_impact": "Reduces default probability by ~10%. Strong financial buffer.",
        "tips": [
            "Your habits are strong — consider allocating surplus to your highest-interest debt",
            "Look into high-yield savings accounts to maximize your emergency buffer",
            "Consider increasing retirement contributions while your cashflow is stable",
        ],
    },
    {
        "name": "Weekend Splurger",
        "traits": [
            "Significantly higher spending on weekends",
            "Frequent small impulse purchases",
            "Entertainment and dining dominate weekend categories",
            "Savings rate below average",
        ],
        "risk_impact": "Increases cashflow volatility by ~15%. Weekend spending spikes reduce payment reliability.",
        "tips": [
            "Set a fixed weekend spending cap and track it separately",
            "Use the 24-hour rule: wait a day before non-essential purchases over a certain threshold (e.g. IDR 200K)",
            "Move 'fun money' to a separate account to create a natural spending limit",
            "Try meal prepping — dining out is likely your largest weekend expense",
        ],
    },
    {
        "name": "Subscription Hoarder",
        "traits": [
            "Very high recurring transaction ratio",
            "Many small monthly charges spread across services",
            "Low category diversity — spending concentrated in subscriptions",
            "Steady but persistent cashflow drain",
        ],
        "risk_impact": "Subscriptions create fixed obligations that reduce surplus available for debt payments.",
        "tips": [
            "Audit all subscriptions — cancel services unused in the last 30 days",
            "Consolidate streaming services: rotate between them monthly instead of keeping all active",
            "Check for duplicate or overlapping services (e.g., multiple cloud storage plans)",
            "Potential monthly savings: IDR 200K–500K (or equivalent) by cutting 2-3 unused subscriptions",
        ],
    },
    {
        "name": "Feast-or-Famine",
        "traits": [
            "Highly variable monthly income",
            "Spending swings dramatically between months",
            "Diverse spending categories with no clear pattern",
            "Emergency buffer is thin or non-existent",
        ],
        "risk_impact": "Highest volatility contributor. Income instability significantly increases default probability.",
        "tips": [
            "Build a 3-month emergency buffer as your top priority",
            "Use income smoothing: save 30% of 'feast' months for 'famine' months",
            "Set up automatic transfers to a buffer account every payday",
            "Consider diversifying income sources to reduce volatility",
        ],
    },
    {
        "name": "Debt Juggler",
        "traits": [
            "Multiple active debts consuming significant income",
            "Growing expense trend (spending outpacing income)",
            "Negative or near-zero savings rate",
            "Moderate instability across all metrics",
        ],
        "risk_impact": "Highest risk archetype. Multiple debt obligations with negative savings trend signals potential default.",
        "tips": [
            "Focus on one debt at a time — follow the AI optimization plan to eliminate the highest-priority debt first",
            "Debt consolidation may reduce total interest — consult the risk assessment for details",
            "Cut discretionary spending by 20% and redirect entirely to debt payments",
            "Avoid taking on any new debt until at least one existing debt is fully paid off",
        ],
    },
]

FEATURE_NAMES = [
    "weekend_spend_ratio",
    "category_entropy",
    "income_stability",
    "expense_trend",
    "savings_rate",
    "recurring_ratio",
    "impulse_score",
]


def classify_profile(features: dict[str, float]) -> dict:
    """
    Classify a user's spending profile by finding the nearest archetype centroid.

    Args:
        features: dict with 7 behavioral feature values from feature_engineering.extract_features()

    Returns:
        dict with keys: profile, traits, risk_impact, actionable_tips, features
    """
    # Build feature vector in the correct order
    feature_vector = np.array([features.get(f, 0.0) for f in FEATURE_NAMES])

    # Compute Euclidean distance to each archetype centroid
    distances = np.linalg.norm(ARCHETYPE_CENTROIDS - feature_vector, axis=1)

    # Nearest centroid = assigned archetype
    cluster_idx = int(np.argmin(distances))
    archetype = ARCHETYPES[cluster_idx]

    return {
        "profile": archetype["name"],
        "traits": archetype["traits"],
        "risk_impact": archetype["risk_impact"],
        "actionable_tips": archetype["tips"],
        "features": features,
        "cluster_distances": {
            ARCHETYPES[i]["name"]: round(float(d), 4)
            for i, d in enumerate(distances)
        },
    }
