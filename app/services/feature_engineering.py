"""
Phase 5 — Behavioral Feature Engineering

Extracts 7 behavioral features from 6 months of transaction data
for K-Means spending archetype classification.
"""

import numpy as np
from collections import Counter
from datetime import datetime


def _parse_date(date_str: str) -> datetime:
    """Parse ISO date string to datetime."""
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(date_str[:19], fmt)
        except ValueError:
            continue
    return datetime.now()


def _is_weekend(date_str: str) -> bool:
    """Check if a date falls on Saturday or Sunday."""
    dt = _parse_date(date_str)
    return dt.weekday() >= 5  # 5=Saturday, 6=Sunday


def _get_month_key(date_str: str) -> str:
    """Extract YYYY-MM from date string."""
    return date_str[:7]


def weekend_spend_ratio(transactions: list[dict]) -> float:
    """
    Ratio of weekend spending to total spending.
    High ratio → impulsive/leisure-driven spending.
    """
    expenses = [t for t in transactions if t["type"] == "expense"]
    if not expenses:
        return 0.0

    total = sum(t["amount"] for t in expenses)
    weekend = sum(t["amount"] for t in expenses if _is_weekend(t["date"]))

    return round(weekend / (total + 1e-9), 4)


def category_entropy(transactions: list[dict]) -> float:
    """
    Shannon entropy of expense category distribution.
    High entropy → diversified spending across many categories.
    Low entropy → concentrated in few categories.
    """
    expenses = [t for t in transactions if t["type"] == "expense"]
    if not expenses:
        return 0.0

    counts = Counter(t["category"] for t in expenses)
    total = sum(counts.values())
    probabilities = [c / total for c in counts.values()]

    entropy = -sum(p * np.log2(p + 1e-9) for p in probabilities)

    # Normalize to 0–1 range (max entropy = log2(num_categories))
    max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
    return round(entropy / (max_entropy + 1e-9), 4)


def income_stability(transactions: list[dict]) -> float:
    """
    Coefficient of variation of monthly income.
    Low value → stable employment/income.
    High value → irregular income (freelance, seasonal).
    """
    incomes = [t for t in transactions if t["type"] == "income"]
    if not incomes:
        return 1.0  # No income data = maximum instability

    monthly = {}
    for t in incomes:
        key = _get_month_key(t["date"])
        monthly[key] = monthly.get(key, 0) + t["amount"]

    if len(monthly) < 2:
        return 0.0

    values = list(monthly.values())
    mean_val = np.mean(values)
    std_val = np.std(values)

    return round(std_val / (mean_val + 1e-9), 4)


def expense_trend(transactions: list[dict]) -> float:
    """
    Linear regression slope of monthly expenses (normalized).
    Positive → spending is growing over time.
    Negative → spending is shrinking.
    """
    expenses = [t for t in transactions if t["type"] == "expense"]
    if not expenses:
        return 0.0

    monthly = {}
    for t in expenses:
        key = _get_month_key(t["date"])
        monthly[key] = monthly.get(key, 0) + t["amount"]

    if len(monthly) < 2:
        return 0.0

    sorted_months = sorted(monthly.keys())
    y_values = [monthly[m] for m in sorted_months]
    x_values = list(range(len(y_values)))

    # Simple linear regression slope
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    slope = np.polyfit(x, y, 1)[0]

    # Normalize: slope relative to mean expense
    mean_expense = np.mean(y)
    normalized = slope / (mean_expense + 1e-9)

    return round(float(np.clip(normalized, -1.0, 1.0)), 4)


def savings_rate(transactions: list[dict]) -> float:
    """
    Overall savings rate: (total_income - total_expenses) / total_income.
    Higher → healthier financial behavior.
    """
    total_income = sum(t["amount"] for t in transactions if t["type"] == "income")
    total_expense = sum(t["amount"] for t in transactions if t["type"] == "expense")

    if total_income == 0:
        return -1.0  # No income → negative savings indicator

    rate = (total_income - total_expense) / total_income
    return round(float(np.clip(rate, -1.0, 1.0)), 4)


def recurring_ratio(transactions: list[dict]) -> float:
    """
    Ratio of recurring transactions to total transactions.
    High ratio → predictable outflows (subscriptions, bills).
    """
    if not transactions:
        return 0.0

    # Detect recurring: same category + similar amount appearing monthly
    expenses = [t for t in transactions if t["type"] == "expense"]
    if not expenses:
        return 0.0

    # Group by category and check for monthly repetition
    by_category = {}
    for t in expenses:
        cat = t["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(t)

    recurring_count = 0
    for cat, txns in by_category.items():
        if len(txns) < 2:
            continue
        # Check if amounts are similar (within 20%) and span multiple months
        amounts = [t["amount"] for t in txns]
        months = set(_get_month_key(t["date"]) for t in txns)
        mean_amt = np.mean(amounts)
        is_similar = all(abs(a - mean_amt) / (mean_amt + 1e-9) < 0.2 for a in amounts)
        if is_similar and len(months) >= 2:
            recurring_count += len(txns)

    return round(recurring_count / (len(expenses) + 1e-9), 4)


def impulse_score(transactions: list[dict]) -> float:
    """
    Frequency of small, unplanned transactions relative to total.
    High score → poor spending discipline.
    """
    expenses = [t for t in transactions if t["type"] == "expense"]
    if not expenses:
        return 0.0

    amounts = [t["amount"] for t in expenses]
    median_amount = np.median(amounts)

    # "Small" = below 30% of median expense
    threshold = median_amount * 0.3
    small_count = sum(1 for a in amounts if a <= threshold)

    return round(small_count / (len(expenses) + 1e-9), 4)


def extract_features(transactions: list[dict]) -> dict[str, float]:
    """
    Extract all 7 behavioral features from a list of transactions.

    Each transaction dict should have: amount, category, date, type ('income'/'expense')

    Returns a dict with feature names as keys and float values.
    """
    return {
        "weekend_spend_ratio": weekend_spend_ratio(transactions),
        "category_entropy": category_entropy(transactions),
        "income_stability": income_stability(transactions),
        "expense_trend": expense_trend(transactions),
        "savings_rate": savings_rate(transactions),
        "recurring_ratio": recurring_ratio(transactions),
        "impulse_score": impulse_score(transactions),
    }
