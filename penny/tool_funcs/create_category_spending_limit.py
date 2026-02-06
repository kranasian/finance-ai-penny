"""
Set a spending limit for a category (e.g. groceries, dining). Uses validate_goal and normalize_dates from create_budget_or_goal.
"""
from typing import Tuple

from penny.tool_funcs.create_budget_or_goal import validate_goal, normalize_dates


def create_category_spending_limit(
    category: str,
    granularity: str,
    start_date: str,
    end_date: str,
    amount: float,
    title: str,
) -> Tuple[bool, str]:
    """
    Set a spending limit (budget cap) for a category. Use when the user wants to limit how much they spend in a category (e.g. groceries, shopping, dining).

    Args:
        category: Category from OFFICIAL CATEGORIES (spending or income).
        granularity: "weekly", "monthly", or "yearly".
        start_date: YYYY-MM-DD. Can be empty string.
        end_date: YYYY-MM-DD. Can be empty string for ongoing.
        amount: Maximum amount to spend (the limit). Can be 0.0.
        title: Goal name/title.

    Returns:
        Tuple[bool, str]: (success, output_info)
    """
    goal = {
        "category": category,
        "type": "category",
        "granularity": granularity,
        "start_date": start_date,
        "end_date": end_date,
        "amount": amount,
        "title": title,
    }
    user_asks = validate_goal(goal)
    if user_asks:
        msg = "\n".join(user_asks)
        return False, msg
    goal = normalize_dates(goal)
    goal_name = goal.get("title") or goal.get("category") or "goal"
    msg = (
        f"Successfully created '{goal_name}' "
        f"from {goal['start_date']} to {goal['end_date']} "
        f"with target amount ${goal.get('amount', 0):.2f}."
    )
    return True, msg
