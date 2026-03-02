"""
Set an income goal for an income category (e.g. salary, side gig).
Uses create_category_spending_limit internally; only accepts income categories.
"""
from typing import Tuple

from penny.tool_funcs.create_category_spending_limit import create_category_spending_limit

INCOME_CATEGORIES = frozenset({
    "income_salary",
    "income_sidegig",
    "income_business",
    "income_interest",
})


def create_income_goal(
    category: str,
    granularity: str,
    start_date: str,
    end_date: str,
    amount: float,
    title: str,
) -> Tuple[bool, str]:
    """
    Set an income goal (target) for an income category. Use when the user wants
    to set a goal for how much they aim to earn in a category (e.g. salary, side gig).

    Only accepts income categories from OFFICIAL_CATEGORIES (income_salary,
    income_sidegig, income_business, income_interest). For spending categories,
    use create_category_spending_limit instead.

    Args:
        category: Income category slug (must be one of INCOME_CATEGORIES).
        granularity: "weekly", "monthly", or "yearly".
        start_date: YYYY-MM-DD. Can be empty string.
        end_date: YYYY-MM-DD. Can be empty string for ongoing.
        amount: Target amount to earn. Can be 0.0.
        title: Goal name/title.

    Returns:
        Tuple[bool, str]: (success, output_info)
    """
    if category not in INCOME_CATEGORIES:
        return False, (
            f"Category '{category}' is not an income category. "
            f"Income goals only support: {', '.join(sorted(INCOME_CATEGORIES))}."
        )
    return create_category_spending_limit(
        category=category,
        granularity=granularity,
        start_date=start_date,
        end_date=end_date,
        amount=amount,
        title=title,
    )
