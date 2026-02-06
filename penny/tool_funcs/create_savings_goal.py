"""
Create a savings goal. Uses validate_goal and normalize_dates from create_budget_or_goal.
Skips normalize_dates when granularity is not provided (so dates stay as given).

Goal types (see UserGoals.md):
- save_X_amount: Save toward a total amount by a date (e.g. $10000 by end of year). amount = total target; end_date = target date.
- save_0: Save a fixed amount per period (e.g. $200/month). amount = per-period amount; granularity required.
"""
from datetime import datetime
from typing import Tuple, Optional

from penny.tool_funcs.create_budget_or_goal import validate_goal, normalize_dates

VALID_SAVINGS_GOAL_TYPES = ("save_X_amount", "save_0")


def create_savings_goal(
    amount: float,
    end_date: str,
    title: str,
    goal_type: str = "save_X_amount",
    granularity: Optional[str] = None,
    start_date: str = "",
) -> Tuple[bool, str]:
    """
    Set a savings goal. Use when the user wants to save money (e.g. save $X per month, or save $X total by a date).

    Args:
        amount: For save_X_amount = total to save for the goal. For save_0 = amount to save per period.
        end_date: Target date (YYYY-MM-DD). For save_X_amount use target date or default; for save_0 use default if no deadline.
        title: Goal name/title.
        goal_type: "save_X_amount" (total by date) or "save_0" (amount per period). See UserGoals.md.
        granularity: "weekly", "monthly", or "yearly". Required for save_X_amount; for save_0 defaults to "monthly".
        start_date: Optional start date. Default empty → today's date.

    Returns:
        Tuple[bool, str]: (success, output_info)
    """
    print("\n" + "=" * 80, flush=True)
    print("[CREATE_SAVINGS_GOAL] Calling create_savings_goal", flush=True)
    print(f"  amount: {amount}, end_date: {end_date}, title: {title!r}, goal_type: {goal_type!r}", flush=True)
    print(f"  granularity: {granularity!r}, start_date: {start_date!r}", flush=True)
    print("=" * 80 + "\n", flush=True)

    goal_type = (goal_type or "").strip()
    if goal_type not in VALID_SAVINGS_GOAL_TYPES:
        msg = (
            "How would you like to save? For example: put away a set amount regularly (e.g. save $200 per month), "
            "or save toward a total amount by a date (e.g. save $5000 for a car by next year)?"
        )
        print("=" * 80, flush=True)
        print("[CREATE_SAVINGS_GOAL] Execution result:", flush=True)
        print("  success: False", flush=True)
        print(f"  message: {msg}", flush=True)
        print("=" * 80 + "\n", flush=True)
        return False, msg

    if not (start_date and start_date.strip()):
        start_date = datetime.today().strftime("%Y-%m-%d")
    gran_provided = granularity is not None and (granularity or "").strip()
    gran_for_goal = (granularity or "").strip() or "monthly"
    goal = {
        "category": None,
        "type": goal_type,
        "granularity": gran_for_goal,
        "start_date": start_date,
        "end_date": end_date,
        "amount": amount,
        "title": title,
    }
    user_asks = validate_goal(goal)
    if user_asks:
        msg = "\n".join(user_asks)
        print("=" * 80, flush=True)
        print("[CREATE_SAVINGS_GOAL] Execution result:", flush=True)
        print(f"  success: False", flush=True)
        print(f"  message: {msg}", flush=True)
        print("=" * 80 + "\n", flush=True)
        return False, msg
    if gran_provided:
        goal = normalize_dates(goal)
    goal_name = goal.get("title") or "goal"
    msg = (
        f"Successfully created '{goal_name}' "
        f"from {goal['start_date']} to {goal['end_date']} "
        f"with target amount ${goal.get('amount', 0):.2f}."
    )
    print("=" * 80, flush=True)
    print("[CREATE_SAVINGS_GOAL] Execution result:", flush=True)
    print(f"  success: True", flush=True)
    print(f"  message: {msg}", flush=True)
    print(f"  start_date: {goal['start_date']}, end_date: {goal['end_date']}", flush=True)
    print("=" * 80 + "\n", flush=True)
    return True, msg
