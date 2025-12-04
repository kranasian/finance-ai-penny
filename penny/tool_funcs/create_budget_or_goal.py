from datetime import datetime
from typing import Tuple
import pandas as pd

# Valid granularities
VALID_GRANULARITIES = ["weekly", "monthly", "yearly"]

# Valid goal types
VALID_GOAL_TYPES = ["category", "credit_X_amount", "save_X_amount", "credit_0", "save_0"]


def validate_goal(goal: dict) -> list:
    """
    Validate a single goal dictionary and return list of user clarification prompts.
    
    Args:
        goal: Goal dictionary with keys: category, match_category, type, granularity, 
              start_date, end_date, amount, title, etc.
    
    Returns:
        List of clarification prompts (empty if valid)
    """
    user_asks = []
    goal_name = (goal.get("title") if goal.get("title")
                 else goal.get("category") if goal.get("category") else "")
    
    # Check granularity
    if "granularity" not in goal or goal["granularity"] not in VALID_GRANULARITIES:
        suffix = f" {goal_name} budget" if goal_name else ""
        user_asks.append(f"What time periods are you looking to track for this{suffix}, like monthly?")
    
    # Check amount
    if "amount" not in goal or goal["amount"] is None or float(goal.get("amount", 0)) < 0:
        suffix = f" of the {goal_name}?" if goal_name else "?"
        user_asks.append(f"What is the target amount{suffix}")
    
    # Check dates
    if "start_date" in goal and goal["start_date"]:
        try:
            datetime.strptime(goal["start_date"], "%Y-%m-%d")
        except ValueError:
            user_asks.append("Please clarify when do you want this to start?")
    
    if "end_date" in goal and goal["end_date"]:
        try:
            datetime.strptime(goal["end_date"], "%Y-%m-%d")
        except ValueError:
            user_asks.append("Please clarify when do you want this to end?")
    
    # Check date range validity
    if ("start_date" in goal and goal["start_date"] and 
        "end_date" in goal and goal["end_date"]):
        try:
            starting_date = datetime.strptime(goal["start_date"], "%Y-%m-%d")
            ending_date = datetime.strptime(goal["end_date"], "%Y-%m-%d")
            if ending_date < starting_date:
                user_asks.append("Please clarify the start and end dates for this, we might have reversed it.")
            else:
                min_days = 6 if goal.get("granularity") == "weekly" else 7
                if (ending_date - starting_date).days < min_days:
                    user_asks.append("Please clarify the start and end dates as it is too short. It needs to cover the full selected period.")
        except ValueError:
            pass
    
    return user_asks


def normalize_dates(goal: dict) -> dict:
    """
    Normalize start and end dates based on granularity.
    
    Args:
        goal: Goal dictionary
    
    Returns:
        Goal dictionary with normalized dates
    """
    granularity = goal.get("granularity")
    today = datetime.today()
    
    # Normalize start_date
    start_date = goal.get("start_date")
    if not start_date:
        start_date = today.strftime("%Y-%m-%d")
    
    if granularity == "weekly":
        start_date = (pd.to_datetime(start_date) + pd.DateOffset(days=1) - pd.offsets.Week(weekday=6)).date().strftime("%Y-%m-%d")
    elif granularity in ["monthly", "yearly"]:
        start_date = pd.to_datetime(start_date).replace(day=1).date().strftime("%Y-%m-%d")
    
    # Normalize end_date
    end_date = goal.get("end_date")
    if end_date and end_date.strip():
        if granularity == "weekly":
            end_date = (pd.to_datetime(end_date) + pd.offsets.Week(weekday=6) - pd.DateOffset(days=1)).date().strftime("%Y-%m-%d")
        elif granularity in ["monthly", "yearly"]:
            end_date = pd.to_datetime(end_date).replace(day=1) + pd.offsets.MonthEnd()
            end_date = end_date.date().strftime("%Y-%m-%d")
    else:
        end_date = "2099-12-31"
    
    goal["start_date"] = start_date
    goal["end_date"] = end_date
    return goal


def create_budget_or_goal(
    category: str,
    granularity: str,
    start_date: str,
    end_date: str,
    amount: float,
    title: str
) -> Tuple[bool, str]:
    """
    Create a budget or goal based on individual parameters.
    
    Args:
        category: The category from the OFFICIAL CATEGORY LIST. This is the primary category identifier.
        granularity: The time period for the goal. Must be one of: "weekly", "monthly", or "yearly". Can be empty string.
        start_date: The start date for the goal in YYYY-MM-DD format. Can be empty string.
        end_date: The end date for the goal in YYYY-MM-DD format. Can be empty string.
        amount: The target dollar amount for the specified category and granularity. Can be 0.0.
        title: Goal name/title.
    
    Returns:
        Tuple[bool, str]: (success, output_info)
        - success: True if goal/budget was created successfully, False if clarification is needed
        - output_info: Success message or clarification prompts
    """
    # Convert parameters to dictionary for processing
    goal = {
        "category": category,
        "type": "category",
        "granularity": granularity,
        "start_date": start_date,
        "end_date": end_date,
        "amount": amount,
        "title": title
    }
    
    # Validate the goal
    user_asks = validate_goal(goal)
    if user_asks:
        return False, "\n".join(user_asks)
    
    # Normalize dates
    goal = normalize_dates(goal)
    
    # Get goal name for confirmation message
    goal_name = (goal.get("title") if goal.get("title")
                 else goal.get("category") if goal.get("category") else "goal")
    
    # Build confirmation message
    confirmation_message = (
        f"Successfully created '{goal_name}' "
        f"from {goal['start_date']} to {goal['end_date']} "
        f"with target amount ${goal.get('amount', 0):.2f}."
    )
    
    return True, confirmation_message

