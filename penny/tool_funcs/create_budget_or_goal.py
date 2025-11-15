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
    
    # Check amount (for category, credit_X_amount, save_X_amount types)
    goal_type = goal.get("type", "category")
    if goal_type in ["category", "credit_X_amount", "save_X_amount"]:
        if "amount" not in goal or goal["amount"] is None or float(goal.get("amount", 0)) < 0:
            suffix = f" of the {goal_name}?" if goal_name else "?"
            user_asks.append(f"What is the target amount{suffix}")
    
    # Check amount for credit_0 and save_0 (target amount by date)
    if goal_type in ["credit_0", "save_0"]:
        if "amount" not in goal or goal["amount"] is None or float(goal.get("amount", 0)) <= 0:
            suffix = f" for {goal_name}?" if goal_name else "?"
            user_asks.append(f"What is the target amount{suffix}")
        if "end_date" not in goal or not goal.get("end_date"):
            user_asks.append("Please specify when you want to reach this goal by (end date).")
    
    # Check category (only required for category type)
    if goal_type == "category":
        if ("match_category" not in goal or not goal["match_category"]):
            user_asks.append(f"Could you clarify the category for {goal_name}?")
    
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


def create_budget_or_goal(goal_dicts: list) -> Tuple[bool, str]:
    """
    Create budgets/goals based on the list of goal dictionaries.
    
    Args:
        goal_dicts: A list of dictionaries, where each dictionary represents a spending goal/budget
                   with keys: category, match_category, match_caveats, clarification_needed, type,
                   granularity, start_date, end_date, amount, title, budget_or_goal
    
    Returns:
        Tuple[bool, str]: (success, output_info)
        - success: True if goals were created successfully, False if clarification is needed
        - output_info: Success message or clarification prompts
    """
    if not goal_dicts:
        return False, "No goals provided. Please specify at least one goal to create."
    
    # Check for clarifications needed
    clarification_results = [g.get("clarification_needed") for g in goal_dicts if g.get("clarification_needed")]
    if clarification_results:
        return False, "\n".join(clarification_results)
    
    # Validate all goals
    all_user_asks = []
    for goal in goal_dicts:
        user_asks = validate_goal(goal)
        if user_asks:
            all_user_asks.extend(user_asks)
    
    if all_user_asks:
        return False, "\n".join(all_user_asks)
    
    # Normalize dates and build confirmation messages
    # Count budgets and goals separately for the summary message
    budget_count = sum(1 for g in goal_dicts if g.get("budget_or_goal", "goal") == "budget")
    goal_count = len(goal_dicts) - budget_count
    
    summary_parts = []
    if budget_count > 0:
        summary_parts.append(f"{budget_count} budget{'s' if budget_count != 1 else ''}")
    if goal_count > 0:
        summary_parts.append(f"{goal_count} goal{'s' if goal_count != 1 else ''}")
    
    if summary_parts:
        confirmation_messages = [f"Successfully created {', '.join(summary_parts)}."]
    else:
        confirmation_messages = [f"Successfully created {len(goal_dicts)} budget(s) or goal(s)."]
    
    caveats_results = []
    
    for goal in goal_dicts:
        # Create a copy to avoid mutating the original
        goal_copy = goal.copy()
        goal_copy = normalize_dates(goal_copy)
        goal_name = (goal_copy.get("title") if goal_copy.get("title")
                     else goal_copy.get("category") if goal_copy.get("category") else "goal")
        
        # Get budget_or_goal from dict, validate and default to "goal" if invalid or missing
        budget_or_goal = goal_copy.get("budget_or_goal", "goal")
        if budget_or_goal not in ["budget", "goal"]:
            budget_or_goal = "goal"  # Default to "goal" if invalid
        
        confirmation_messages.append(
            f"Created {budget_or_goal} '{goal_name}' "
            f"from {goal_copy['start_date']} to {goal_copy['end_date']} "
            f"with target amount ${goal_copy.get('amount', 0):.2f}."
        )
        
        # Include match caveats if present
        if goal_copy.get("match_caveats"):
            caveats_results.append(goal_copy["match_caveats"])
    
    # Combine caveats and confirmations
    all_messages = caveats_results + confirmation_messages
    output_info = "\n".join([m for m in all_messages if m])
    
    return True, output_info

