from penny.tool_funcs.sandbox_logging import log
import json

def create_goal(goals: list[dict]) -> tuple[str, dict]:
  """
  Creates spending budgets or goals for financial categories.
  Supports creating single or multiple budgets/goals with weekly, monthly, or yearly frequencies.
  
  Args:
    goals: A list of goal dictionaries, each containing:
      - type: str - "category", "credit_X_amount", "save_X_amount", "credit_0", or "save_0" (required)
      - granularity: str - "weekly", "monthly", or "yearly" (required)
      - title: str - Goal title/name (required)
      - amount: float - Target dollar amount (required, >= 0)
      - start_date: str - Start date in YYYY-MM-DD format (required)
      - end_date: str - End date in YYYY-MM-DD format (optional, defaults to "2099-12-31")
      - category: str - Raw category from user input (optional, for type="category")
      - match_category: str - Official category name (required for type="category")
      - match_caveats: str - Matching constraints explanation (optional)
      - clarification_needed: str - Clarification prompt if needed (optional)
      - description: str - Goal description string (required)
      - account_id: int - Account ID for credit_X_amount/save_X_amount/credit_0/save_0 types (optional)
      - percent: float - Target percent (0-100) for credit_X_amount/save_X_amount types (optional)
  
  Returns:
    tuple[str, dict]: 
      - str: A natural language string confirming the creation of the budget/goal or requesting further clarification from the user.
      - dict: Metadata containing the goals created, with the following structure:
        {
          "goals": [
            {
              "title": str,           # Goal title
              "amount": float,        # Target amount
              "start_date": str,      # Start date in YYYY-MM-DD format
              "end_date": str,        # End date in YYYY-MM-DD format
              "category": str,        # Raw category from user input
              "matched_category": str, # Official category name
              "granularity": str,     # "weekly", "monthly", or "yearly"
              "type": str,            # "category", "credit_X_amount", "save_X_amount", "credit_0", or "save_0"
              "match_caveats": str or None,  # Matching caveats if any
              "clarification_needed": str or None,  # Clarification needed if any
              "description": str,    # Goal description
              "account_id": int or None,  # Account ID if applicable
              "percent": float or None   # Target percent if applicable
            },
            ...
          ]
        }
    
  Note:
    - Goal types:
      - "category": Spending goal based on category inflow/spending (requires match_category)
      - "credit_X_amount": Paying down credit X amount per period (requires account_id)
      - "save_X_amount": Saving X amount per period (requires account_id)
      - "credit_0": Paying down credit to target amount by end_date (requires account_id)
      - "save_0": Saving to target amount by end_date (requires account_id)
    - Granularity: "weekly", "monthly", or "yearly"
    - Weekly goals start on Sunday and end on Saturday
    - Monthly goals start on the 1st of the month
    - Yearly goals start on January 1st
    - Categories must match official category names (e.g., "transportation_car", "meals_dining_out")
    - If no end_date is specified, defaults to "2099-12-31"
    - percent is only valid for credit_X_amount/save_X_amount types
  """
  log(f"**Create Goal**: Processing {len(goals)} goal(s)")
  
  if not goals:
    response_message = "No goals provided."
    metadata = {"goals": []}
    log(f"**Goal Creation Failed**: No goals provided")
    return response_message, metadata
  
  try:
    created_goals = []
    caveats = []
    
    for i, goal_data in enumerate(goals):
      # Extract goal data
      goal_type = goal_data.get("type", "category")
      granularity = goal_data.get("granularity", "")
      title = goal_data.get("title", "")
      amount = goal_data.get("amount", 0.0)
      start_date = goal_data.get("start_date", "")
      end_date = goal_data.get("end_date", "")
      category = goal_data.get("category", "")
      match_category = goal_data.get("match_category", "")
      match_caveats = goal_data.get("match_caveats", None)
      clarification_needed = goal_data.get("clarification_needed", None)
      description = goal_data.get("description", "")
      account_id = goal_data.get("account_id", None)
      percent = goal_data.get("percent", None)
      
      log(f"**Processing Goal {i+1}**: type={goal_type}, granularity={granularity}, title={title}, amount={amount}, start_date={start_date}, end_date={end_date}, category={category}, match_category={match_category}, account_id={account_id}, percent={percent}")
      
      # Check for clarification needed
      if clarification_needed:
        response_message = clarification_needed
        metadata = {"goals": []}
        log(f"**Goal Creation Skipped**: Clarification needed - {clarification_needed}")
        return response_message, metadata
      
      # Validate required fields
      if not granularity or granularity not in ["weekly", "monthly", "yearly"]:
        response_message = f"Goal {i+1}: Please specify the granularity (weekly, monthly, or yearly) for this goal."
        metadata = {"goals": []}
        log(f"**Goal Creation Failed**: Missing granularity for goal {i+1}")
        return response_message, metadata
      
      if amount is None or amount < 0:
        response_message = f"Goal {i+1}: Please specify a valid target amount (greater than or equal to 0) for this goal."
        metadata = {"goals": []}
        log(f"**Goal Creation Failed**: Invalid amount for goal {i+1}")
        return response_message, metadata
      
      # Validate goal type
      valid_types = ["category", "credit_X_amount", "save_X_amount", "credit_0", "save_0"]
      if goal_type not in valid_types:
        response_message = f"Goal {i+1}: Invalid goal type '{goal_type}'. Must be one of: {', '.join(valid_types)}."
        metadata = {"goals": []}
        log(f"**Goal Creation Failed**: Invalid type for goal {i+1}")
        return response_message, metadata
      
      # Validate type-specific requirements
      if goal_type == "category":
        if not match_category:
          response_message = f"Goal {i+1}: Could you clarify the category for {category if category else 'this goal'}? Please specify a valid category name."
          metadata = {"goals": []}
          log(f"**Goal Creation Failed**: Missing match_category for category type goal {i+1}")
          return response_message, metadata
      elif goal_type in ["credit_X_amount", "save_X_amount", "credit_0", "save_0"]:
        if account_id is None:
          response_message = f"Goal {i+1}: Please specify an account_id for {goal_type} type goals."
          metadata = {"goals": []}
          log(f"**Goal Creation Failed**: Missing account_id for {goal_type} type goal {i+1}")
          return response_message, metadata
        
        # Validate percent for credit_X_amount/save_X_amount
        if goal_type in ["credit_X_amount", "save_X_amount"] and percent is not None:
          if percent < 0 or percent > 100:
            response_message = f"Goal {i+1}: Percent must be between 0 and 100."
            metadata = {"goals": []}
            log(f"**Goal Creation Failed**: Invalid percent for goal {i+1}")
            return response_message, metadata
      
      # Set defaults
      if not end_date:
        end_date = "2099-12-31"
      
      # Create goal object
      goal = {
        "title": title,
        "amount": float(amount),
        "start_date": start_date,
        "end_date": end_date,
        "category": category,
        "matched_category": match_category,
        "granularity": granularity,
        "type": goal_type,
        "match_caveats": match_caveats,
        "clarification_needed": clarification_needed,
        "description": description,
        "account_id": int(account_id) if account_id is not None else None,
        "percent": float(percent) if percent is not None else None
      }
      
      created_goals.append(goal)
      
      # Collect caveats
      if match_caveats:
        caveats.append(match_caveats)
    
    # Build success message
    response_parts = []
    
    # Add caveats first if they exist
    if caveats:
      response_parts.extend(caveats)
    
    # Success message
    response_parts.append(f"Successfully created {len(created_goals)} budget(s) or goal(s).")
    
    # Add goal descriptions from the description field
    descriptions = []
    for goal in created_goals:
      if goal.get("description"):
        descriptions.append(goal["description"])
        response_parts.append(goal["description"])
    
    # Join with newlines
    response_message = "\n\n".join(response_parts)
    
    metadata = {"goals": created_goals}
    
    log(f"**Returning** {len(created_goals)} goal(s) created")
    if descriptions:
      log(f"**Goal Descriptions**:\n  - `{'`\n  - `'.join(descriptions)}`")
    log(f"**Metadata**:\n```json\n{json.dumps(metadata, indent=2)}\n```")
    
    return response_message, metadata
    
  except Exception as e:
    log(f"**Error creating goal**: {str(e)}")
    response_message = f"I encountered an error while creating your goal: {str(e)}. Please check your parameters and try again."
    metadata = {"goals": []}
    return response_message, metadata

