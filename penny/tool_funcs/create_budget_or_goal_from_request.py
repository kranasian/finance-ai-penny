"""
Parse creation_request (and optional input_info) and dispatch to create_category_spending_limit or create_savings_goal.
Used by create_budget_or_goal_or_reminder flow; goal_agent_optimizer calls create_category_spending_limit and create_savings_goal directly.
"""
import re
from typing import Tuple, Optional

_CATEGORY_MAP = {
    "grocery": "meals_groceries", "groceries": "meals_groceries", "food": "meals_groceries",
    "dining": "meals_dining_out", "dining out": "meals_dining_out", "shopping": "shopping_clothing",
    "gas": "transportation_car", "transportation": "transportation_car",
    "entertainment": "leisure_entertainment", "streaming": "leisure_entertainment",
    "utilities": "shelter_utilities", "rent": "shelter_home", "insurance": "bills_insurance",
}
_DEFAULT_END = "2099-12-31"


def _is_savings(creation_request: str) -> bool:
    text = (creation_request or "").lower()
    phrases = (
        "save", "saving", "put away", "reach ", "by end of", "by 20",
        "emergency fund", "vacation", "down payment", "for a car", "every month",
        "every week", "per month", "per week", "total of", "total amount",
    )
    return any(p in text for p in phrases)


def _parse_category(creation_request: str, input_info: Optional[str] = None) -> Optional[dict]:
    text = (creation_request or "") + " " + (input_info or "")
    text_lower = text.lower()
    m = re.search(r"\$?\s*([\d,]+(?:\.\d+)?)", text)
    amount = float(m.group(1).replace(",", "")) if m else None
    if amount is None:
        return None
    granularity = "monthly"
    if "week" in text_lower or "weekly" in text_lower:
        granularity = "weekly"
    elif "year" in text_lower or "yearly" in text_lower:
        granularity = "yearly"
    category = "meals_groceries"
    for kw, slug in _CATEGORY_MAP.items():
        if kw in text_lower:
            category = slug
            break
    title = (creation_request or "Budget").strip()[:80]
    return {"category": category, "granularity": granularity, "start_date": "", "end_date": "", "amount": amount, "title": title}


def _parse_savings(creation_request: str, input_info: Optional[str] = None) -> Optional[dict]:
    text = (creation_request or "") + " " + (input_info or "")
    text_lower = text.lower()
    m = re.search(r"\$?\s*([\d,]+(?:\.\d+)?)", text)
    amount = float(m.group(1).replace(",", "")) if m else None
    if amount is None:
        return None
    goal_type = "save_X_amount"
    if any(x in text_lower for x in ("by ", "total of", "total amount", "by end", "by december", "by next year")):
        goal_type = "save_0"
    if "per month" in text_lower or "monthly" in text_lower or "/month" in text_lower or "every month" in text_lower:
        goal_type = "save_X_amount"
    if "per week" in text_lower or "weekly" in text_lower or "/week" in text_lower or "every week" in text_lower:
        goal_type = "save_X_amount"
    granularity = "monthly"
    if "week" in text_lower or "weekly" in text_lower:
        granularity = "weekly"
    elif "year" in text_lower and "yearly" in text_lower:
        granularity = "yearly"
    end_date = _DEFAULT_END
    ymd = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if ymd:
        end_date = ymd.group(0)
    title = (creation_request or "Savings goal").strip()[:80]
    return {"amount": amount, "end_date": end_date, "title": title, "goal_type": goal_type, "granularity": granularity, "start_date": ""}


def create_budget_or_goal_from_request(
    creation_request: str,
    input_info: Optional[str] = None,
) -> Tuple[bool, str]:
    """Parse creation_request and call create_category_spending_limit or create_savings_goal."""
    if _is_savings(creation_request):
        from penny.tool_funcs.create_savings_goal import create_savings_goal
        parsed = _parse_savings(creation_request, input_info)
        if not parsed:
            return False, "Could not parse a target amount. Please specify an amount (e.g. $200 per month or $5000 total)."
        return create_savings_goal(
            amount=parsed["amount"],
            end_date=parsed["end_date"],
            title=parsed["title"],
            goal_type=parsed["goal_type"],
            granularity=parsed["granularity"],
            start_date=parsed["start_date"],
        )
    from penny.tool_funcs.create_category_spending_limit import create_category_spending_limit
    parsed = _parse_category(creation_request, input_info)
    if not parsed:
        return False, "Could not parse a target amount. Please specify an amount (e.g. $500 per month)."
    return create_category_spending_limit(
        category=parsed["category"],
        granularity=parsed["granularity"],
        start_date=parsed["start_date"],
        end_date=parsed["end_date"],
        amount=parsed["amount"],
        title=parsed["title"],
    )
