"""Planner skill: Hey Penny app usage Q&A (delegates to PennyAppUsageInfoOptimizer)."""
import os
import sys
from typing import Optional

from dotenv import load_dotenv

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_active_experiments = os.path.join(_repo_root, "active_experiments")
if _active_experiments not in sys.path:
    sys.path.insert(0, _active_experiments)

from penny_app_usage_info_optimizer import PennyAppUsageInfoOptimizer

load_dotenv()

_app_usage_info_optimizer: Optional[PennyAppUsageInfoOptimizer] = None


def app_usage_info(usage_request: str, *args, **kwargs) -> tuple[bool, str]:
    """
    Answer Hey Penny app usage questions: navigation (tabs, sections, buttons), category
    definitions and defaults, how features work (e.g. Split It Up, goals), and product limits
    (e.g. no manual transactions, no custom categories). Uses real in-app labels only—no user
    financial data. Delegates to PennyAppUsageInfoOptimizer.generate_response, which only
    accepts the usage question string; ``*args`` / ``kwargs`` (e.g. ``input_info=``) are ignored.
    """
    global _app_usage_info_optimizer
    if _app_usage_info_optimizer is None:
        _app_usage_info_optimizer = PennyAppUsageInfoOptimizer()
    try:
        result = _app_usage_info_optimizer.generate_response(usage_request.strip())
        return True, result["reply"]
    except Exception as e:
        return False, str(e)
