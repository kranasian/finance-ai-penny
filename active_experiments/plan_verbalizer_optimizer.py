"""
Optimizer runner for **P:PlanVerbalizer** (Gemini prompt tuning).

Input is verbalized ``# Financial Need`` (with ``## Need Details``), the matching ``# Financial Strategy`` subsection
(recommended or alternative only) and a ``### Spending Schedule`` block with compact spending
bullets for one scenario (recommended by default, or
``--scenario-id``), plus ``### Projection`` when simulation months are known.

Objective: verbalize that one plan as ``plan_title``, ``plan_badge``, ``plan_summary``,
``table_title``, ``spending_budget_table``, ``chart_title``, ``chart_type``, ``chart_info_months``,
and ``chart_target_balance``.

Run from ``finance-ai-penny`` repo root (``finance-ai-penny/.venv`` or ``finance-ai-llm-server/llm``):

  python3 active_experiments/plan_verbalizer_optimizer.py --test 0
  python3 active_experiments/plan_verbalizer_optimizer.py --test all
  python3 active_experiments/plan_verbalizer_optimizer.py --simulate-agent-outcome-id 1148 --print-input-only
  python3 active_experiments/plan_verbalizer_optimizer.py --simulate-agent-outcome-id 1148 --scenario-id gradual_paydown_savings
  python3 active_experiments/plan_verbalizer_optimizer.py --user-id 3

DB-backed runs read ``SLAVE_DB`` from ``finance-ai-llm-server/config.ini``. Requires ``psycopg2-binary``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import ClientError
except Exception:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]
    ClientError = Exception  # type: ignore[misc, assignment]

from active_experiments.verbalizer_optimizer_db import (
    _bundled_input_from_test_case,
    _fetch_ai_agent_outcome_row,
    _fetch_user_goal_plan,
    _finalize_goal_plan_for_bundle,
    _goal_plan_active_scenario,
    _normalize_goal_plan_list,
    _resolve_ideal_response,
    _scenario_ids_from_simulate_strategy,
    load_simulate_agent_outcome_markdown,
    resolve_simulate_agent_outcome_id,
)
from active_experiments.need_verbalizer_optimizer import (
    NeedVerbalizerOptimizer,
    _category_display_label,
    _format_goal_plan_narrative,
    _ordered_category_amounts,
    _parse_model_json_object,
    _positive_category_amounts,
    build_need_verbalizer_input_bundle,
    ensure_blank_line_after_plan_headings,
    format_financial_need_block,
    trim_simulate_outcome_for_plan_bundle,
)

if load_dotenv is not None:
    load_dotenv()

GEMINI_FLASH_LITE = "gemini-flash-lite-latest"
PLAN_VERBALIZER_THINKING_BUDGET = 128
PLAN_VERBALIZER_MAX_OUTPUT_TOKENS = 2048

_CHART_TYPES = (
    "projected_total_credit_balance",
    "projected_total_depository_balance",
    "projected_combined_net_balance",
)
_MIN_CHART_INFO_MONTHS = 3

SYSTEM_PROMPT = """You are Penny — a sharp, witty money coach who explains one financial plan in clear, concrete detail.

Use ``# Financial Need``, ``## Need Details``, matching plan prose in ``# Financial Strategy``, ``### Current Spending``, and caps under ``### Spending Schedule``.

- ``plan_title``: one-line headline for this plan (max **5 words** and **40 characters**; punchy, no jargon).
- ``plan_badge``: adjective for how hard or how unique this plan is (e.g., "Disciplined", "Balanced", "Austerity", "Rigorous", "Empathetic", "Steady"). Use Title Case.
- ``plan_summary``: short description of what this plan does (max **20 words**, max **3 lines**, and **155 characters**); ground every **$** and date in the input.
- ``table_title``: short title for the spending comparison table (max **6 words** and **40 characters**).
- ``spending_budget_table``: one markdown table with columns ``Spending``, ``Current``, ``Budget`` (separate rows using standard markdown newlines `\\n`, do NOT use `<br>` to separate rows).
  - One row per category in ``### Spending Schedule`` (use display names from the schedule, capitalized for a premium look).
  - ``Current`` from ``### Current Spending`` for that category (ground every **$**).
  - ``Budget`` may use multiple lines in a cell (separate with ``<br>``) when the schedule has multiple phases:
    - first phase: ``$amount (n% cut)`` vs Current (use ``0% cut`` or ``n% up`` if not a cut)
    - later phases: ``$amount N months later`` (months from plan start to that phase)
  - Final row: ``Total`` with summed Current and Budget totals (Budget totals also multi-line when phased).
- ``chart_title``: short title for the plan chart (max **6 words** and **40 characters**); describe what the selected ``chart_type`` shows.
- ``chart_type``: choose the projected chart that best shows the plan's primary outcome over time. Must be exactly one of:
  * ``projected_total_credit_balance`` — when the plan goal is paying credit down (to ``$0`` or a stated floor)
  * ``projected_total_depository_balance`` — when the plan goal is building savings, an emergency fund, or holding a cash buffer
  * ``projected_combined_net_balance`` — when net position (depository minus credit) is the main story
- ``chart_info_months``: integer months of projected data to display (minimum 3).
- ``chart_target_balance``: integer goal line (``0`` for full credit payoff, the payoff floor for partial paydown, the savings target for depository charts).

The budget table already shows category caps over time. The chart should show the primary outcome the plan is driving toward.

Do not invent Current amounts — only use ``### Current Spending``. Output compact JSON only — no extra fields.
"""


def _build_output_schema() -> "types.Schema":
    if types is None:  # pragma: no cover
        raise RuntimeError("Install `google-genai` for this optimizer.")
    return types.Schema(
        type=types.Type.OBJECT,
        required=[
            "plan_title",
            "plan_badge",
            "plan_summary",
            "table_title",
            "spending_budget_table",
            "chart_title",
            "chart_type",
            "chart_info_months",
            "chart_target_balance",
        ],
        properties={
            "plan_title": types.Schema(
                type=types.Type.STRING,
                description="One-line plan headline (max 5 words and 40 characters).",
            ),
            "plan_badge": types.Schema(
                type=types.Type.STRING,
                description="Adjective for how hard or unique the plan is.",
            ),
            "plan_summary": types.Schema(
                type=types.Type.STRING,
                description="Short plan description (max 20 words, max 3 lines, and 155 characters).",
            ),
            "table_title": types.Schema(
                type=types.Type.STRING,
                description="Title for the spending comparison table (max 6 words and 40 characters).",
            ),
            "spending_budget_table": types.Schema(
                type=types.Type.STRING,
                description="Markdown table: Spending, Current, Budget; Total row; phased Budget cells use <br>.",
            ),
            "chart_title": types.Schema(
                type=types.Type.STRING,
                description="Title for the plan chart (max 6 words and 40 characters); aligned with chart_type.",
            ),
            "chart_type": types.Schema(
                type=types.Type.STRING,
                enum=list(_CHART_TYPES),
                description="Projected chart showing the plan's primary outcome.",
            ),
            "chart_info_months": types.Schema(
                type=types.Type.INTEGER,
                description="Months of projected chart data to display (minimum 3).",
            ),
            "chart_target_balance": types.Schema(
                type=types.Type.INTEGER,
                description=(
                    "Goal line: 0 for full credit payoff, payoff floor for partial paydown, "
                    "or savings target for depository charts."
                ),
            ),
        },
    )


SPENDING_SCHEDULE_H3 = "### Spending Schedule"
PROJECTION_H3 = "### Projection"
CURRENT_SPENDING_H3 = "### Current Spending"
FINANCIAL_NEED_H1 = "# Financial Need"
_FINANCIAL_STRATEGY_H1 = "# Financial Strategy"


def _format_projection_block(
    projected_months: Any,
    *,
    stop_reason: str = "goal achieved",
) -> str:
    try:
        months = int(projected_months)
    except (TypeError, ValueError):
        return ""
    if months < _MIN_CHART_INFO_MONTHS:
        return ""
    reason = str(stop_reason or "goal achieved").strip() or "goal achieved"
    return f"{PROJECTION_H3}\n\n- Projection: {months} mo, stop {reason}.\n"


def _format_current_spending_block(current_spending: Any) -> str:
    if not isinstance(current_spending, dict):
        return ""
    cats = _positive_category_amounts(current_spending)
    if not cats:
        return ""
    lines = [CURRENT_SPENDING_H3, ""]
    for slug, amount in _ordered_category_amounts(cats):
        lines.append(f"- {_category_display_label(slug)} ${amount}")
    return "\n".join(lines) + "\n"


def _validate_plan_response(parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        raise ValueError("Response must be a JSON object")
    plan_title = parsed.get("plan_title")
    if not isinstance(plan_title, str) or not plan_title.strip():
        raise ValueError("plan_title must be a non-empty string")
    plan_badge = parsed.get("plan_badge")
    if not isinstance(plan_badge, str) or not plan_badge.strip():
        raise ValueError("plan_badge must be a non-empty string")
    plan_summary = parsed.get("plan_summary")
    if not isinstance(plan_summary, str) or not plan_summary.strip():
        raise ValueError("plan_summary must be a non-empty string")
    table_title = parsed.get("table_title")
    if not isinstance(table_title, str) or not table_title.strip():
        raise ValueError("table_title must be a non-empty string")
    spending_budget_table = parsed.get("spending_budget_table")
    if not isinstance(spending_budget_table, str) or not spending_budget_table.strip():
        raise ValueError("spending_budget_table must be a non-empty string")
    chart_title = parsed.get("chart_title")
    if not isinstance(chart_title, str) or not chart_title.strip():
        raise ValueError("chart_title must be a non-empty string")
    chart_type = parsed.get("chart_type")
    if not isinstance(chart_type, str) or chart_type not in _CHART_TYPES:
        raise ValueError(f"chart_type must be one of: {', '.join(_CHART_TYPES)}")
    chart_info_months = parsed.get("chart_info_months")
    try:
        chart_info_months = int(chart_info_months)
    except (TypeError, ValueError) as exc:
        raise ValueError("chart_info_months must be an integer") from exc
    if chart_info_months < _MIN_CHART_INFO_MONTHS:
        raise ValueError(f"chart_info_months must be >= {_MIN_CHART_INFO_MONTHS}")
    chart_target_balance = parsed.get("chart_target_balance")
    try:
        chart_target_balance = int(chart_target_balance)
    except (TypeError, ValueError) as exc:
        raise ValueError("chart_target_balance must be an integer") from exc
    if chart_target_balance < 0:
        raise ValueError("chart_target_balance must be >= 0")
    return {
        "plan_title": plan_title.strip(),
        "plan_badge": plan_badge.strip(),
        "plan_summary": plan_summary.strip(),
        "table_title": table_title.strip(),
        "spending_budget_table": spending_budget_table.strip(),
        "chart_title": chart_title.strip(),
        "chart_type": chart_type,
        "chart_info_months": chart_info_months,
        "chart_target_balance": chart_target_balance,
    }


TEST_CASES: list[dict[str, Any]] = [
    {
        "name": "debt_paydown_recommended",
        "batch": 1,
        "input": """
# Financial Need

$312 in interest every 90 days on your $8,400 balance while spending tracks income.

## Need Details

Interest tool: **$312** on Venture in 90 days. Next due **2026-04-18** per payment schedule.

# Financial Strategy

## Recommended plan: gradual_paydown_savings
* Goal: pay Venture to **$0**. Phased dining and leisure trims keep month-1 cuts modest, then deepen after month 3; route **$200**/mo to savings only after the card hits **$0**.

## Alternative plan: steady_cut
* Goal: pay Venture to **$0**. Flat **$700** food and **$350** leisure from month 1 clears the card about two months sooner but leaves thinner checking buffers in the first quarter.

### Current Spending
- food $1000
- leisure $500

### Spending Schedule
- 04/26-06/26: Cap food $850, leisure $450 monthly
- 07/26-03/28: Cap food $700, leisure $350 monthly

### Projection
- Projection: 12 mo, stop goal achieved.
""",
        "ideal_response": {
            "plan_title": "Gradual paydown",
            "plan_badge": "Gentle",
            "plan_summary": "Pay Venture to $0 with phased cuts, then save $200/mo.",
            "table_title": "Spending vs plan budget",
            "spending_budget_table": (
                "| Spending | Current | Budget |\n"
                "| --- | --- | --- |\n"
                "| food | $1,000 | $850 (15% cut)<br>$700 3 months later |\n"
                "| leisure | $500 | $450 (10% cut)<br>$350 3 months later |\n"
                "| Total | $1,500 | $1,300<br>$1,050 |"
            ),
            "chart_title": "Credit balance paydown",
            "chart_type": "projected_total_credit_balance",
            "chart_info_months": 12,
            "chart_target_balance": 0,
        },
    },
    {
        "name": "debt_paydown_alternative",
        "batch": 1,
        "scenario_id": "steady_cut",
        "input": """
# Financial Need

$312 in interest every 90 days on your $8,400 balance while spending tracks income.

## Need Details

Interest tool: **$312** on Venture in 90 days. Next due **2026-04-18** per payment schedule.

# Financial Strategy

## Recommended plan: gradual_paydown_savings
* Goal: pay Venture to **$0**. Phased dining and leisure trims keep month-1 cuts modest, then deepen after month 3; route **$200**/mo to savings only after the card hits **$0**.

## Alternative plan: steady_cut
* Goal: pay Venture to **$0**. Flat **$700** food and **$350** leisure from month 1 clears the card about two months sooner but leaves thinner checking buffers in the first quarter.

### Current Spending
- food $1000
- leisure $500

### Spending Schedule
- 04/26-03/28: Cap food $700, leisure $350 monthly

### Projection
- Projection: 8 mo, stop goal achieved.
""",
        "ideal_response": {
            "plan_title": "Steady cut",
            "plan_badge": "Hard",
            "plan_summary": "Pay Venture to $0 with food at $700/mo and leisure at $350/mo from month one.",
            "table_title": "Spending vs plan budget",
            "spending_budget_table": (
                "| Spending | Current | Budget |\n"
                "| --- | --- | --- |\n"
                "| food | $1,000 | $700 (30% cut) |\n"
                "| leisure | $500 | $350 (30% cut) |\n"
                "| Total | $1,500 | $1,050 |"
            ),
            "chart_title": "Credit balance paydown",
            "chart_type": "projected_total_credit_balance",
            "chart_info_months": 8,
            "chart_target_balance": 0,
        },
    },
    {
        "name": "cash_flow_recommended",
        "batch": 1,
        "input": """
# Financial Need

Credit card balance sits at **$4,200** and keeps climbing with minimum-style payments.

## Need Details

Card balance **$4,200**. Interest about **$90**/mo at the current APR. Forecast committed outflows **$3,600**/mo vs income **$4,000**/mo.

# Financial Strategy

## Recommended plan: protect_fixed_cut_flex
* Goal: pay the card down to **$3,000**. Trim **$200**/mo from food and shopping months 1–3, then reassess.

## Alternative plan: aggressive_flex_cut
* Goal: pay the card down to **$1,500**. Cut food to **$450** and shopping to **$150** immediately — deeper paydown, but checking may dip below **$500** early on.

### Current Spending
- food $650
- shopping $250

### Spending Schedule
- 04/26-03/28: Cap food $520, shopping $180 monthly

### Projection
- Projection: 6 mo, stop goal achieved.
""",
        "ideal_response": {
            "plan_title": "Pay to three thousand",
            "plan_badge": "Moderate",
            "plan_summary": "Pay the card down to $3,000 while trimming food and shopping.",
            "table_title": "Spending vs plan budget",
            "spending_budget_table": (
                "| Spending | Current | Budget |\n"
                "| --- | --- | --- |\n"
                "| food | $650 | $520 (20% cut) |\n"
                "| shopping | $250 | $180 (28% cut) |\n"
                "| Total | $900 | $700 |"
            ),
            "chart_title": "Credit balance paydown",
            "chart_type": "projected_total_credit_balance",
            "chart_info_months": 6,
            "chart_target_balance": 3000,
        },
    },
    {
        "name": "cash_flow_alternative",
        "batch": 1,
        "scenario_id": "aggressive_flex_cut",
        "input": """
# Financial Need

Credit card balance sits at **$4,200** and keeps climbing with minimum-style payments.

## Need Details

Card balance **$4,200**. Interest about **$90**/mo at the current APR. Forecast committed outflows **$3,600**/mo vs income **$4,000**/mo.

# Financial Strategy

## Recommended plan: protect_fixed_cut_flex
* Goal: pay the card down to **$3,000**. Trim **$200**/mo from food and shopping months 1–3, then reassess.

## Alternative plan: aggressive_flex_cut
* Goal: pay the card down to **$1,500**. Cut food to **$450** and shopping to **$150** immediately — deeper paydown, but checking may dip below **$500** early on.

### Current Spending
- food $650
- shopping $250

### Spending Schedule
- 04/26-03/28: Cap food $450, shopping $150 monthly

### Projection
- Projection: 5 mo, stop goal achieved.
""",
        "ideal_response": {
            "plan_title": "Aggressive flex cut",
            "plan_badge": "Strict",
            "plan_summary": "Pay the card down to $1,500 with food at $450/mo and shopping at $150/mo.",
            "table_title": "Spending vs plan budget",
            "spending_budget_table": (
                "| Spending | Current | Budget |\n"
                "| --- | --- | --- |\n"
                "| food | $650 | $450 (31% cut) |\n"
                "| shopping | $250 | $150 (40% cut) |\n"
                "| Total | $900 | $600 |"
            ),
            "chart_title": "Credit balance paydown",
            "chart_type": "projected_total_credit_balance",
            "chart_info_months": 5,
            "chart_target_balance": 1500,
        },
    },
    {
        "name": "slow_debt_recommended",
        "batch": 2,
        "input": """
# Financial Need

Balance rose $300 in three months despite $115/mo payments on $4,800 owed.

## Need Details

Balance up **$300** over three months despite **$115**/mo payments. APR tool: **~21.8%** on Platinum.

# Financial Strategy

## Recommended plan: balanced_trim
* Goal: pay Platinum to **$0** by **Dec 2026**. Trim food to **$520** and leisure to **$300** from month 1; saves about **$420** interest vs status quo.

## Alternative plan: leisure_first
* Goal: pay Platinum down to **$2,000**. Protect leisure at **$380** but cut food harder to **$450** — reaches the **$2,000** floor with more dining sacrifice and less social spend risk.

### Current Spending
- food $650
- leisure $400

### Spending Schedule
- 04/26-03/28: Cap food $520, leisure $300 monthly

### Projection
- Projection: 12 mo, stop goal achieved.
""",
        "ideal_response": {
            "plan_title": "Balanced trim",
            "plan_badge": "Moderate",
            "plan_summary": "Pay Platinum to $0 by Dec 2026 with food at $520/mo and leisure at $300/mo.",
            "table_title": "Spending vs plan budget",
            "spending_budget_table": (
                "| Spending | Current | Budget |\n"
                "| --- | --- | --- |\n"
                "| food | $650 | $520 (20% cut) |\n"
                "| leisure | $400 | $300 (25% cut) |\n"
                "| Total | $1,050 | $820 |"
            ),
            "chart_title": "Credit balance paydown",
            "chart_type": "projected_total_credit_balance",
            "chart_info_months": 12,
            "chart_target_balance": 0,
        },
    },
    {
        "name": "slow_debt_alternative",
        "batch": 2,
        "scenario_id": "leisure_first",
        "input": """
# Financial Need

Balance rose $300 in three months despite $115/mo payments on $4,800 owed.

## Need Details

Balance up **$300** over three months despite **$115**/mo payments. APR tool: **~21.8%** on Platinum.

# Financial Strategy

## Recommended plan: balanced_trim
* Goal: pay Platinum to **$0** by **Dec 2026**. Trim food to **$520** and leisure to **$300** from month 1; saves about **$420** interest vs status quo.

## Alternative plan: leisure_first
* Goal: pay Platinum down to **$2,000**. Protect leisure at **$380** but cut food harder to **$450** — reaches the **$2,000** floor with more dining sacrifice and less social spend risk.

### Current Spending
- food $650
- leisure $400

### Spending Schedule
- 04/26-03/28: Cap food $450, leisure $380 monthly

### Projection
- Projection: 9 mo, stop goal achieved.
""",
        "ideal_response": {
            "plan_title": "Leisure-first",
            "plan_badge": "Unique",
            "plan_summary": "Pay Platinum down to $2,000 with leisure at $380/mo and food at $450/mo.",
            "table_title": "Spending vs plan budget",
            "spending_budget_table": (
                "| Spending | Current | Budget |\n"
                "| --- | --- | --- |\n"
                "| food | $650 | $450 (31% cut) |\n"
                "| leisure | $400 | $380 (5% cut) |\n"
                "| Total | $1,050 | $830 |"
            ),
            "chart_title": "Credit balance paydown",
            "chart_type": "projected_total_credit_balance",
            "chart_info_months": 9,
            "chart_target_balance": 2000,
        },
    },
    {
        "name": "spending_drift_recommended",
        "batch": 3,
        "simulate_agent_outcome_id": 1252,
        "input": """
# Financial Need

Fixed overhead consumes ~75% of $6,369 take-home, forcing credit-card bridging.

## Need Details

Shelter $2,850 plus school/daycare $500 and utilities ~$350 consume most take-home. Core fixed costs leave almost no margin for emergencies or savings.

# Financial Strategy

## Recommended plan: empathetic_staged_adjustment
* Goal: pay high-interest revolving credit to **$0**. Step down discretionary spend over six months so the cut sticks, clearing the balance in the same quarter as faster options without a shock cut.

## Alternative plan: rapid_debt_sprint
* Goal: pay high-interest revolving credit to **$0**. Kill the 24.99% APR cycle as fast as possible with harder immediate caps, even if the short-term cut feels restrictive.

### Current Spending
- food $1400
- leisure $400
- shopping $80
- health $80
- education $450
- uncategorized $350

### Spending Schedule
- 08/26-10/26: Cap food $1200, leisure $300, shopping $50, health $80, education $450, uncategorized $300 monthly
- 11/26-01/27: Cap food $750, leisure $200, shopping $50, health $80, education $450, uncategorized $300 monthly
- 02/27-: Cap food $500, leisure $100, shopping $50, health $80, education $450, uncategorized $300 monthly

### Projection
- Projection: 9 mo, stop goal achieved.
""",
        "ideal_response": {
            "plan_title": "Staged drift reset",
            "plan_badge": "Gentle",
            "plan_summary": "Pay high-interest credit to $0 by stepping food and leisure down over three phases.",
            "table_title": "Spending vs plan budget",
            "spending_budget_table": (
                "| Spending | Current | Budget |\n"
                "| --- | --- | --- |\n"
                "| food | $1,400 | $1,200 (14% cut)<br>$750 3 months later<br>$500 6 months later |\n"
                "| leisure | $400 | $300 (25% cut)<br>$200 3 months later<br>$100 6 months later |\n"
                "| shopping | $80 | $50 (38% cut)<br>$50 3 months later<br>$50 6 months later |\n"
                "| health | $80 | $80 (0% cut)<br>$80 3 months later<br>$80 6 months later |\n"
                "| education | $450 | $450 (0% cut)<br>$450 3 months later<br>$450 6 months later |\n"
                "| uncategorized | $350 | $300 (14% cut)<br>$300 3 months later<br>$300 6 months later |\n"
                "| Total | $2,760 | $2,380<br>$1,830<br>$1,480 |"
            ),
            "chart_title": "Credit balance paydown",
            "chart_type": "projected_total_credit_balance",
            "chart_info_months": 9,
            "chart_target_balance": 0,
        },
    },
    {
        "name": "emergency_savings_target_recommended",
        "batch": 4,
        "input": """
# Financial Need

You want an emergency buffer of **$6,000**, but your current savings is only **$1,000**, and spending is close to income.

## Need Details

Savings gap is **$5,000** to reach **$6,000**. Committed spend leaves little slack, so steady discretionary cuts are needed.

# Financial Strategy

## Recommended plan: save_to_emergency
* Goal: save **$6,000**. Trim food and leisure for the first stretch, then hold steady to finish the gap.

## Alternative plan: faster_saving
* Goal: save **$4,000** sooner with harder cuts, but less breathing room if spending drifts.

### Current Spending
- food $650
- leisure $400
- shopping $80

### Spending Schedule
- 04/26-03/28: Cap food $520, leisure $300, shopping $50 monthly

### Projection
- Projection: 12 mo, stop goal achieved.
""",
        "ideal_response": {
            "plan_title": "Emergency fund target",
            "plan_badge": "Focused",
            "plan_summary": "Save $6,000 by keeping food at $520/mo and leisure at $300/mo, then holding to finish the gap.",
            "table_title": "Spending vs plan budget",
            "spending_budget_table": (
                "| Spending | Current | Budget |\n"
                "| --- | --- | --- |\n"
                "| food | $650 | $520 (20% cut) |\n"
                "| leisure | $400 | $300 (25% cut) |\n"
                "| shopping | $80 | $50 (38% cut) |\n"
                "| Total | $1,130 | $870 |"
            ),
            "chart_title": "Savings balance growth",
            "chart_type": "projected_total_depository_balance",
            "chart_info_months": 12,
            "chart_target_balance": 6000,
        },
    },
]





def _strip_financial_strategy_plan_section(simulate_md: str, *, section: str) -> str:
    if section not in ("recommended", "alternative"):
        raise ValueError(f"section must be 'recommended' or 'alternative', got {section!r}")
    header = "Recommended plan" if section == "recommended" else "Alternative plan"
    pattern = (
        rf"(?ms)^[ \t]*## {re.escape(header)}:.*?(?=^[ \t]*## |\Z)"
    )
    text = re.sub(pattern, "", simulate_md or "")
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    return text + "\n"


def filter_financial_strategy_for_scenario(
    simulate_md: str,
    scenario_id: str,
    *,
    is_active: bool | None = None,
) -> str:
    """Keep only the ``# Financial Strategy`` subsection for the intended scenario."""
    sid = str(scenario_id or "").strip()
    if not sid:
        return simulate_md

    rec_id, alt_id = _scenario_ids_from_simulate_strategy(simulate_md)
    if rec_id and sid == rec_id:
        return _strip_financial_strategy_plan_section(simulate_md, section="alternative")
    if alt_id and sid == alt_id:
        return _strip_financial_strategy_plan_section(simulate_md, section="recommended")
    if is_active is True:
        return _strip_financial_strategy_plan_section(simulate_md, section="alternative")
    if is_active is False:
        return _strip_financial_strategy_plan_section(simulate_md, section="recommended")
    return simulate_md


def _apply_strategy_filter_to_plan_bundle(
    bundle_md: str,
    scenario_id: str,
    *,
    is_active: bool | None = None,
) -> str:
    marker = SPENDING_SCHEDULE_H3
    if marker not in bundle_md:
        return filter_financial_strategy_for_scenario(
            bundle_md,
            scenario_id,
            is_active=is_active,
        )
    before, after = bundle_md.split(marker, 1)
    filtered = filter_financial_strategy_for_scenario(
        before,
        scenario_id,
        is_active=is_active,
    )
    return filtered.rstrip() + "\n\n" + marker + after


def _select_goal_plan_scenario(goal_plan: Any, *, scenario_id: str | None = None) -> dict[str, Any]:
    entries = _normalize_goal_plan_list(goal_plan)
    if not entries:
        raise ValueError("goal_plan must include at least one scenario")

    if scenario_id:
        sid = str(scenario_id).strip()
        for entry in entries:
            if str(entry.get("scenario_id") or "").strip() == sid:
                return dict(entry)
        raise ValueError(f"scenario_id not found in goal_plan: {sid}")

    active = _goal_plan_active_scenario(goal_plan)
    if not active:
        raise ValueError("goal_plan has no scenarios")
    return dict(active)


def _prepend_verbalized_need_to_plan_bundle(
    bundle_md: str,
    need_verbalizer_response: dict[str, Any],
) -> str:
    need_block = format_financial_need_block(need_verbalizer_response)
    text = (bundle_md or "").strip()
    if FINANCIAL_NEED_H1 in text:
        strategy_idx = text.find(_FINANCIAL_STRATEGY_H1)
        if strategy_idx < 0:
            return need_block.rstrip() + "\n\n" + text + "\n"
        return need_block.rstrip() + "\n\n" + text[strategy_idx:].strip() + "\n"
    strategy_idx = text.find(_FINANCIAL_STRATEGY_H1)
    if strategy_idx < 0:
        raise ValueError("plan bundle must include # Financial Strategy")
    return need_block.rstrip() + "\n\n" + text[strategy_idx:].strip() + "\n"


def build_plan_verbalizer_input_bundle(
    *,
    need_verbalizer_response: dict[str, Any],
    simulate_outcome_md: str,
    goal_plan_scenario: dict[str, Any],
) -> str:
    """Verbalized need + matching strategy subsection + current spend + ``### Spending Schedule``."""
    need_block = format_financial_need_block(need_verbalizer_response)
    simulate = trim_simulate_outcome_for_plan_bundle(simulate_outcome_md)
    scenario_id = str(goal_plan_scenario.get("scenario_id") or "").strip()
    if scenario_id:
        simulate = filter_financial_strategy_for_scenario(
            simulate,
            scenario_id,
            is_active=goal_plan_scenario.get("is_active"),
        )
    simulate = ensure_blank_line_after_plan_headings(simulate)
    current_block = _format_current_spending_block(goal_plan_scenario.get("current_spending"))
    plan_block = _format_goal_plan_narrative([goal_plan_scenario])
    parts = [need_block.rstrip(), simulate.rstrip()]
    if current_block:
        parts.append(current_block.rstrip())
    if plan_block:
        parts.append(plan_block.rstrip())
    projection_block = _format_projection_block(goal_plan_scenario.get("projected_months"))
    if projection_block:
        parts.append(projection_block.rstrip())
    return "\n\n".join(parts) + "\n"


def build_plan_verbalizer_input(
    *,
    simulate_agent_outcome_id: int | None = None,
    user_id: int | None = None,
    scenario_id: str | None = None,
    need_verbalizer_response: dict[str, Any] | None = None,
    need_optimizer: NeedVerbalizerOptimizer | None = None,
) -> str:
    """Build bundled input from DB outcomes + one ``users.goal_plan`` scenario."""
    sim_id = resolve_simulate_agent_outcome_id(
        user_id=user_id,
        simulate_agent_outcome_id=simulate_agent_outcome_id,
    )
    sim_uid, simulate_md = load_simulate_agent_outcome_markdown(sim_id)
    sim_row = _fetch_ai_agent_outcome_row(sim_id)
    if not sim_row:
        raise ValueError(f"simulate_agent_outcome_id not found: {sim_id}")

    if need_verbalizer_response is None:
        need_input = build_need_verbalizer_input_bundle(simulate_outcome_md=simulate_md)
        optimizer = need_optimizer or NeedVerbalizerOptimizer()
        need_verbalizer_response = optimizer.generate_response(need_input)

    goal_plan = _fetch_user_goal_plan(sim_uid)
    if goal_plan is None:
        raise ValueError(
            f"users.goal_plan is empty for user_id={sim_uid}; "
            "run simulate_financial_strategy with persistence first"
        )
    goal_plan = _finalize_goal_plan_for_bundle(
        goal_plan,
        simulate_md,
        simulate_calls=sim_row.get("calls"),
    )
    scenario = _select_goal_plan_scenario(goal_plan, scenario_id=scenario_id)
    return build_plan_verbalizer_input_bundle(
        need_verbalizer_response=need_verbalizer_response,
        simulate_outcome_md=simulate_md,
        goal_plan_scenario=scenario,
    )


def resolve_plan_test_case_input(
    test_case: dict[str, Any],
    *,
    scenario_id: str | None = None,
) -> str:
    """Return bundled input from a test-case dict."""
    bundle: str | None = None
    for key in ("input", "bundled_input"):
        raw = test_case.get(key)
        if isinstance(raw, str) and raw.strip():
            bundle = raw.strip() + "\n"
            break
    if bundle is None:
        bundled = _bundled_input_from_test_case(test_case)
        if bundled:
            bundle = bundled.strip() + "\n"
    if bundle is None:
        raise ValueError("test case must include bundled input")

    need_response = test_case.get("need_verbalizer_response")
    if isinstance(need_response, dict):
        bundle = _prepend_verbalized_need_to_plan_bundle(bundle, need_response)

    sid = str(scenario_id or test_case.get("scenario_id") or "").strip()
    if sid:
        bundle = _apply_strategy_filter_to_plan_bundle(bundle, sid)
    return bundle


def format_plan_verbalizer_user_message(profile_input: str) -> str:
    body = (profile_input or "").strip()
    if not body:
        raise ValueError("profile_input must be non-empty markdown.")
    if FINANCIAL_NEED_H1 not in body:
        raise ValueError(f"profile_input must include {FINANCIAL_NEED_H1}.")
    if SPENDING_SCHEDULE_H3 not in body:
        raise ValueError(f"profile_input must include {SPENDING_SCHEDULE_H3}.")
    return body + "\n"


def _parse_plan_json_response(text: str) -> dict[str, Any]:
    try:
        return _parse_model_json_object(text)
    except (json.JSONDecodeError, ValueError):
        raw = (text or "").strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            return _parse_model_json_object(raw[start:end + 1])
        raise


def _collect_model_response(response: Any) -> tuple[str, str, Any]:
    output_text = ""
    thought_summary = ""
    finish_reason = None
    for cand in getattr(response, "candidates", None) or []:
        reason = getattr(cand, "finish_reason", None)
        if reason is not None:
            finish_reason = reason
        content = getattr(cand, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", None) or []:
            t = getattr(part, "text", None)
            if not isinstance(t, str) or not t:
                continue
            if getattr(part, "thought", False):
                continue
            output_text += t
    return output_text, thought_summary, finish_reason


class PlanVerbalizerOptimizer:
    """Gemini runner for the single-plan verbalizer system prompt."""

    def __init__(
        self,
        model_name: str = GEMINI_FLASH_LITE,
        *,
        thinking_budget: int = PLAN_VERBALIZER_THINKING_BUDGET,
        max_output_tokens: int = PLAN_VERBALIZER_MAX_OUTPUT_TOKENS,
    ):
        if genai is None or types is None:  # pragma: no cover
            raise RuntimeError("Install `google-genai` (and optionally `python-dotenv`) for this optimizer.")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.thinking_budget = thinking_budget
        self.max_output_tokens = max_output_tokens
        self.temperature = 0.35
        self.top_p = 0.95
        self.top_k = 40
        self.safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ]
        self.system_prompt = SYSTEM_PROMPT
        self.output_schema = _build_output_schema()

    def _build_generate_config(self, *, max_output_tokens: int) -> "types.GenerateContentConfig":
        return types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_output_tokens=max_output_tokens,
            safety_settings=self.safety_settings,
            system_instruction=[types.Part.from_text(text=self.system_prompt)],
            thinking_config=types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
                include_thoughts=False,
            ),
            response_mime_type="application/json",
            response_schema=self.output_schema,
        )

    def generate_response(self, profile_input: str) -> dict[str, Any]:
        user_text = format_plan_verbalizer_user_message(profile_input)
        request_text = types.Part.from_text(text=user_text)
        contents = [types.Content(role="user", parts=[request_text])]

        token_limits = [self.max_output_tokens]
        retry_limit = self.max_output_tokens * 2
        if retry_limit > self.max_output_tokens:
            token_limits.append(retry_limit)

        last_error: Exception | None = None
        for attempt_idx, max_tokens in enumerate(token_limits):
            cfg = self._build_generate_config(max_output_tokens=max_tokens)
            output_text = ""
            finish_reason = None
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=cfg,
                )
                output_text, _, finish_reason = _collect_model_response(response)
            except ClientError as e:
                if self.thinking_budget == 0 and "only works in thinking mode" in (str(e) or ""):
                    print(
                        "\n[NOTE] This model requires thinking mode; use default (no --no-thinking) or a different model.",
                        flush=True,
                    )
                    sys.exit(1)
                raise

            if not (output_text or "").strip():
                last_error = ValueError(
                    f"Empty JSON response from model. finish_reason={finish_reason!r}"
                )
                if attempt_idx < len(token_limits) - 1:
                    print(
                        f"\n[RETRY] Empty response at max_output_tokens={max_tokens}; "
                        f"retrying with {token_limits[attempt_idx + 1]}.\n",
                        flush=True,
                    )
                    continue
                raise last_error

            try:
                parsed = _parse_plan_json_response(output_text)
            except (json.JSONDecodeError, ValueError) as exc:
                reason = str(finish_reason or "unknown")
                preview = output_text.strip()[:240].replace("\n", " ")
                last_error = ValueError(
                    f"Invalid JSON response. finish_reason={reason!r}; "
                    f"max_output_tokens={max_tokens}; preview={preview!r}"
                )
                last_error.__cause__ = exc
                if "MAX_TOKENS" in reason and attempt_idx < len(token_limits) - 1:
                    print(
                        f"\n[RETRY] MAX_TOKENS at max_output_tokens={max_tokens}; "
                        f"retrying with {token_limits[attempt_idx + 1]}.\n",
                        flush=True,
                    )
                    continue
                raise last_error from exc

            try:
                validated = _validate_plan_response(parsed)
                return validated
            except ValueError as exc:
                raise ValueError(f"Response failed validation: {exc}") from exc

        if last_error is not None:
            raise last_error
        raise ValueError("Invalid JSON response from model.")


def _run_test(
    profile_input: str,
    optimizer: PlanVerbalizerOptimizer | None = None,
    *,
    ideal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if optimizer is None:
        optimizer = PlanVerbalizerOptimizer()
    wrapped = format_plan_verbalizer_user_message(profile_input)
    print("=" * 80)
    print("LLM INPUT:")
    print("=" * 80)
    print(wrapped)
    result = optimizer.generate_response(profile_input)
    print("=" * 80)
    print("LLM OUTPUT:")
    print("=" * 80)
    print(json.dumps(result, indent=2))
    if ideal is not None:
        print("=" * 80)
        print("IDEAL RESPONSE:")
        print("=" * 80)
        print(json.dumps(ideal, indent=2))
    print("=" * 80 + "\n")
    return result


def get_test_case(test_name_or_index: str | int) -> dict[str, Any] | None:
    if isinstance(test_name_or_index, int):
        if 0 <= test_name_or_index < len(TEST_CASES):
            return TEST_CASES[test_name_or_index]
        return None
    for tc in TEST_CASES:
        if tc["name"] == test_name_or_index:
            return tc
    return None


def run_test(
    test_name_or_index_or_dict: str | int | dict[str, Any],
    optimizer: PlanVerbalizerOptimizer | None = None,
    *,
    scenario_id: str | None = None,
) -> dict[str, Any] | None:
    if optimizer is None:
        optimizer = PlanVerbalizerOptimizer()

    if isinstance(test_name_or_index_or_dict, dict):
        tc = test_name_or_index_or_dict
        name = tc.get("name", "custom_test")
        try:
            payload = resolve_plan_test_case_input(tc, scenario_id=scenario_id)
        except ValueError as exc:
            print(f"Invalid test dict: {exc}")
            return None
        print(f"\n{'=' * 80}\nRunning test: {name}\n{'=' * 80}\n")
        ideal = _resolve_ideal_response(tc)
        return _run_test(payload, optimizer, ideal=ideal)

    tc = get_test_case(test_name_or_index_or_dict)
    if tc is None:
        print(f"Test case '{test_name_or_index_or_dict}' not found.")
        return None
    name = tc["name"]
    sid = scenario_id or tc.get("scenario_id")
    print(f"\n{'=' * 80}\nRunning test: {name}\n{'=' * 80}\n")
    ideal = _resolve_ideal_response(tc)
    return _run_test(resolve_plan_test_case_input(tc, scenario_id=sid), optimizer, ideal=ideal)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P:PlanVerbalizer optimizer tests")
    parser.add_argument("--test", type=str, help='Test name or index (e.g. "0" or "debt_paydown_interest_drag")')
    parser.add_argument("--batch", type=int, help="Run all tests in batch N")
    parser.add_argument(
        "--user-id",
        type=int,
        help="User id; when simulate-agent-outcome-id is omitted, use the latest simulate_financial_strategy outcome.",
    )
    parser.add_argument(
        "--simulate-agent-outcome-id",
        type=int,
        help="simulate_financial_strategy ai_agent_outcomes.agent_outcome_id",
    )
    parser.add_argument(
        "--scenario-id",
        type=str,
        help="Goal-plan scenario_id to verbalize (default: recommended is_active=true)",
    )
    parser.add_argument(
        "--print-input-only",
        action="store_true",
        help="Only print built markdown input (no model call)",
    )
    parser.add_argument("--model", type=str, default=GEMINI_FLASH_LITE)
    parser.add_argument("--no-thinking", action="store_true", help="Set thinking_budget=0")
    args = parser.parse_args()

    if args.user_id is not None or args.simulate_agent_outcome_id is not None:
        sim_id = resolve_simulate_agent_outcome_id(
            user_id=args.user_id,
            simulate_agent_outcome_id=args.simulate_agent_outcome_id,
        )
        built = build_plan_verbalizer_input(
            simulate_agent_outcome_id=sim_id,
            scenario_id=args.scenario_id,
        )
        print(f"Using simulate_agent_outcome_id={sim_id}")
        print("BUILT PLAN VERBALIZER INPUT")
        print("-" * 80)
        print(built)
        if args.print_input_only:
            return
        thinking_budget = 0 if args.no_thinking else PLAN_VERBALIZER_THINKING_BUDGET
        optimizer = PlanVerbalizerOptimizer(model_name=args.model, thinking_budget=thinking_budget)
        print("\nPLAN VERBALIZER LLM OUTPUT")
        print("-" * 80)
        print(json.dumps(optimizer.generate_response(built), indent=2))
        return

    if args.print_input_only:
        print("Error: --print-input-only requires --user-id or --simulate-agent-outcome-id", file=sys.stderr)
        raise SystemExit(1)

    if args.batch is None and args.test is None:
        _print_usage()
        return

    thinking_budget = 0 if args.no_thinking else PLAN_VERBALIZER_THINKING_BUDGET
    optimizer = PlanVerbalizerOptimizer(model_name=args.model, thinking_budget=thinking_budget)

    if args.batch is not None:
        cases = [tc for tc in TEST_CASES if int(tc.get("batch") or 0) == int(args.batch)]
        if not cases:
            raise SystemExit(f"No tests found for batch={args.batch}")
        for i, tc in enumerate(cases):
            if i:
                print("\n" + "-" * 80 + "\n")
            run_test(tc, optimizer)
        return

    if args.test is not None:
        if args.test.strip().lower() == "all":
            for i in range(len(TEST_CASES)):
                run_test(i, optimizer)
                if i < len(TEST_CASES) - 1:
                    print("\n" + "-" * 80 + "\n")
            return
        test_val: str | int = int(args.test) if args.test.isdigit() else args.test
        run_test(
            test_val,
            optimizer,
            scenario_id=args.scenario_id,
        )
        return


def _print_usage() -> None:
    print("Usage:")
    print("  Run a single test: --test <name_or_index>")
    print("  Run all tests: --test all")
    print("  Run batch: --batch <N>")
    print("  Build input from DB: --user-id <id> | --simulate-agent-outcome-id <id> [--scenario-id <id>]")
    print("  Print built input only: --user-id <id> --print-input-only")
    print("\nAvailable test cases:")
    for i, tc in enumerate(TEST_CASES):
        print(f"  [{i}] {tc['name']} (batch {tc.get('batch', '?')})")


if __name__ == "__main__":
    main()
