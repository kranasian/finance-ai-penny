"""
Optimizer runner for **P:PlansVerbalizer** (Gemini prompt tuning).

Input is verbalized `# Financial Need` (with `## Need Details`) and `# Financial Strategy` with two `## Plan A:` / `## Plan B:` sections.
Each plan section includes prose, `### Plan Details`, and `### Projection`.
Does not include `### Current Spending` — that input goes to **P:PlanSpendingBudgetVerbalizer**.

Objective: verbalize both plans in one JSON object with a `plans` array of exactly two entries.
Each entry includes `scenario_id`, `plan_title`, `plan_badge`, `plan_badge_severity_level`,
`concise_plan`, `full_plan`, `table_title`, `chart_title`, `chart_type`, and `chart_target_balance`.
Titles and badges must clearly differentiate the two options.

Run from `finance-ai-penny` repo root:

  python3 active_experiments/plans_verbalizer_optimizer.py --test 0
  python3 active_experiments/plans_verbalizer_optimizer.py --test all
  python3 active_experiments/plans_verbalizer_optimizer.py --simulate-agent-outcome-id 1148 --print-input-only
  python3 active_experiments/plans_verbalizer_optimizer.py --user-id 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import ClientError
except Exception:
    genai = None
    types = None
    ClientError = Exception

from active_experiments.verbalizer_optimizer_db import (
    _LLM_SERVER_ROOT,
    _fetch_ai_agent_outcome_row,
    _finalize_goal_plan_for_bundle,
    _load_slave_db_connect_kwargs,
    _resolve_ideal_response,
    load_simulate_agent_outcome_markdown,
    resolve_simulate_agent_outcome_id,
)

if str(_LLM_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(_LLM_SERVER_ROOT))

try:
    import psycopg2
except Exception:
    psycopg2 = None

from propose_next_steps.goal_plan_bundle import (
    build_plans_verbalizer_input_bundle,
    load_user_goal_plan,
    select_plans_for_verbalizer,
)
from propose_next_steps.goal_plan_narrative import FINANCIAL_NEED_H1, FINANCIAL_STRATEGY_H1, NEED_DETAILS_H2
from active_experiments.need_verbalizer_optimizer import (
    NeedVerbalizerOptimizer,
    _parse_model_json_object,
    build_need_verbalizer_input_bundle,
)

if load_dotenv is not None:
    load_dotenv()

GEMINI_FLASH_LITE = "gemini-flash-lite-latest"
PLANS_VERBALIZER_THINKING_BUDGET = 256
PLANS_VERBALIZER_MAX_OUTPUT_TOKENS = 4096

_CHART_TYPES = (
    "projected_total_credit_balance",
    "projected_total_depository_balance",
    "projected_combined_net_balance",
)

SYSTEM_PROMPT = """You are Penny — a clever, fun, and conversational money coach who helps people compare two financial plan options with clear, differentiated copy.

Tone & Style:
- Sound encouraging, upbeat, and conversational like a friendly financial partner who makes budgeting engaging.
- Use `we` / `our` when describing shared action steps, adjusting spending, establishing habits, and following the plan.
- Use `you` / `your` when referring to the user's personal accounts, balances, spending, debts, and savings goals.
- Keep the language punchy, approachable, and natural. Avoid robotic phrasing and dry financial jargon.
- When monetary figures or timelines are mentioned, use standard currency symbols and digits (never spell out numbers in words).

Context & Inputs:
Use `# Financial Need`, `## Need Details`, and `# Financial Strategy` with two plan sections: `## Plan A: {scenario_id}` and `## Plan B: {scenario_id}`. Each plan section includes prose, `### Plan Details`, then `### Projection`. Use `### Existing plan name` on a plan only when present for that plan.

Differentiation rules:
- Return exactly two plan objects in `plans`, one per `## Plan A:` / `## Plan B:` heading in the input.
- Each `scenario_id` must match the input heading exactly.
- `plan_title`, `plan_badge`, `concise_plan`, and `full_plan` must be clearly distinct between the two plans so the user can tell them apart at a glance.
- Ground each plan in that plan's pacing and caps, not the sibling plan's schedule.
- If one plan is gentler and one is more aggressive, reflect that contrast in titles, badges, and severity levels.

Per-plan field guidelines:
- `plan_title`: fun, punchy, action-oriented (max 5 words and 40 characters; no jargon).
- `plan_badge`: descriptive motivating phrase in Title Case reflecting adjustment intensity.
- `plan_badge_severity_level`: integer 1–5 (1 = gentle, 5 = intensive).
- `concise_plan`: one sentence TLDR (max 18 words) focused on lifestyle steps, not dollar lists.
- `full_plan`: supporting breakdown (max 40 words) prioritizing actions over granular amounts. State when the plan starts using the first month in that plan's `### Plan Details`.
- `table_title`: short title for the spending comparison table (max 6 words and 40 characters).
- `chart_title`: short chart title (max 6 words and 40 characters).
- `chart_type`: `projected_total_credit_balance`, `projected_total_depository_balance`, or `projected_combined_net_balance`.
- `chart_target_balance`: integer goal line (`0` for full credit payoff, payoff floor for partial paydown, or savings target for depository charts).

Output compact JSON only — no extra fields.
"""


def _build_output_schema() -> "types.Schema":
    if types is None:
        raise RuntimeError("Install `google-genai` for this optimizer.")
    plan_schema = types.Schema(
        type=types.Type.OBJECT,
        required=[
            "scenario_id",
            "plan_title",
            "plan_badge",
            "plan_badge_severity_level",
            "concise_plan",
            "full_plan",
            "table_title",
            "chart_title",
            "chart_type",
            "chart_target_balance",
        ],
        properties={
            "scenario_id": types.Schema(
                type=types.Type.STRING,
                description=(
                    "Must match one of the ## Plan A: or ## Plan B: scenario_id headings in the input."
                ),
            ),
            "plan_title": types.Schema(
                type=types.Type.STRING,
                description=(
                    "Punchy, fun action-oriented title for what we will do (max 5 words and 40 "
                    "characters; no jargon). Grounded in pacing and caps. Must differ from the other "
                    "plan title."
                ),
            ),
            "plan_badge": types.Schema(
                type=types.Type.STRING,
                description=(
                    "Descriptive word or phrase in Title Case from which plan difficulty and "
                    "adjustment intensity can be inferred."
                ),
            ),
            "plan_badge_severity_level": types.Schema(
                type=types.Type.INTEGER,
                description=(
                    "Integer from 1 to 5 representing the plan difficulty and adjustment intensity "
                    "(5 = highest difficulty)."
                ),
            ),
            "concise_plan": types.Schema(
                type=types.Type.STRING,
                description=(
                    "One sentence TLDR (max 18 words) focusing on concrete lifestyle steps and "
                    "actions to fulfill the goal rather than listing specific dollar amounts."
                ),
            ),
            "full_plan": types.Schema(
                type=types.Type.STRING,
                description=(
                    "Expounded plan summary (max 40 words) prioritizing specific actions, "
                    "behavioral adjustments, and goal resolution over granular category amounts. "
                    "State when the plan starts using the first month in ### Plan Details."
                ),
            ),
            "table_title": types.Schema(
                type=types.Type.STRING,
                description="Short title for the spending comparison table (max 6 words and 40 characters).",
            ),
            "chart_title": types.Schema(
                type=types.Type.STRING,
                description=(
                    "Short title describing the projected chart outcome (max 6 words and 40 characters)."
                ),
            ),
            "chart_type": types.Schema(
                type=types.Type.STRING,
                enum=list(_CHART_TYPES),
                description=(
                    "Projected chart showing the primary outcome: credit payoff, savings/depository "
                    "buffer, or combined net balance."
                ),
            ),
            "chart_target_balance": types.Schema(
                type=types.Type.INTEGER,
                description=(
                    "Integer goal line: 0 for full credit payoff, payoff floor for partial paydown, "
                    "or savings target for depository charts."
                ),
            ),
        },
    )
    return types.Schema(
        type=types.Type.OBJECT,
        required=["plans"],
        properties={
            "plans": types.Schema(
                type=types.Type.ARRAY,
                description="Exactly two differentiated plan verbalizations, one per scenario_id in the input.",
                items=plan_schema,
            ),
        },
    )


def _validate_plan_entry(parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        raise ValueError("Each plan must be a JSON object")
    scenario_id = parsed.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id.strip():
        raise ValueError("scenario_id must be a non-empty string")
    plan_title = parsed.get("plan_title")
    if not isinstance(plan_title, str) or not plan_title.strip():
        raise ValueError("plan_title must be a non-empty string")
    plan_badge = parsed.get("plan_badge")
    if not isinstance(plan_badge, str) or not plan_badge.strip():
        raise ValueError("plan_badge must be a non-empty string")
    plan_badge_severity_level = parsed.get("plan_badge_severity_level")
    try:
        plan_badge_severity_level = int(plan_badge_severity_level)
    except (TypeError, ValueError) as exc:
        raise ValueError("plan_badge_severity_level must be an integer") from exc
    if plan_badge_severity_level < 1 or plan_badge_severity_level > 5:
        raise ValueError("plan_badge_severity_level must be an integer from 1 to 5")
    concise_plan = parsed.get("concise_plan")
    if not isinstance(concise_plan, str) or not concise_plan.strip():
        raise ValueError("concise_plan must be a non-empty string")
    full_plan = parsed.get("full_plan")
    if not isinstance(full_plan, str) or not full_plan.strip():
        raise ValueError("full_plan must be a non-empty string")
    table_title = parsed.get("table_title")
    if not isinstance(table_title, str) or not table_title.strip():
        raise ValueError("table_title must be a non-empty string")
    chart_title = parsed.get("chart_title")
    if not isinstance(chart_title, str) or not chart_title.strip():
        raise ValueError("chart_title must be a non-empty string")
    chart_type = parsed.get("chart_type")
    if not isinstance(chart_type, str) or chart_type not in _CHART_TYPES:
        raise ValueError(f"chart_type must be one of: {', '.join(_CHART_TYPES)}")
    chart_target_balance = parsed.get("chart_target_balance")
    try:
        chart_target_balance = int(chart_target_balance)
    except (TypeError, ValueError) as exc:
        raise ValueError("chart_target_balance must be an integer") from exc
    if chart_target_balance < 0:
        raise ValueError("chart_target_balance must be >= 0")
    return {
        "scenario_id": scenario_id.strip(),
        "plan_title": plan_title.strip(),
        "plan_badge": plan_badge.strip(),
        "plan_badge_severity_level": plan_badge_severity_level,
        "concise_plan": concise_plan.strip(),
        "full_plan": full_plan.strip(),
        "table_title": table_title.strip(),
        "chart_title": chart_title.strip(),
        "chart_type": chart_type,
        "chart_target_balance": chart_target_balance,
    }


def _validate_plans_response(parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        raise ValueError("Response must be a JSON object")
    plans_raw = parsed.get("plans")
    if not isinstance(plans_raw, list) or len(plans_raw) != 2:
        raise ValueError("plans must be an array of exactly 2 plan objects")
    plans = [_validate_plan_entry(entry) for entry in plans_raw]
    scenario_ids = [entry["scenario_id"] for entry in plans]
    if len(set(scenario_ids)) != 2:
        raise ValueError("plans must include two distinct scenario_id values")
    titles = [entry["plan_title"].lower() for entry in plans]
    if titles[0] == titles[1]:
        raise ValueError("plan_title values must differ between the two plans")
    return {"plans": plans}


TEST_CASES: list[dict[str, Any]] = [
    {
        "name": "debt_paydown_pair",
        "batch": 1,
        "input": """
# Financial Need

$312 in interest every 90 days on your $8,400 balance while spending tracks income.

## Need Details

Interest tool: **$312** on Venture in 90 days. Next due **2026-04-18** per payment schedule.

# Financial Strategy

## Plan A: gradual_paydown_savings

Goal: pay Venture to **$0**. Phased dining and leisure trims keep month-1 cuts modest, then deepen after month 3; route **$200**/mo to savings only after the card hits **$0**.

### Plan Details

- 04/26-06/26: Cap food $850, leisure $450 monthly
- 07/26-03/28: Cap food $700, leisure $350 monthly

### Projection

- Projection: 12 mo, stop goal achieved.

## Plan B: steady_cut

Goal: pay Venture to **$0**. Flat **$700** food and **$350** leisure from month 1 clears the card about two months sooner but leaves thinner checking buffers in the first quarter.

### Plan Details

- 04/26-03/28: Cap food $700, leisure $350 monthly

### Projection

- Projection: 8 mo, stop goal achieved.
""",
        "ideal_response": {
            "plans": [
                {
                    "scenario_id": "gradual_paydown_savings",
                    "plan_title": "Phase cuts, clear Venture",
                    "plan_badge": "Light Phased Trim",
                    "plan_badge_severity_level": 2,
                    "concise_plan": "Phased cuts pay Venture to $0, then save $200/mo.",
                    "full_plan": "Pay Venture to $0 with phased food and leisure cuts, then save $200/mo.",
                    "table_title": "Spending vs plan budget",
                    "chart_title": "Credit balance paydown",
                    "chart_type": "projected_total_credit_balance",
                    "chart_target_balance": 0,
                },
                {
                    "scenario_id": "steady_cut",
                    "plan_title": "Cut food now, clear Venture",
                    "plan_badge": "Immediate Deep Cut",
                    "plan_badge_severity_level": 4,
                    "concise_plan": "Cut food and leisure from month one to clear Venture faster.",
                    "full_plan": "Pay Venture to $0 with food at $700/mo and leisure at $350/mo from month one.",
                    "table_title": "Spending vs plan budget",
                    "chart_title": "Credit balance paydown",
                    "chart_type": "projected_total_credit_balance",
                    "chart_target_balance": 0,
                },
            ],
        },
    },
    {
        "name": "cash_flow_pair",
        "batch": 1,
        "input": """
# Financial Need

Credit card balance sits at **$4,200** and keeps climbing with minimum-style payments.

## Need Details

Card balance **$4,200**. Interest about **$90**/mo at the current APR. Forecast committed outflows **$3,600**/mo vs income **$4,000**/mo.

# Financial Strategy

## Plan A: protect_fixed_cut_flex

Goal: pay the card down to **$3,000**. Trim **$200**/mo from food and shopping months 1–3, then reassess.

### Plan Details

- 04/26-03/28: Cap food $520, shopping $180 monthly

### Projection

- Projection: 6 mo, stop goal achieved.

## Plan B: aggressive_flex_cut

Goal: pay the card down to **$1,500**. Cut food to **$450** and shopping to **$150** immediately — deeper paydown, but less breathing room if spending drifts.

### Plan Details

- 04/26-03/28: Cap food $450, shopping $150 monthly

### Projection

- Projection: 5 mo, stop goal achieved.
""",
        "ideal_response": {
            "plans": [
                {
                    "scenario_id": "protect_fixed_cut_flex",
                    "plan_title": "Trim spend, reach $3,000",
                    "plan_badge": "Steady Mid Cut",
                    "plan_badge_severity_level": 3,
                    "concise_plan": "Trim food and shopping to reach a $3,000 balance.",
                    "full_plan": "Pay the card down to $3,000 while trimming food and shopping.",
                    "table_title": "Spending vs plan budget",
                    "chart_title": "Credit balance paydown",
                    "chart_type": "projected_total_credit_balance",
                    "chart_target_balance": 3000,
                },
                {
                    "scenario_id": "aggressive_flex_cut",
                    "plan_title": "Cut food, hit $1,500",
                    "plan_badge": "Tight Buffer Sprint",
                    "plan_badge_severity_level": 5,
                    "concise_plan": "Cap food and shopping to pay the card to $1,500.",
                    "full_plan": "Pay the card down to $1,500 with food at $450/mo and shopping at $150/mo.",
                    "table_title": "Spending vs plan budget",
                    "chart_title": "Credit balance paydown",
                    "chart_type": "projected_total_credit_balance",
                    "chart_target_balance": 1500,
                },
            ],
        },
    },
    {
        "name": "slow_debt_pair",
        "batch": 2,
        "input": """
# Financial Need

Balance rose $300 in three months despite $115/mo payments on $4,800 owed.

## Need Details

Balance up **$300** over three months despite **$115**/mo payments. APR tool: **~21.8%** on Platinum.

# Financial Strategy

## Plan A: balanced_trim

Goal: pay Platinum to **$0** by **Dec 2026**. Trim food to **$520** and leisure to **$300** from month 1.

### Plan Details

- 04/26-03/28: Cap food $520, leisure $300 monthly

### Projection

- Projection: 12 mo, stop goal achieved.

## Plan B: leisure_first

Goal: pay Platinum down to **$2,000**. Protect leisure at **$380** but cut food harder to **$450**.

### Plan Details

- 04/26-03/28: Cap food $450, leisure $380 monthly

### Projection

- Projection: 9 mo, stop goal achieved.
""",
        "ideal_response": {
            "plans": [
                {
                    "scenario_id": "balanced_trim",
                    "plan_title": "Trim meals, clear Platinum",
                    "plan_badge": "Steady Mid Cut",
                    "plan_badge_severity_level": 3,
                    "concise_plan": "Cap food and leisure to clear Platinum by Dec 2026.",
                    "full_plan": "Pay Platinum to $0 by Dec 2026 with food at $520/mo and leisure at $300/mo.",
                    "table_title": "Spending vs plan budget",
                    "chart_title": "Credit balance paydown",
                    "chart_type": "projected_total_credit_balance",
                    "chart_target_balance": 0,
                },
                {
                    "scenario_id": "leisure_first",
                    "plan_title": "Protect fun, hit $2,000",
                    "plan_badge": "Leisure-First Squeeze",
                    "plan_badge_severity_level": 3,
                    "concise_plan": "Hold leisure steadier while cutting food to reach $2,000.",
                    "full_plan": "Pay Platinum down to $2,000 with leisure at $380/mo and food at $450/mo.",
                    "table_title": "Spending vs plan budget",
                    "chart_title": "Credit balance paydown",
                    "chart_type": "projected_total_credit_balance",
                    "chart_target_balance": 2000,
                },
            ],
        },
    },
    {
        "name": "emergency_savings_pair",
        "batch": 3,
        "input": """
# Financial Need

You want an emergency buffer of **$6,000**, but your current savings is only **$1,000**, and spending is close to income.

## Need Details

Savings gap is **$5,000** to reach **$6,000**. Committed spend leaves little slack, so steady discretionary cuts are needed.

# Financial Strategy

## Plan A: save_to_emergency

Goal: save **$6,000**. Trim food and leisure for the first stretch, then hold steady to finish the gap.

### Plan Details

- 04/26-03/28: Cap food $520, leisure $300, shopping $50 monthly

### Projection

- Projection: 12 mo, stop goal achieved.

## Plan B: faster_saving

Goal: save **$4,000** sooner with harder cuts, but less breathing room if spending drifts.

### Plan Details

- 04/26-03/28: Cap food $450, leisure $200, shopping $30 monthly

### Projection

- Projection: 8 mo, stop goal achieved.
""",
        "ideal_response": {
            "plans": [
                {
                    "scenario_id": "save_to_emergency",
                    "plan_title": "Hold spend, save $6,000",
                    "plan_badge": "Steady Savings Push",
                    "plan_badge_severity_level": 3,
                    "concise_plan": "Hold food and leisure steady to save $6,000.",
                    "full_plan": "Save $6,000 by keeping food at $520/mo and leisure at $300/mo, then holding to finish the gap.",
                    "table_title": "Spending vs plan budget",
                    "chart_title": "Savings balance growth",
                    "chart_type": "projected_total_depository_balance",
                    "chart_target_balance": 6000,
                },
                {
                    "scenario_id": "faster_saving",
                    "plan_title": "Cut harder, save faster",
                    "plan_badge": "Aggressive Savings Sprint",
                    "plan_badge_severity_level": 4,
                    "concise_plan": "Tighter food and leisure caps to reach $4,000 sooner.",
                    "full_plan": "Save $4,000 faster with food at $450/mo and leisure at $200/mo.",
                    "table_title": "Spending vs plan budget",
                    "chart_title": "Savings balance growth",
                    "chart_type": "projected_total_depository_balance",
                    "chart_target_balance": 4000,
                },
            ],
        },
    },
    {
        "name": "credit_freedom_pair",
        "batch": 1,
        "input": """
# Financial Need

Managing your credit balances can feel overwhelming when you are working toward financial freedom.

## Need Details

Your current credit card balances are impacting your financial flexibility. You are building a path to pay down this debt and reduce your reliance on credit cards.

# Financial Strategy

## Plan A: accelerated_debt_freedom

This plan offers a disciplined, accelerated path that prioritizes immediate debt clearance and early capital accumulation by setting sustainable discretionary caps that remain respectful of your realistic floor. By immediately reducing discretionary spending, you break the cycle of credit reliance in just 6 months and successfully reach the $35,000 savings milestone by February 2030.

### Plan Details

- 10/26-: Cap food $1350, leisure $250, shopping $200, uncategorized $60 monthly

### Projection

- Projection: 41 mo, stop goal achieved.

## Plan B: balanced_phased_growth

This is a strong second choice if you prefer a gentler transition, as it allows for moderate adjustments for the first 6 months before shifting into more rigorous savings in month 7. It acknowledges your need for lifestyle stability in the near term while still ensuring you achieve your long-term goal of a 6-month runway by October 2030.

### Plan Details

- 10/26-03/27: Cap food $1600, leisure $450, shopping $300, uncategorized $80 monthly
- 04/27-: Cap food $1400, leisure $300, shopping $200, uncategorized $50 monthly

### Projection

- Projection: 49 mo, stop goal achieved.
""",
        "ideal_response": {
            "plans": [
                {
                    "scenario_id": "accelerated_debt_freedom",
                    "plan_title": "Fast debt freedom sprint",
                    "plan_badge": "Accelerated Paydown",
                    "plan_badge_severity_level": 5,
                    "concise_plan": "Cut discretionary spend now to clear debt and reach $35,000 savings.",
                    "full_plan": "Tighten food, leisure, shopping, and uncategorized caps immediately to break credit reliance and build savings faster.",
                    "table_title": "Spending vs plan budget",
                    "chart_title": "Credit balance paydown",
                    "chart_type": "projected_total_credit_balance",
                    "chart_target_balance": 0,
                },
                {
                    "scenario_id": "balanced_phased_growth",
                    "plan_title": "Phase in, build runway",
                    "plan_badge": "Gentle Phased Trim",
                    "plan_badge_severity_level": 2,
                    "concise_plan": "Ease into tighter caps after six months while building a 6-month runway.",
                    "full_plan": "Start with moderate food and leisure caps, then tighten in month 7 to reach your runway goal by October 2030.",
                    "table_title": "Spending vs plan budget",
                    "chart_title": "Savings balance growth",
                    "chart_type": "projected_total_depository_balance",
                    "chart_target_balance": 35000,
                },
            ],
        },
    },
]


def format_plans_verbalizer_user_message(profile_input: str) -> str:
    body = (profile_input or "").strip()
    if not body:
        raise ValueError("profile_input must be non-empty markdown.")
    if FINANCIAL_NEED_H1 not in body:
        raise ValueError(f"profile_input must include {FINANCIAL_NEED_H1}.")
    if NEED_DETAILS_H2 not in body:
        raise ValueError(f"profile_input must include {NEED_DETAILS_H2}.")
    if FINANCIAL_STRATEGY_H1 not in body:
        raise ValueError(f"profile_input must include {FINANCIAL_STRATEGY_H1}.")
    if "## Plan A:" not in body or "## Plan B:" not in body:
        raise ValueError("profile_input must include ## Plan A: and ## Plan B: sections.")
    return body + "\n"


def build_plans_verbalizer_input(
    *,
    simulate_agent_outcome_id: int | None = None,
    user_id: int | None = None,
    need_verbalizer_response: dict[str, Any] | None = None,
    need_optimizer: NeedVerbalizerOptimizer | None = None,
) -> str:
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

    if psycopg2 is None:
        raise RuntimeError("Missing dependency `psycopg2`.")
    conn = psycopg2.connect(**_load_slave_db_connect_kwargs())
    try:
        goal_plan = load_user_goal_plan(conn, user_id=sim_uid)
    finally:
        conn.close()
    if goal_plan is None:
        raise ValueError(
            f"user_plans is empty for user_id={sim_uid}; "
            "run simulate_financial_strategy with persistence first"
        )
    goal_plan = _finalize_goal_plan_for_bundle(
        goal_plan,
        simulate_md,
        simulate_calls=sim_row.get("calls"),
    )
    scenarios = select_plans_for_verbalizer(goal_plan)
    profile_input, _, _ = build_plans_verbalizer_input_bundle(
        need_verbalizer_response=need_verbalizer_response,
        simulate_outcome_md=simulate_md,
        goal_plan_scenarios=scenarios,
    )
    return profile_input


def resolve_plans_test_case_input(test_case: dict[str, Any]) -> str:
    for key in ("input", "bundled_input"):
        raw = test_case.get(key)
        if isinstance(raw, str) and raw.strip():
            text = raw.strip() + "\n"
            if FINANCIAL_NEED_H1 not in text:
                raise ValueError("test case input must include # Financial Need")
            if NEED_DETAILS_H2 not in text:
                raise ValueError("test case input must include ## Need Details")
            if FINANCIAL_STRATEGY_H1 not in text:
                raise ValueError("test case input must include # Financial Strategy")
            if "## Plan A:" not in text or "## Plan B:" not in text:
                raise ValueError("test case input must include ## Plan A: and ## Plan B:")
            plan_a = text.split("## Plan A:", 1)[1].split("## Plan B:", 1)[0]
            plan_b = text.split("## Plan B:", 1)[1]
            for label, section in (("## Plan A:", plan_a), ("## Plan B:", plan_b)):
                if "### Plan Details" not in section:
                    raise ValueError(f"each plan section must include ### Plan Details ({label})")
                details_idx = section.index("### Plan Details")
                projection_idx = section.find("### Projection")
                if projection_idx < 0 or projection_idx < details_idx:
                    raise ValueError(f"each plan section must place ### Projection after ### Plan Details ({label})")
            return text
    raise ValueError("test case must include bundled input")


def _parse_plans_json_response(text: str) -> dict[str, Any]:
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


class PlansVerbalizerOptimizer:
    def __init__(
        self,
        model_name: str = GEMINI_FLASH_LITE,
        *,
        thinking_budget: int = PLANS_VERBALIZER_THINKING_BUDGET,
        max_output_tokens: int = PLANS_VERBALIZER_MAX_OUTPUT_TOKENS,
    ):
        if genai is None or types is None:
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
        user_text = format_plans_verbalizer_user_message(profile_input)
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
                parsed = _parse_plans_json_response(output_text)
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
                return _validate_plans_response(parsed)
            except ValueError as exc:
                raise ValueError(f"Response failed validation: {exc}") from exc

        if last_error is not None:
            raise last_error
        raise ValueError("Invalid JSON response from model.")


def _run_test(
    profile_input: str,
    optimizer: PlansVerbalizerOptimizer | None = None,
    *,
    ideal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if optimizer is None:
        optimizer = PlansVerbalizerOptimizer()
    wrapped = format_plans_verbalizer_user_message(profile_input)
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
    optimizer: PlansVerbalizerOptimizer | None = None,
) -> dict[str, Any] | None:
    if optimizer is None:
        optimizer = PlansVerbalizerOptimizer()

    if isinstance(test_name_or_index_or_dict, dict):
        tc = test_name_or_index_or_dict
        name = tc.get("name", "custom_test")
        try:
            payload = resolve_plans_test_case_input(tc)
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
    print(f"\n{'=' * 80}\nRunning test: {name}\n{'=' * 80}\n")
    ideal = _resolve_ideal_response(tc)
    return _run_test(resolve_plans_test_case_input(tc), optimizer, ideal=ideal)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P:PlansVerbalizer optimizer tests")
    parser.add_argument("--test", type=str, help='Test name or index (e.g. "0" or "debt_paydown_pair")')
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
        built = build_plans_verbalizer_input(simulate_agent_outcome_id=sim_id)
        print(f"Using simulate_agent_outcome_id={sim_id}")
        print("BUILT PLANS VERBALIZER INPUT")
        print("-" * 80)
        print(built)
        if args.print_input_only:
            return
        thinking_budget = 0 if args.no_thinking else PLANS_VERBALIZER_THINKING_BUDGET
        optimizer = PlansVerbalizerOptimizer(model_name=args.model, thinking_budget=thinking_budget)
        print("\nPLANS VERBALIZER LLM OUTPUT")
        print("-" * 80)
        print(json.dumps(optimizer.generate_response(built), indent=2))
        return

    if args.print_input_only:
        print("Error: --print-input-only requires --user-id or --simulate-agent-outcome-id", file=sys.stderr)
        raise SystemExit(1)

    if args.batch is None and args.test is None:
        _print_usage()
        return

    thinking_budget = 0 if args.no_thinking else PLANS_VERBALIZER_THINKING_BUDGET
    optimizer = PlansVerbalizerOptimizer(model_name=args.model, thinking_budget=thinking_budget)

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
        run_test(test_val, optimizer)
        return


def _print_usage() -> None:
    print("Usage:")
    print("  Run a single test: --test <name_or_index>")
    print("  Run all tests: --test all")
    print("  Run batch: --batch <N>")
    print("  Build input from DB: --user-id <id> | --simulate-agent-outcome-id <id>")
    print("  Print built input only: --user-id <id> --print-input-only")
    print("\nAvailable test cases:")
    for i, tc in enumerate(TEST_CASES):
        print(f"  [{i}] {tc['name']} (batch {tc.get('batch', '?')})")


if __name__ == "__main__":
    main()
