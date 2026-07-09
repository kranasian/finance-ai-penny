"""
Optimizer runner for **P:PlanVerbalizer** (Gemini prompt tuning).

Input is trimmed ``# Financial Needs`` markdown plus the matching ``# Financial Strategy`` subsection
(recommended or alternative only) and a ``### Spending Schedule`` block with compact spending
bullets for one scenario (recommended by default, or
``--scenario-id``).

Objective: verbalize that one plan as ``plan_title``, ``plan_summary``, and ``plan_details``
(``spending_phase_descriptions``, ``payoff``, ``trade_off`` from the model; ``spending_phases`` with
``period``, ``caps``, and ``description`` assembled after the call from ``### Spending Schedule``).

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
    _format_goal_plan_narrative,
    _parse_model_json_object,
    ensure_blank_line_after_plan_headings,
    trim_simulate_outcome_for_plan_bundle,
)

if load_dotenv is not None:
    load_dotenv()

GEMINI_FLASH_LITE = "gemini-flash-lite-latest"
PLAN_VERBALIZER_THINKING_BUDGET = 128
PLAN_VERBALIZER_MAX_OUTPUT_TOKENS = 2048

SYSTEM_PROMPT = """You are Penny — a sharp, witty money coach who explains one financial plan in clear, concrete detail.

Use the financial-needs context, matching plan prose in ``# Financial Strategy``, and the spending caps under ``### Spending Schedule``.

- ``plan_title``: short headline for this plan (max **8 words**; punchy, no jargon).
- ``plan_summary``: one sentence on what this plan does (max **25 words**); ground every **$** and date in the input.
- ``plan_details``: JSON object with:
  - ``spending_phase_descriptions``: array with one entry per spending-schedule line, in order. Each entry is one sentence describing that phase (max **20 words**). Name only categories whose caps change in that phase; do not list every dollar amount.
  - ``payoff``: one sentence — balance or savings targets and interest impact when stated (max **20 words**).
  - ``trade_off``: one sentence — main sacrifice from ``# Financial Strategy`` (max **20 words**).
  Exact ``period`` and caps live under ``### Spending Schedule`` — do not repeat them in descriptions.

Keep ``plan_summary`` to one sentence (max **25 words**). Output compact JSON only — no extra fields or whitespace padding.
"""


def _build_plan_details_schema() -> "types.Schema":
    if types is None:  # pragma: no cover
        raise RuntimeError("Install `google-genai` for this optimizer.")
    return types.Schema(
        type=types.Type.OBJECT,
        required=["spending_phase_descriptions", "payoff", "trade_off"],
        properties={
            "spending_phase_descriptions": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(
                    type=types.Type.STRING,
                    description="One-sentence description of a spending-schedule phase.",
                ),
                description="One entry per ### Spending Schedule line, in order.",
            ),
            "payoff": types.Schema(
                type=types.Type.STRING,
                description="Payoff timeline, balance or savings targets, and interest impact when stated.",
            ),
            "trade_off": types.Schema(
                type=types.Type.STRING,
                description="Main sacrifice or risk for this plan.",
            ),
        },
    )


def _build_output_schema() -> "types.Schema":
    if types is None:  # pragma: no cover
        raise RuntimeError("Install `google-genai` for this optimizer.")
    return types.Schema(
        type=types.Type.OBJECT,
        required=["plan_title", "plan_summary", "plan_details"],
        properties={
            "plan_title": types.Schema(
                type=types.Type.STRING,
                description="Short headline for this plan (max 8 words).",
            ),
            "plan_summary": types.Schema(
                type=types.Type.STRING,
                description="One-sentence summary of what this plan does (max 25 words).",
            ),
            "plan_details": _build_plan_details_schema(),
        },
    )


SPENDING_SCHEDULE_H3 = "### Spending Schedule"

_RE_SPENDING_SCHEDULE_BULLET = re.compile(r"^-\s+(.+):\s+Cap\s+(.+)$")


def _spending_phases_from_bundle(bundle_md: str) -> list[dict[str, str]]:
    if SPENDING_SCHEDULE_H3 not in bundle_md:
        return []
    block_lines: list[str] = []
    started = False
    for line in bundle_md.splitlines():
        if line.strip() == SPENDING_SCHEDULE_H3:
            started = True
            continue
        if not started:
            continue
        stripped = line.strip()
        if stripped.startswith("#"):
            break
        if stripped:
            block_lines.append(stripped)
    phases: list[dict[str, str]] = []
    for stripped in block_lines:
        match = _RE_SPENDING_SCHEDULE_BULLET.match(stripped)
        if not match:
            continue
        period = match.group(1).strip()
        if period.endswith("-"):
            period = f"{period}:"
        phases.append({
            "period": period,
            "caps": match.group(2).strip(),
        })
    return phases


def _attach_spending_phases(response: dict[str, Any], profile_input: str) -> dict[str, Any]:
    phases = _spending_phases_from_bundle(profile_input)
    if not phases:
        return response
    plan_details = dict(response["plan_details"])
    descriptions = plan_details.pop("spending_phase_descriptions", None)
    if not isinstance(descriptions, list) or not descriptions:
        raise ValueError("plan_details.spending_phase_descriptions must be a non-empty array")
    if len(descriptions) != len(phases):
        raise ValueError(
            "plan_details.spending_phase_descriptions length must match "
            f"spending-schedule lines ({len(phases)} expected, {len(descriptions)} got)"
        )
    merged_phases: list[dict[str, str]] = []
    for phase, description in zip(phases, descriptions, strict=True):
        if not isinstance(description, str) or not description.strip():
            raise ValueError("each spending_phase_descriptions entry must be a non-empty string")
        merged_phases.append({
            **phase,
            "description": description.strip(),
        })
    plan_details["spending_phases"] = merged_phases
    return {**response, "plan_details": plan_details}


def _validate_plan_details(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("plan_details must be a JSON object")
    descriptions = value.get("spending_phase_descriptions")
    if not isinstance(descriptions, list) or not descriptions:
        raise ValueError("plan_details.spending_phase_descriptions must be a non-empty array")
    normalized_descriptions: list[str] = []
    for i, description in enumerate(descriptions):
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"plan_details.spending_phase_descriptions[{i}] must be a non-empty string")
        normalized_descriptions.append(description.strip())
    payoff = value.get("payoff")
    if not isinstance(payoff, str) or not payoff.strip():
        raise ValueError("plan_details.payoff must be a non-empty string")
    trade_off = value.get("trade_off")
    if not isinstance(trade_off, str) or not trade_off.strip():
        raise ValueError("plan_details.trade_off must be a non-empty string")
    return {
        "spending_phase_descriptions": normalized_descriptions,
        "payoff": payoff.strip(),
        "trade_off": trade_off.strip(),
    }


def _validate_plan_response(parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        raise ValueError("Response must be a JSON object")
    plan_title = parsed.get("plan_title")
    if not isinstance(plan_title, str) or not plan_title.strip():
        raise ValueError("plan_title must be a non-empty string")
    plan_summary = parsed.get("plan_summary")
    if not isinstance(plan_summary, str) or not plan_summary.strip():
        raise ValueError("plan_summary must be a non-empty string")
    plan_details = _validate_plan_details(parsed.get("plan_details"))
    return {
        "plan_title": plan_title.strip(),
        "plan_summary": plan_summary.strip(),
        "plan_details": plan_details,
    }


TEST_CASES: list[dict[str, Any]] = [
    {
        "name": "debt_paydown_recommended",
        "batch": 1,
        "input": """
# Financial Needs

## Primary needs
1. **Reduce interest drag**: Venture balance **$8,400** with **$312** interest paid over 90 days while spending tracks near income.

## Evidence
* **Reduce interest drag**
  - Interest tool: **$312** on Venture in 90 days.
  - Next due **2026-04-18** per payment schedule.

# Financial Strategy

## Recommended plan: gradual_paydown_savings
* Phased dining and leisure trims keep month-1 cuts modest, then deepen after month 3 while routing **$200**/mo to savings once the card hits **$0**.

## Alternative plan: steady_cut
* Flat **$700** food and **$350** leisure from month 1 hits **$0** debt about two months sooner but leaves thinner checking buffers in the first quarter.

### Spending Schedule
- 04/26-06/26: Cap food $850, leisure $450 monthly
- 07/26-03/28: Cap food $700, leisure $350 monthly
""",
        "ideal_response": {
            "plan_title": "Gradual paydown path",
            "plan_summary": "Phase dining and leisure cuts, then route $200/mo to savings after Venture hits $0.",
            "plan_details": {
                "payoff": "Route $200/mo to savings once Venture hits $0; savings target $6,500.",
                "trade_off": "Phased trims keep month-1 cuts modest — debt clears slower than steady_cut but checking buffers stay thicker in the first quarter.",
                "spending_phases": [
                    {
                        "period": "04/26-06/26",
                        "caps": "food $850, leisure $450 monthly",
                        "description": "Modest dining and leisure trim for the first three months.",
                    },
                    {
                        "period": "07/26-03/28",
                        "caps": "food $700, leisure $350 monthly",
                        "description": "Deeper food and leisure caps through Mar '28.",
                    },
                ],
            },
        },
    },
    {
        "name": "debt_paydown_alternative",
        "batch": 1,
        "input": """
# Financial Needs

## Primary needs
1. **Reduce interest drag**: Venture balance **$8,400** with **$312** interest paid over 90 days while spending tracks near income.

## Evidence
* **Reduce interest drag**
  - Interest tool: **$312** on Venture in 90 days.
  - Next due **2026-04-18** per payment schedule.

# Financial Strategy

## Recommended plan: gradual_paydown_savings
* Phased dining and leisure trims keep month-1 cuts modest, then deepen after month 3 while routing **$200**/mo to savings once the card hits **$0**.

## Alternative plan: steady_cut
* Flat **$700** food and **$350** leisure from month 1 hits **$0** debt about two months sooner but leaves thinner checking buffers in the first quarter.

### Spending Schedule
- 04/26-03/28: Cap food $700, leisure $350 monthly
""",
        "ideal_response": {
            "plan_title": "Steady cut from day one",
            "plan_summary": "Hold food at $700/mo and leisure at $350/mo from month one to clear Venture faster.",
            "plan_details": {
                "payoff": "Targets $0 Venture balance about two months sooner than gradual_paydown_savings.",
                "trade_off": "Thinner checking buffers in the first quarter while cuts hit immediately.",
                "spending_phases": [
                    {
                        "period": "04/26-03/28",
                        "caps": "food $700, leisure $350 monthly",
                        "description": "Flat food and leisure caps from day one — no ramp.",
                    },
                ],
            },
        },
    },
    {
        "name": "cash_flow_recommended",
        "batch": 1,
        "input": """
# Financial Needs

## Primary needs
1. **Stabilize cash flow**: Checking **$800** with **$2,100** mortgage due **2026-04-01** — liquidity risk before flexible spend cuts matter.

## Evidence
* **Stabilize cash flow**
  - Checking **$800** vs mortgage **$2,100** on the 1st.
  - Forecast committed outflows **$3,600**/mo vs income **$4,000**/mo.

# Financial Strategy

## Recommended plan: protect_fixed_cut_flex
* Hold checking above **$2,200** before the mortgage, trim **$200**/mo from food and shopping months 1–3, then reassess.

## Alternative plan: aggressive_flex_cut
* Cut food to **$450** and shopping to **$150** immediately — debt-free by **Aug 2026** but checking may dip below **$500** in April.

### Spending Schedule
- 04/26-03/28: Cap food $520, shopping $180 monthly
""",
        "ideal_response": {
            "plan_title": "Protect fixed, trim flex",
            "plan_summary": "Hold checking above $2,200 before the mortgage while trimming food and shopping.",
            "plan_details": {
                "payoff": "Keep checking above $2,200 before the $2,100 mortgage due 2026-04-01; trim flexible spend $200/mo in months 1–3, then reassess.",
                "trade_off": "Debt-free timing is slower than aggressive_flex_cut, but liquidity risk eases first.",
                "spending_phases": [
                    {
                        "period": "04/26-03/28",
                        "caps": "food $520, shopping $180 monthly",
                        "description": "Steady food and shopping caps while protecting fixed bills.",
                    },
                ],
            },
        },
    },
    {
        "name": "cash_flow_alternative",
        "batch": 1,
        "input": """
# Financial Needs

## Primary needs
1. **Stabilize cash flow**: Checking **$800** with **$2,100** mortgage due **2026-04-01** — liquidity risk before flexible spend cuts matter.

## Evidence
* **Stabilize cash flow**
  - Checking **$800** vs mortgage **$2,100** on the 1st.
  - Forecast committed outflows **$3,600**/mo vs income **$4,000**/mo.

# Financial Strategy

## Recommended plan: protect_fixed_cut_flex
* Hold checking above **$2,200** before the mortgage, trim **$200**/mo from food and shopping months 1–3, then reassess.

## Alternative plan: aggressive_flex_cut
* Cut food to **$450** and shopping to **$150** immediately — debt-free by **Aug 2026** but checking may dip below **$500** in April.

### Spending Schedule
- 04/26-03/28: Cap food $450, shopping $150 monthly
""",
        "ideal_response": {
            "plan_title": "Aggressive flex cut",
            "plan_summary": "Cut food to $450/mo and shopping to $150/mo immediately to be debt-free by Aug 2026.",
            "plan_details": {
                "payoff": "Debt-free by Aug 2026 — faster than protect_fixed_cut_flex.",
                "trade_off": "Checking may dip below $500 in April while flexible spend drops hard from day one.",
                "spending_phases": [
                    {
                        "period": "04/26-03/28",
                        "caps": "food $450, shopping $150 monthly",
                        "description": "Hard food and shopping caps from day one.",
                    },
                ],
            },
        },
    },
    {
        "name": "slow_debt_recommended",
        "batch": 2,
        "input": """
# Financial Needs

## Primary needs
1. **Settle debt**: Platinum **$4,800** with slow paydown at minimum-style payments.

## Evidence
* **Settle debt**
  - Balance up **$300** over three months despite **$115**/mo payments.
  - APR tool: **~21.8%** on Platinum.

# Financial Strategy

## Recommended plan: balanced_trim
* Trim food to **$520** and leisure to **$300** from month 1; **$0** debt by **Dec 2026**, saves about **$420** interest vs status quo.

## Alternative plan: leisure_first
* Protect leisure at **$380** but cut food harder to **$450** — similar debt-free date with more dining sacrifice and less social spend risk.

### Spending Schedule
- 04/26-03/28: Cap food $520, leisure $300 monthly
""",
        "ideal_response": {
            "plan_title": "Balanced trim",
            "plan_summary": "Trim food to $520/mo and leisure to $300/mo from month one for a Dec 2026 debt-free date.",
            "plan_details": {
                "payoff": "$0 Platinum debt by Dec 2026; saves about $420 interest versus status quo at ~21.8% APR.",
                "trade_off": "Both dining and social spend tighten together from the start — no leisure protection.",
                "spending_phases": [
                    {
                        "period": "04/26-03/28",
                        "caps": "food $520, leisure $300 monthly",
                        "description": "Food and leisure tighten together from month one.",
                    },
                ],
            },
        },
    },
    {
        "name": "slow_debt_alternative",
        "batch": 2,
        "input": """
# Financial Needs

## Primary needs
1. **Settle debt**: Platinum **$4,800** with slow paydown at minimum-style payments.

## Evidence
* **Settle debt**
  - Balance up **$300** over three months despite **$115**/mo payments.
  - APR tool: **~21.8%** on Platinum.

# Financial Strategy

## Recommended plan: balanced_trim
* Trim food to **$520** and leisure to **$300** from month 1; **$0** debt by **Dec 2026**, saves about **$420** interest vs status quo.

## Alternative plan: leisure_first
* Protect leisure at **$380** but cut food harder to **$450** — similar debt-free date with more dining sacrifice and less social spend risk.

### Spending Schedule
- 04/26-03/28: Cap food $450, leisure $380 monthly
""",
        "ideal_response": {
            "plan_title": "Leisure-first trim",
            "plan_summary": "Protect leisure at $380/mo but cut food harder to $450/mo for a similar debt-free date.",
            "plan_details": {
                "payoff": "Similar debt-free timing to balanced_trim on the $4,800 Platinum balance.",
                "trade_off": "More dining sacrifice throughout; leisure stays steadier than the balanced plan.",
                "spending_phases": [
                    {
                        "period": "04/26-03/28",
                        "caps": "food $450, leisure $380 monthly",
                        "description": "Protect leisure; food takes the deeper cut.",
                    },
                ],
            },
        },
    },
    {
        "name": "spending_drift_recommended",
        "batch": 3,
        "simulate_agent_outcome_id": 1252,
        "input": """
# Financial Needs

## Primary needs

1. **Structural Liquidity Trap**: Your finances are built on a high-overhead, low-buffer foundation where nearly your entire monthly income is consumed by static commitments, forcing you into a continuous cycle of credit-card debt to bridge cash-flow gaps.
2. **"Spending Drift" in Discretionary Categories**: Your actual "need" is masked by excessive discretionary spending (dining, entertainment, and delivered food) that consistently exceeds $2,000–$2,500 monthly, effectively cannibalizing the cash meant for savings and debt reduction.
3. **Interest Leakage and Debt Cycle**: You are paying significant credit card interest—upwards of $180+ monthly—due to delayed payments or carrying balances that could be cleared if the discretionary "drift" were curtailed, creating a negative feedback loop where interest fees further restrict your liquidity.

## Evidence

* **Structural Liquidity Trap**
  * Your fixed monthly overhead for shelter alone is $2,850, plus recurring school/daycare ($500), utilities (~$350), and insurance/connectivity (~$200). 
  * These core fixed costs consume ~75% of your standard $6,369 monthly take-home pay, leaving almost no margin for emergencies, which explains the persistent overdrafts on your savings account and the use of credit cards as a primary cash-flow tool.
* **"Spending Drift" in Discretionary Categories**
  * Analysis of your transaction ledger reveals consistent, high-frequency discretionary spending. For example, in June 2026, you spent over $1,200 on "meals_dining_out" and "meals_delivered_food" alone, combined with nearly $600 on "leisure_entertainment."
  * This "drift"—small, frequent transactions at cafes, cinemas, and for food delivery—totaling ~$2,000/month, is the primary driver of your inability to build a cash buffer in your checking or savings accounts.
* **Interest Leakage and Debt Cycle**
  * You are incurring consistent monthly finance charges (e.g., $148.33 and $36.34 on your credit cards in July 2026).
  * Because your checking account balance is often depleted to the "overdraft" or near-zero level by mid-month, you are forced to rely on credit cards for routine purchases. This triggers high interest charges, which further reduces your available income for the following month, maintaining the debt cycle.

# Financial Strategy

## Recommended plan: empathetic_staged_adjustment
* This plan is the most sustainable because it recognizes that a sudden, drastic cut in discretionary spending is often unsustainable and stressful. By allowing for a "step-down" period over six months, you build the discipline to manage your "spending drift" while still successfully clearing your high-interest debt within the same quarter as the more aggressive options. This builds confidence and creates a long-term habit rather than just a temporary fix.

## Alternative plan: rapid_debt_sprint
* This is a strong second choice for those who prioritize immediate math over psychological comfort. It eliminates interest charges the fastest, providing an immediate sense of relief and mathematical efficiency. It is the best choice if your top priority is to kill the 24.99% APR interest cycle as quickly as humanly possible, even if it feels more restrictive in the short term.

### Spending Schedule
- 08/26-10/26: Cap food $1200, leisure $300, shopping $50, health $80, education $450, uncategorized $300 monthly
- 11/26-01/27: Cap food $750, leisure $200, shopping $50, health $80, education $450, uncategorized $300 monthly
- 02/27-: Cap food $500, leisure $100, shopping $50, health $80, education $450, uncategorized $300 monthly
""",
        "ideal_response": {
            "plan_title": "Staged drift reset",
            "plan_summary": "Step down food and leisure over three phases to kill 24.99% APR interest without a shock cut.",
            "plan_details": {
                "payoff": "Clears high-interest revolving balances within the quarter while building sustainable spending habits.",
                "trade_off": "Slower than rapid_debt_sprint, but the six-month step-down is easier to sustain than a shock cut.",
                "spending_phases": [
                    {
                        "period": "08/26-10/26",
                        "caps": "food $1200, leisure $300, shopping $50, health $80, education $450, uncategorized $300 monthly",
                        "description": "Warm-up: modest trim on food and leisure.",
                    },
                    {
                        "period": "11/26-01/27",
                        "caps": "food $750, leisure $200, shopping $50, health $80, education $450, uncategorized $300 monthly",
                        "description": "Step down food and leisure again before the long hold.",
                    },
                    {
                        "period": "02/27-:",
                        "caps": "food $500, leisure $100, shopping $50, health $80, education $450, uncategorized $300 monthly",
                        "description": "Sustain target caps indefinitely from Feb '27.",
                    },
                ],
            },
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


def build_plan_verbalizer_input_bundle(
    *,
    simulate_outcome_md: str,
    goal_plan_scenario: dict[str, Any],
) -> str:
    """Trimmed needs markdown plus ``### Spending Schedule`` for one scenario."""
    simulate = trim_simulate_outcome_for_plan_bundle(simulate_outcome_md)
    scenario_id = str(goal_plan_scenario.get("scenario_id") or "").strip()
    if scenario_id:
        simulate = filter_financial_strategy_for_scenario(
            simulate,
            scenario_id,
            is_active=goal_plan_scenario.get("is_active"),
        )
    simulate = ensure_blank_line_after_plan_headings(simulate)
    plan_block = _format_goal_plan_narrative([goal_plan_scenario])
    parts = [simulate.rstrip()]
    if plan_block:
        parts.append(plan_block.rstrip())
    return "\n\n".join(parts) + "\n"


def build_plan_verbalizer_input(
    *,
    simulate_agent_outcome_id: int | None = None,
    user_id: int | None = None,
    scenario_id: str | None = None,
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

    sid = str(scenario_id or test_case.get("scenario_id") or "").strip()
    if sid:
        bundle = _apply_strategy_filter_to_plan_bundle(bundle, sid)
    return bundle


def format_plan_verbalizer_user_message(profile_input: str) -> str:
    body = (profile_input or "").strip()
    if not body:
        raise ValueError("profile_input must be non-empty markdown.")
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
                return _attach_spending_phases(validated, profile_input)
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
