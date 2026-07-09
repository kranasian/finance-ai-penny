"""
Optimizer runner for **P:PlanVerbalizer** (Gemini prompt tuning).

Same bundled input as ``need_plans_verbalizer_optimizer.py`` — trimmed
``<SIMULATE_FINANCIAL_STRATEGY_OUTCOME>`` plus ``<GOAL_PLAN>`` — except only the
intended scenario is included in ``<GOAL_PLAN>`` (recommended by default, or
``--scenario-id``).

Objective: detailed verbalization of that one plan (phased spending, targets, payoff).

Run from ``finance-ai-penny`` repo root (``finance-ai-penny/.venv`` or ``finance-ai-llm-server/llm``):

  python3 active_experiments/plan_verbalizer_optimizer.py --test 0
  python3 active_experiments/plan_verbalizer_optimizer.py --test all
  python3 active_experiments/plan_verbalizer_optimizer.py --simulate-agent-outcome-id 1148 --print-input-only
  python3 active_experiments/plan_verbalizer_optimizer.py --simulate-agent-outcome-id 1148 --scenario-id gradual_paydown_savings

DB-backed runs read ``SLAVE_DB`` from ``finance-ai-llm-server/config.ini``. Requires ``psycopg2-binary``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
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

from active_experiments.need_plans_verbalizer_optimizer import (
    _bundled_input_from_test_case,
    _fetch_ai_agent_outcome_row,
    _fetch_user_goal_plan,
    _finalize_goal_plan_for_bundle,
    _goal_plan_active_scenario,
    _goal_plan_entries_for_bundle,
    _normalize_goal_plan_list,
    _resolve_ideal_response,
    build_need_plans_verbalizer_input_bundle,
    load_simulate_agent_outcome_markdown,
)

if load_dotenv is not None:
    load_dotenv()

GEMINI_FLASH_LITE = "gemini-flash-lite-latest"
PLAN_VERBALIZER_THINKING_BUDGET = 256
PLAN_VERBALIZER_MAX_OUTPUT_TOKENS = 4096

SYSTEM_PROMPT = """You are Penny — a sharp, witty money coach who explains one financial plan in clear, concrete detail.

Use the plan JSON in the input ``<GOAL_PLAN>`` block. Copy ``scenario_id`` and ``scenario_title`` verbatim. Draw details from that JSON and the matching plan prose in ``<SIMULATE_FINANCIAL_STRATEGY_OUTCOME>``.

- ``mission``: 1–2 sentences — what this plan optimizes for.
- ``blueprint``: one string per spending phase (``spending_schedule`` / ``spending_timeline``) with exact **$**/mo targets; summarize when many categories.
- ``result``: 1–2 sentences — payoff timeline, balance or savings targets, interest impact when stated.
- ``trade_off``: 1 sentence — main sacrifice or risk.

Ground every figure in the input. Fun and confident, never cheesy, patronizing, or naggy. No exclamation marks, superlatives, or emoji.
"""


def _build_output_schema() -> "types.Schema":
    if types is None:  # pragma: no cover
        raise RuntimeError("Install `google-genai` for this optimizer.")
    return types.Schema(
        type=types.Type.OBJECT,
        required=["scenario_id", "scenario_title", "mission", "blueprint", "result", "trade_off"],
        properties={
            "scenario_id": types.Schema(
                type=types.Type.STRING,
                description="Verbatim scenario_id from the input GOAL_PLAN JSON.",
            ),
            "scenario_title": types.Schema(
                type=types.Type.STRING,
                description="Verbatim scenario_title from the input GOAL_PLAN JSON.",
            ),
            "mission": types.Schema(
                type=types.Type.STRING,
                description="1–2 sentences on what this plan optimizes for.",
            ),
            "blueprint": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(
                    type=types.Type.STRING,
                    description="One string per spending phase with exact $/mo targets.",
                ),
            ),
            "result": types.Schema(
                type=types.Type.STRING,
                description="1–2 sentences on payoff timeline, targets, and interest impact when stated.",
            ),
            "trade_off": types.Schema(
                type=types.Type.STRING,
                description="One sentence on the main sacrifice or risk.",
            ),
        },
    )


def _validate_plan_response(
    parsed: Any,
    *,
    expected_scenario_id: str | None = None,
    expected_scenario_title: str | None = None,
) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        raise ValueError("Response must be a JSON object")
    for key in ("scenario_id", "scenario_title", "mission", "result", "trade_off"):
        val = parsed.get(key)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"{key} must be a non-empty string")
    blueprint = parsed.get("blueprint")
    if not isinstance(blueprint, list) or not blueprint:
        raise ValueError("blueprint must be a non-empty array of strings")
    for i, item in enumerate(blueprint):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"blueprint[{i}] must be a non-empty string")
    if expected_scenario_id and parsed["scenario_id"] != expected_scenario_id:
        raise ValueError(
            f"scenario_id mismatch: expected {expected_scenario_id!r}, got {parsed['scenario_id']!r}"
        )
    if expected_scenario_title and parsed["scenario_title"] != expected_scenario_title:
        raise ValueError(
            f"scenario_title mismatch: expected {expected_scenario_title!r}, "
            f"got {parsed['scenario_title']!r}"
        )
    return parsed


TEST_CASES: list[dict[str, Any]] = [
    {
        "name": "debt_paydown_recommended",
        "batch": 1,
        "input": """
<SIMULATE_FINANCIAL_STRATEGY_OUTCOME>

# Financial Needs

## Primary needs
1. **Reduce interest drag**: Venture balance **$8,400** with **$312** interest paid over 90 days while spending tracks near income.

## Evidence
* **Reduce interest drag**
  - Interest tool: **$312** on Venture in 90 days.
  - Next due **2026-04-18** per payment schedule.

## Credit Interest Rates
* **Venture (11)**: **~24.9%** (from recent interest charges vs average daily balance).

# Financial Strategy

## Recommended plan: gradual_paydown_savings
* Phased dining and leisure trims keep month-1 cuts modest, then deepen after month 3 while routing **$200**/mo to savings once the card hits **$0**.

## Alternative plan: steady_cut
* Flat **$700** meals and **$350** leisure from month 1 hits **$0** debt about two months sooner but leaves thinner checking buffers in the first quarter.


</SIMULATE_FINANCIAL_STRATEGY_OUTCOME>

<GOAL_PLAN>

```json
[
  {
    "scenario_id": "gradual_paydown_savings",
    "scenario_title": "Gradual Paydown Savings",
    "is_active": true,
    "current_spending": {
      "meals": 974,
      "leisure": 520
    },
    "spending_schedule": [
      {
        "start_end_month": "1-3",
        "categories": {
          "meals": 850,
          "leisure": 450
        }
      },
      {
        "start_end_month": "4+",
        "categories": {
          "meals": 700,
          "leisure": 350
        }
      }
    ],
    "spending_timeline": [
      {
        "categories": {
          "meals": 850,
          "leisure": 450
        },
        "start_month": "04/26",
        "end_month": "06/26"
      },
      {
        "categories": {
          "meals": 700,
          "leisure": 350
        },
        "start_month": "07/26",
        "end_month": "03/28"
      }
    ],
    "credit_balance_target": 0,
    "savings_per_month": 200,
    "savings_targets": [
      6500
    ]
  }
]
```

</GOAL_PLAN>
""",
        "ideal_response": {
            "scenario_id": "gradual_paydown_savings",
            "scenario_title": "Gradual Paydown Savings",
            "mission": "Phase dining and leisure cuts, then route $200/mo to savings after Venture hits $0.",
            "blueprint": [
                "Months 1-3: meals $850/mo, leisure $450/mo.",
                "Month 4+: meals $700/mo, leisure $350/mo."
            ],
            "result": "Card balance targets $0 with $200/mo savings once paid off; $6,500 savings goal.",
            "trade_off": "Debt clears slower than the flat-cut plan while buffers stay thicker early on."
        },
    },
    {
        "name": "debt_paydown_alternative",
        "batch": 1,
        "input": """
<SIMULATE_FINANCIAL_STRATEGY_OUTCOME>

# Financial Needs

## Primary needs
1. **Reduce interest drag**: Venture balance **$8,400** with **$312** interest paid over 90 days while spending tracks near income.

## Evidence
* **Reduce interest drag**
  - Interest tool: **$312** on Venture in 90 days.
  - Next due **2026-04-18** per payment schedule.

## Credit Interest Rates
* **Venture (11)**: **~24.9%** (from recent interest charges vs average daily balance).

# Financial Strategy

## Recommended plan: gradual_paydown_savings
* Phased dining and leisure trims keep month-1 cuts modest, then deepen after month 3 while routing **$200**/mo to savings once the card hits **$0**.

## Alternative plan: steady_cut
* Flat **$700** meals and **$350** leisure from month 1 hits **$0** debt about two months sooner but leaves thinner checking buffers in the first quarter.


</SIMULATE_FINANCIAL_STRATEGY_OUTCOME>

<GOAL_PLAN>

```json
[
  {
    "scenario_id": "steady_cut",
    "scenario_title": "Steady Cut",
    "is_active": false,
    "current_spending": {
      "meals": 974,
      "leisure": 520
    },
    "spending_schedule": [
      {
        "start_end_month": "1+",
        "categories": {
          "meals": 700,
          "leisure": 350
        }
      }
    ],
    "spending_timeline": [
      {
        "categories": {
          "meals": 700,
          "leisure": 350
        },
        "start_month": "04/26",
        "end_month": "03/28"
      }
    ],
    "credit_balance_target": 0
  }
]
```

</GOAL_PLAN>
""",
        "ideal_response": {
            "scenario_id": "steady_cut",
            "scenario_title": "Steady Cut",
            "mission": "Hold meals at $700 and leisure at $350 from month 1 to clear Venture debt faster.",
            "blueprint": [
                "Month 1+: meals $700/mo, leisure $350/mo."
            ],
            "result": "Reaches $0 card balance about two months sooner than the gradual plan.",
            "trade_off": "Thinner checking buffers in the first quarter while cuts stay flat."
        },
    },
    {
        "name": "cash_flow_recommended",
        "batch": 1,
        "input": """
<SIMULATE_FINANCIAL_STRATEGY_OUTCOME>

# Financial Needs

## Primary needs
1. **Stabilize cash flow**: Checking **$800** with **$2,100** mortgage due **2026-04-01** — liquidity risk before flexible spend cuts matter.

## Evidence
* **Stabilize cash flow**
  - Checking **$800** vs mortgage **$2,100** on the 1st.
  - Forecast committed outflows **$3,600**/mo vs income **$4,000**/mo.

## Credit Interest Rates
* **Rewards (21)**: **~22.4%** (derived from last-cycle interest charge).

# Financial Strategy

## Recommended plan: protect_fixed_cut_flex
* Hold checking above **$2,200** before the mortgage, trim **$200**/mo from meals and shopping months 1–3, then reassess.

## Alternative plan: aggressive_flex_cut
* Cut meals to **$450** and shopping to **$150** immediately — debt-free by **Aug 2026** but checking may dip below **$500** in April.


</SIMULATE_FINANCIAL_STRATEGY_OUTCOME>

<GOAL_PLAN>

```json
[
  {
    "scenario_id": "protect_fixed_cut_flex",
    "scenario_title": "Protect Fixed Cut Flex",
    "is_active": true,
    "current_spending": {
      "meals": 620,
      "shopping": 280
    },
    "spending_schedule": [
      {
        "start_end_month": "1-3",
        "categories": {
          "meals": 520,
          "shopping": 180
        }
      },
      {
        "start_end_month": "4+",
        "categories": {
          "meals": 520,
          "shopping": 180
        }
      }
    ],
    "spending_timeline": [
      {
        "categories": {
          "meals": 520,
          "shopping": 180
        },
        "start_month": "04/26",
        "end_month": "06/26"
      },
      {
        "categories": {
          "meals": 520,
          "shopping": 180
        },
        "start_month": "07/26",
        "end_month": "03/28"
      }
    ],
    "account_balance_targets": [
      {
        "account_id": 20,
        "balance_target": 2200
      }
    ]
  }
]
```

</GOAL_PLAN>
""",
        "ideal_response": {
            "scenario_id": "protect_fixed_cut_flex",
            "scenario_title": "Protect Fixed Cut Flex",
            "mission": "Keep checking above $2,200 before the mortgage while trimming flexible spend.",
            "blueprint": [
                "Months 1-3: meals $520/mo, shopping $180/mo.",
                "Month 4+: meals $520/mo, shopping $180/mo."
            ],
            "result": "Targets checking balance at $2,200 before the $2,100 mortgage on April 1.",
            "trade_off": "Slower debt payoff than the aggressive flex cut if card balances remain."
        },
    },
    {
        "name": "cash_flow_alternative",
        "batch": 1,
        "input": """
<SIMULATE_FINANCIAL_STRATEGY_OUTCOME>

# Financial Needs

## Primary needs
1. **Stabilize cash flow**: Checking **$800** with **$2,100** mortgage due **2026-04-01** — liquidity risk before flexible spend cuts matter.

## Evidence
* **Stabilize cash flow**
  - Checking **$800** vs mortgage **$2,100** on the 1st.
  - Forecast committed outflows **$3,600**/mo vs income **$4,000**/mo.

## Credit Interest Rates
* **Rewards (21)**: **~22.4%** (derived from last-cycle interest charge).

# Financial Strategy

## Recommended plan: protect_fixed_cut_flex
* Hold checking above **$2,200** before the mortgage, trim **$200**/mo from meals and shopping months 1–3, then reassess.

## Alternative plan: aggressive_flex_cut
* Cut meals to **$450** and shopping to **$150** immediately — debt-free by **Aug 2026** but checking may dip below **$500** in April.


</SIMULATE_FINANCIAL_STRATEGY_OUTCOME>

<GOAL_PLAN>

```json
[
  {
    "scenario_id": "aggressive_flex_cut",
    "scenario_title": "Aggressive Flex Cut",
    "is_active": false,
    "current_spending": {
      "meals": 620,
      "shopping": 280
    },
    "spending_schedule": [
      {
        "start_end_month": "1+",
        "categories": {
          "meals": 450,
          "shopping": 150
        }
      }
    ],
    "spending_timeline": [
      {
        "categories": {
          "meals": 450,
          "shopping": 150
        },
        "start_month": "04/26",
        "end_month": "03/28"
      }
    ],
    "credit_balance_target": 0
  }
]
```

</GOAL_PLAN>
""",
        "ideal_response": {
            "scenario_id": "aggressive_flex_cut",
            "scenario_title": "Aggressive Flex Cut",
            "mission": "Cut meals and shopping immediately to stabilize cash before the mortgage hits.",
            "blueprint": [
                "Month 1+: meals $450/mo, shopping $150/mo."
            ],
            "result": "Debt-free by Aug 2026 but checking may dip below $500 in April.",
            "trade_off": "Higher liquidity risk in April while spending cuts stay steep from day one."
        },
    },
    {
        "name": "slow_debt_recommended",
        "batch": 2,
        "input": """
<SIMULATE_FINANCIAL_STRATEGY_OUTCOME>

# Financial Needs

## Primary needs
1. **Settle debt**: Platinum **$4,800** with slow paydown at minimum-style payments.

## Evidence
* **Settle debt**
  - Balance up **$300** over three months despite **$115**/mo payments.
  - APR tool: **~21.8%** on Platinum.

## Credit Interest Rates
* **Platinum (31)**: **~21.8%**

# Financial Strategy

## Recommended plan: balanced_trim
* Trim meals to **$520** and leisure to **$300** from month 1; **$0** debt by **Dec 2026**, saves about **$420** interest vs status quo.

## Alternative plan: leisure_first
* Protect leisure at **$380** but cut meals harder to **$450** — similar debt-free date with more dining sacrifice and less social spend risk.


</SIMULATE_FINANCIAL_STRATEGY_OUTCOME>

<GOAL_PLAN>

```json
[
  {
    "scenario_id": "balanced_trim",
    "scenario_title": "Balanced Trim",
    "is_active": true,
    "current_spending": {
      "meals": 640,
      "leisure": 410
    },
    "spending_schedule": [
      {
        "start_end_month": "1+",
        "categories": {
          "meals": 520,
          "leisure": 300
        }
      }
    ],
    "spending_timeline": [
      {
        "categories": {
          "meals": 520,
          "leisure": 300
        },
        "start_month": "04/26",
        "end_month": "03/28"
      }
    ],
    "credit_balance_target": 0
  }
]
```

</GOAL_PLAN>
""",
        "ideal_response": {
            "scenario_id": "balanced_trim",
            "scenario_title": "Balanced Trim",
            "mission": "Trim meals and leisure evenly to settle Platinum debt by Dec 2026.",
            "blueprint": [
                "Month 1+: meals $520/mo, leisure $300/mo."
            ],
            "result": "Hits $0 debt by Dec 2026 and saves about $420 interest versus status quo.",
            "trade_off": "Leisure drops more than in the leisure-first alternative."
        },
    },
    {
        "name": "slow_debt_alternative",
        "batch": 2,
        "input": """
<SIMULATE_FINANCIAL_STRATEGY_OUTCOME>

# Financial Needs

## Primary needs
1. **Settle debt**: Platinum **$4,800** with slow paydown at minimum-style payments.

## Evidence
* **Settle debt**
  - Balance up **$300** over three months despite **$115**/mo payments.
  - APR tool: **~21.8%** on Platinum.

## Credit Interest Rates
* **Platinum (31)**: **~21.8%**

# Financial Strategy

## Recommended plan: balanced_trim
* Trim meals to **$520** and leisure to **$300** from month 1; **$0** debt by **Dec 2026**, saves about **$420** interest vs status quo.

## Alternative plan: leisure_first
* Protect leisure at **$380** but cut meals harder to **$450** — similar debt-free date with more dining sacrifice and less social spend risk.


</SIMULATE_FINANCIAL_STRATEGY_OUTCOME>

<GOAL_PLAN>

```json
[
  {
    "scenario_id": "leisure_first",
    "scenario_title": "Leisure First",
    "is_active": false,
    "current_spending": {
      "meals": 640,
      "leisure": 410
    },
    "spending_schedule": [
      {
        "start_end_month": "1+",
        "categories": {
          "meals": 450,
          "leisure": 380
        }
      }
    ],
    "spending_timeline": [
      {
        "categories": {
          "meals": 450,
          "leisure": 380
        },
        "start_month": "04/26",
        "end_month": "03/28"
      }
    ],
    "credit_balance_target": 0
  }
]
```

</GOAL_PLAN>
""",
        "ideal_response": {
            "scenario_id": "leisure_first",
            "scenario_title": "Leisure First",
            "mission": "Protect leisure spending while cutting meals harder to settle Platinum debt.",
            "blueprint": [
                "Month 1+: meals $450/mo, leisure $380/mo."
            ],
            "result": "Similar debt-free timing with more dining sacrifice and steadier social spend.",
            "trade_off": "Meals stay tighter than the balanced trim plan throughout."
        },
    },
    {
        "name": "spending_drift_recommended",
        "batch": 3,
        "simulate_agent_outcome_id": 1252,
        "input": """
<SIMULATE_FINANCIAL_STRATEGY_OUTCOME>

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

## Credit Interest Rates



* **Sapphire Preferred (ID: 12827)**: **~24.99% APR**
  * **Derivation Method:** Based on historical interest charges, such as the $256.13 charge on a balance that was hovering around $12,000–$12,500 in May/June, and the $148.33 charge in July on a balance that was recently reduced significantly, the implied monthly interest rate consistently aligns with a ~2.08% monthly factor. This mathematically maps to an Annual Percentage Rate (APR) of approximately 24.99%.

* **Cash Magnet (ID: 12828)**: **~24.99% APR**
  * **Derivation Method:** Analysis of consistent monthly interest charges (e.g., $36.34 in July, $38.49 in June) against average daily balances for this account reveals a similar interest trajectory. The charges remain highly proportional to a ~2.08% monthly interest rate, indicating the same ~24.99% APR as the Sapphire Preferred account. Note: Fluctuations in these specific amounts are likely due to balance timing/payment dates rather than rate variance.

## Immediate Things to Do



1. **Strategic Cash-Flow Buffer Synchronization**
   * **The Move:** Shift your recurring bill payment dates—specifically those for utilities (Dominion, Mid-Carolina) and connectivity—to align with the 1st and 15th of the month, immediately following your two major payroll deposits. Use your high-balance periods (e.g., the 1st) to sweep a fixed "Debt & Fixed-Cost Pool" into a sub-ledger or separate account specifically for these payments.
   * **The Strategic Why:** Currently, your rent ($2,850) and fixed costs hit while your account is already strained, leading to overdrafts and forcing you to use high-interest credit cards for basic necessities. By anchoring payments to your income arrival, you eliminate the "timing gap" that currently forces you into the 24.99% APR debt cycle.

2. **Automated "Friction" for Discretionary Spending**
   * **The Move:** In Penny, set an aggressive monthly limit of $500 for the combined categories of `meals_dining_out`, `meals_delivered_food`, and `leisure_entertainment`. Configure your alerts to trigger a high-friction notification the moment you hit 50% ($250) of this allocation.
   * **The Strategic Why:** Your ledger shows you are "drifting" over $2,000/month in these categories. This creates an immediate structural deficit of ~30% of your net income. Reducing this "drift" by even 50% provides an extra $1,000/month of liquid cash that can be diverted to pay down your $1,600+ combined credit card balance, effectively ending the ~$180/month interest leakage.

3. **Targeted Debt "Snowball" via Interest Elimination**
   * **The Move:** Over the next 90 days, commit the $1,000+ "recovered" cash from your discretionary spending drift to pay down your credit card balances with the highest interest impact. Execute payments immediately upon receipt of your Genentech and CA State paychecks, rather than waiting for the statement due dates.
   * **The Strategic Why:** Your 24.99% APR accounts are essentially an emergency tax on your lack of liquidity. By paying down the balances early in your monthly cycle, you lower your Average Daily Balance (ADB), which directly reduces the dollar amount of interest calculated at the end of the month, instantly improving your net cash flow.

## Next Set of Milestones to Aspire



1. **The "Zero-Interest" Debt Freedom — Target Timeline: 6 Months**
   * **The Metric:** Reduce revolving credit balances on cards 12827 and 12828 to $0 balance by the statement closing date, effectively eliminating all ~$180+/month finance charges.
   * **The Long-Term Path:** This requires transitioning from a "reactive" credit payer (paying bills *as* you spend) to a "proactive" manager. Once the high-interest debt is zeroed, the money previously lost to interest must be automatically diverted to a "Core Reserve" account. This permanently kills the debt cycle and increases your disposable monthly cash flow by 3% of your total income without increasing your earnings.

2. **The "60-Day Runway" Liquidity Baseline — Target Timeline: 12 Months**
   * **The Metric:** Maintain a minimum cash balance of $15,000 in your Checking/Savings accounts that never drops below this floor, covering roughly 60 days of your $7,500 average monthly outflow.
   * **The Long-Term Path:** Establish a "savings gate" where 15% of every paycheck is automatically moved to a high-yield savings vehicle before discretionary spending begins. This habit removes the structural dependency on credit cards for routine expenses and transforms you from a "debt-borrower" into a "cash-buyer," providing the psychological buffer needed to cease the "spending drift" identified in Task 1.

3. **Strategic Wealth-Building Ratio (The 20% Net Allocation) — Target Timeline: 3+ Years**
   * **The Metric:** Achieve a consistent 20% monthly savings/investment rate of your gross take-home income, verified by a monthly surplus in your Penny analytics report.
   * **The Long-Term Path:** By year three, the interest and overhead leakages are gone, and your liquid buffer is secure. You must move from "defensive financial management" to "offensive wealth building." This is maintained by institutionalizing your budget: treating your investments as a non-negotiable "bill" that must be paid on the 1st of the month. This habit compounds wealth and shields you from lifestyle inflation, ensuring that as your career earnings grow, your surplus grows proportionally rather than being consumed by increased discretionary drift.
# Financial Strategy

## Recommended plan: empathetic_staged_adjustment
* This plan is the most sustainable because it recognizes that a sudden, drastic cut in discretionary spending is often unsustainable and stressful. By allowing for a "step-down" period over six months, you build the discipline to manage your "spending drift" while still successfully clearing your high-interest debt within the same quarter as the more aggressive options. This builds confidence and creates a long-term habit rather than just a temporary fix.

## Alternative plan: rapid_debt_sprint
* This is a strong second choice for those who prioritize immediate math over psychological comfort. It eliminates interest charges the fastest, providing an immediate sense of relief and mathematical efficiency. It is the best choice if your top priority is to kill the 24.99% APR interest cycle as quickly as humanly possible, even if it feels more restrictive in the short term.


</SIMULATE_FINANCIAL_STRATEGY_OUTCOME>

<GOAL_PLAN>

```json
[
  {
    "scenario_id": "empathetic_staged_adjustment",
    "scenario_title": "Empathetic Staged Adjustment",
    "is_active": true,
    "spending_schedule": [
      {
        "categories": {
          "meals": 1200,
          "health": 80,
          "leisure": 300,
          "shopping": 50,
          "education": 450,
          "uncategorized": 300
        },
        "start_end_month": "1-3"
      },
      {
        "categories": {
          "meals": 750,
          "health": 80,
          "leisure": 200,
          "shopping": 50,
          "education": 450,
          "uncategorized": 300
        },
        "start_end_month": "4-6"
      },
      {
        "categories": {
          "meals": 500,
          "health": 80,
          "leisure": 100,
          "shopping": 50,
          "education": 450,
          "uncategorized": 300
        },
        "start_end_month": "7+"
      }
    ]
  }
]
```

</GOAL_PLAN>
""",
        "ideal_response": {
            "scenario_id": "empathetic_staged_adjustment",
            "scenario_title": "Empathetic Staged Adjustment",
            "mission": "Step down discretionary drift over six months while clearing high-interest card debt.",
            "blueprint": [
                "Months 1-3: meals $1,200/mo, leisure $300/mo, plus fixed education and health lines.",
                "Months 4-6: meals $750/mo, leisure $200/mo with other categories held steady.",
                "Month 7+: meals $500/mo, leisure $100/mo to kill the 24.99% APR cycle."
            ],
            "result": "Clears revolving balances within the quarter while building sustainable spending habits.",
            "trade_off": "Slower than a rapid debt sprint while discretionary cuts ramp gradually."
        },
    },
]






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
    """Same bundle shape as need-plans, with only the intended scenario in ``<GOAL_PLAN>``."""
    entries = _goal_plan_entries_for_bundle([goal_plan_scenario])
    if not entries:
        raise ValueError("goal_plan_scenario must be non-empty")
    return build_need_plans_verbalizer_input_bundle(
        simulate_outcome_md=simulate_outcome_md,
        goal_plan=entries,
    )


def build_plan_verbalizer_input(
    *,
    simulate_agent_outcome_id: int,
    scenario_id: str | None = None,
) -> str:
    """Build bundled input from DB outcomes + one ``users.goal_plan`` scenario."""
    sim_uid, simulate_md = load_simulate_agent_outcome_markdown(simulate_agent_outcome_id)
    sim_row = _fetch_ai_agent_outcome_row(int(simulate_agent_outcome_id))
    if not sim_row:
        raise ValueError(f"simulate_agent_outcome_id not found: {simulate_agent_outcome_id}")

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


def _goal_plan_from_bundled_input(bundled: str) -> list[dict[str, Any]]:
    marker = "<GOAL_PLAN>"
    if marker not in bundled:
        raise ValueError("bundled input missing <GOAL_PLAN>")
    block = bundled.split(marker, 1)[1].split("</GOAL_PLAN>", 1)[0]
    fence = re.search(r"```json\s*(\[.*?\])\s*```", block, re.DOTALL)
    if not fence:
        raise ValueError("bundled <GOAL_PLAN> missing JSON array")
    parsed = json.loads(fence.group(1))
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("bundled <GOAL_PLAN> must be a non-empty JSON array")
    return [entry for entry in parsed if isinstance(entry, dict)]


def _simulate_md_from_bundled_input(bundled: str) -> str:
    simulate_block = bundled.split("<SIMULATE_FINANCIAL_STRATEGY_OUTCOME>", 1)[1]
    return simulate_block.split("</SIMULATE_FINANCIAL_STRATEGY_OUTCOME>", 1)[0].strip() + "\n"


def resolve_plan_test_case_input(
    test_case: dict[str, Any],
    *,
    scenario_id: str | None = None,
) -> str:
    """Return single-scenario bundle from a test-case dict."""
    bundled = _bundled_input_from_test_case(test_case)
    if not bundled:
        raise ValueError("test case must include bundled input")
    sid = scenario_id or test_case.get("scenario_id")
    if not sid:
        return bundled
    goal_plan = _goal_plan_from_bundled_input(bundled)
    scenario = _select_goal_plan_scenario(goal_plan, scenario_id=sid)
    return build_plan_verbalizer_input_bundle(
        simulate_outcome_md=_simulate_md_from_bundled_input(bundled),
        goal_plan_scenario=scenario,
    )


def format_plan_verbalizer_user_message(profile_input: str) -> str:
    body = (profile_input or "").strip()
    if not body:
        raise ValueError("profile_input must be non-empty markdown.")
    return body + "\n"


def _expected_scenario_fields_from_bundle(bundle_md: str) -> tuple[str, str]:
    marker = "<GOAL_PLAN>"
    if marker not in bundle_md:
        return "", ""
    block = bundle_md.split(marker, 1)[1]
    fence = re.search(r"```json\s*(\[.*?\])\s*```", block, re.DOTALL)
    if not fence:
        return "", ""
    try:
        parsed = json.loads(fence.group(1))
    except json.JSONDecodeError:
        return "", ""
    if not isinstance(parsed, list) or not parsed:
        return "", ""
    first = parsed[0]
    if not isinstance(first, dict):
        return "", ""
    return (
        str(first.get("scenario_id") or "").strip(),
        str(first.get("scenario_title") or "").strip(),
    )


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
                thought_summary = (thought_summary + t) if thought_summary else t
            else:
                output_text += t
    if not output_text:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*non-text parts in the response.*")
            agg = getattr(response, "text", None)
            if isinstance(agg, str) and agg:
                output_text = agg
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

    def generate_response(self, profile_input: str) -> dict[str, Any]:
        user_text = format_plan_verbalizer_user_message(profile_input)
        request_text = types.Part.from_text(text=user_text)
        contents = [types.Content(role="user", parts=[request_text])]
        cfg = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_output_tokens=self.max_output_tokens,
            safety_settings=self.safety_settings,
            system_instruction=[types.Part.from_text(text=self.system_prompt)],
            thinking_config=types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
                include_thoughts=True,
            ),
            response_mime_type="application/json",
            response_schema=self.output_schema,
        )

        output_text = ""
        thought_summary = ""
        finish_reason = None
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=cfg,
            )
            output_text, thought_summary, finish_reason = _collect_model_response(response)
        except ClientError as e:
            if self.thinking_budget == 0 and "only works in thinking mode" in (str(e) or ""):
                print(
                    "\n[NOTE] This model requires thinking mode; use default (no --no-thinking) or a different model.",
                    flush=True,
                )
                sys.exit(1)
            raise

        if thought_summary:
            print("\n" + "-" * 80)
            print("THOUGHT SUMMARY:")
            print("-" * 80)
            print(thought_summary.strip())
            print("-" * 80 + "\n")

        if not (output_text or "").strip():
            hint = " Model returned thoughts only." if thought_summary else ""
            raise ValueError(f"Empty response from model.{hint} Check API key and model availability.")
        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError as exc:
            reason = str(finish_reason or "unknown")
            raise ValueError(f"Invalid JSON response. finish_reason={reason!r}") from exc

        expected_id, expected_title = _expected_scenario_fields_from_bundle(user_text)
        return _validate_plan_response(
            parsed,
            expected_scenario_id=expected_id or None,
            expected_scenario_title=expected_title or None,
        )


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

    if args.simulate_agent_outcome_id is not None:
        built = build_plan_verbalizer_input(
            simulate_agent_outcome_id=args.simulate_agent_outcome_id,
            scenario_id=args.scenario_id,
        )
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
        print("Error: --print-input-only requires --simulate-agent-outcome-id", file=sys.stderr)
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
    print("  Build input from DB: --simulate-agent-outcome-id <id> [--scenario-id <id>]")
    print("  Print built input only: --simulate-agent-outcome-id <id> --print-input-only")
    print("\nAvailable test cases:")
    for i, tc in enumerate(TEST_CASES):
        print(f"  [{i}] {tc['name']} (batch {tc.get('batch', '?')})")


if __name__ == "__main__":
    main()
