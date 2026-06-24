"""
Simulate-financial-strategy rubric optimizer — **quality** (holistic).

Grades ``Hr:SimulateFinancialStrategy`` checker templates.

Run from ``finance-ai-penny`` repo root:

  python3 active_experiments/hr_simulate_financial_strategy_optimizer.py --test all
  python3 active_experiments/hr_simulate_financial_strategy_optimizer.py --test all --check
  python3 active_experiments/hr_simulate_financial_strategy_optimizer.py --batch 1 --check
  python3 active_experiments/hr_simulate_financial_strategy_optimizer.py --model gemini-flash-lite-latest
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

try:
  from dotenv import load_dotenv
except Exception:  # pragma: no cover
  load_dotenv = None

try:
  from google import genai  # type: ignore[import-not-found]
  from google.genai import types  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
  genai = None
  types = None

if load_dotenv is not None:
  load_dotenv()

_TEST_SEPARATOR = "=" * 72
_SECTION_RULE = "-" * 72


def _print_section_banner(title: str) -> None:
  print(f"\n{_SECTION_RULE}\n{title}\n{_SECTION_RULE}\n")


def _parse_expected_output(raw: str | None) -> dict[str, Any] | None:
  if not raw:
    return None
  try:
    return json.loads(raw)
  except Exception:
    return None


def _build_output_schema(_types: Any) -> Any:
  return _types.Schema(
    type=_types.Type.OBJECT,
    required=["score", "notes"],
    properties={
      "score": _types.Schema(type=_types.Type.INTEGER, description="Integer 1–5."),
      "notes": _types.Schema(type=_types.Type.STRING, description="One short sentence."),
    },
  )


SYSTEM_PROMPT = """Grade **simulate_financial_strategy** outcomes for holistic **quality**. Return JSON `{score, notes}` (integer 1–5; one sentence `notes`).

**Bundle:** `<TASK_AND_NEEDS>` … `<STRATEGY>` … `<SIMULATION_CALLS>` … optional `<USER_PREFERENCES>`.

- **`<TASK_AND_NEEDS>`** — `# Task Input`, category spending, `# Financial Needs`.
- **`<STRATEGY>`** — `# Financial Strategy` (`## Recommended plan`, `## Goals and milestone fit`, `## Immediate next steps`).
- **`<SIMULATION_CALLS>`** — `simulate_financial_strategy` tool inputs/outputs (`# Round N`, `## Invoked tools`).
- **`<USER_PREFERENCES>`** (optional) — stated tradeoffs; winning plan should honor or explain tradeoff.

Grade **only** visible text. **Tool output wins** over narrative when they disagree.

**Workflow (from `<SIMULATION_CALLS>`):**
- **≥3** distinct `scenario_id` simulations in **one** round, including **`status_quo`** plus **≥2** improvement plans.
- **≥1** improvement plan should use **phased** `spending_schedule` (2+ entries: smaller cuts first, deeper later).
- Final `<STRATEGY>` should recommend a **cash-flow feasible** scenario when one exists.

**Depository vs savings (critical):**
- **Depository** (Timeline) = checking + savings combined — liquidity only.
- **Savings milestone** achieved date/balance comes **only** from tool `## Result` `achieved` line for the savings account — **never** infer from Timeline depository $.

**Parent categories:** Under **Parent category changes**, expect **only changed** discretionary parents (`slug: $baseline→$new`, with phased month ranges when applicable). **Do not** penalize omitting unchanged parents at baseline $/mo — tool `spending_schedule` passes overrides only. **Do** penalize listing unchanged parents as if they were cuts, leaf-level slugs, or parent $ that contradict the winning tool call.

**Milestones:** Report only **Configured milestone** types from context; status/dates **only** from `## Result` for the recommended `scenario_id`.

**Scoring (worst gap wins):**
- **1** — Invented milestone date/status; depository cited as savings goal met; <3 simulations; recommends infeasible plan as best without disclosure; leaf-level cuts presented as simulation parents.
- **2** — Multiple issues (no phased plan among improvements, narrative contradicts `## Result`, missing `status_quo`).
- **3** — Single clear gap (one wrong parent budget description, preference ignored without tradeoff note, thin milestone section).
- **4** — Minor wording issue while workflow, feasibility, and tool alignment are sound.
- **5** — ≥3 sims incl. status_quo, phased improvement, feasible recommendation, milestones/dates match `## Result`, parent budgets correct, preferences honored.

Return **only** the JSON object (`score`, `notes`).
"""


TEST_CASES: list[dict[str, Any]] = [
  {
    "name": "good_feasible_phased_plan",
    "batch": 1,
    "input": """<TASK_AND_NEEDS>

# Task Input

## Category spending
- Parent `meals` baseline: $974/mo; `leisure` $520/mo

# Financial Needs

## Primary needs
1. **Reduce interest drag** — Pay Venture to **$0**.
2. **Build emergency fund** — **$6,500** savings milestone.

</TASK_AND_NEEDS>

<STRATEGY>

# Financial Strategy

## Recommended plan
- **Scenario id:** `gradual_paydown_savings`
- **Parent category changes:** `meals: $974→$850 (−$124/mo)` mo 1–3, then `$700 (−$274/mo)` mo 4+; `leisure: $520→$450` mo 1–3, then `$350` mo 4+.
- **Payment / savings:** `credit_balance_target: 0`, `savings_targets: [6500]`, `savings_per_month: 200`.
- **Milestones:** Interest-free **Jul 2026**; **$6,500 savings** achieved **Mar 2029** ($6,600 savings account per Result).

## Goals and milestone fit
- **Interest-free credit**: **Achieved** (Jul 2026).
- **Emergency savings fund ($6,500)**: **Achieved** (Mar 2029).

## Immediate next steps
1. Start phased dining cuts in month 1.
2. Enable **$200**/mo savings transfer after card hits **$0**.

</STRATEGY>

<SIMULATION_CALLS>

# Round 1

## Invoked tools

1. **`simulate_financial_strategy`** {"scenario_id":"status_quo"}

2. **`simulate_financial_strategy`** {"scenario_id":"steady_cut","spending_schedule":[{"start_end_month":"1+","categories":{"meals":700,"leisure":350}}],"credit_balance_target":0,"savings_targets":[6500],"savings_per_month":200}

3. **`simulate_financial_strategy`** {"scenario_id":"gradual_paydown_savings","spending_schedule":[{"start_end_month":"1-3","categories":{"meals":850,"leisure":450}},{"start_end_month":"4+","categories":{"meals":700,"leisure":350}}],"credit_balance_target":0,"savings_targets":[6500],"savings_per_month":200}

# Simulation: gradual_paydown_savings

## Result
Cash-flow feasible: yes
- Interest-free credit: achieved By Jul 2026 (mo 4)
- Emergency savings fund ($6,500): achieved By Mar 2029 (mo 40) ($6,600 savings)

## Timeline
- **By Jul 2026** (mo 4): depository $4,100, credit $0

</SIMULATION_CALLS>
""",
    "output": '{"score": 5, "notes": "Three scenarios including phased winner; milestones and dates match ## Result; parent budgets described correctly."}',
  },
  {
    "name": "bad_depository_savings_conflation",
    "batch": 1,
    "input": """<TASK_AND_NEEDS>

# Financial Needs

## Primary needs
1. **Build emergency fund** — **$6,500** savings target.

</TASK_AND_NEEDS>

<STRATEGY>

# Financial Strategy

## Recommended plan
- **Scenario id:** `moderate_cut`
- **Milestones:** **$6,500 savings buffer achieved Oct 2026** because depository reached **$6,094**.

## Goals and milestone fit
- **Emergency savings fund ($6,500)**: **Achieved** (Oct 2026).

</STRATEGY>

<SIMULATION_CALLS>

# Round 1

## Invoked tools

1. **`simulate_financial_strategy`** {"scenario_id":"status_quo"}
2. **`simulate_financial_strategy`** {"scenario_id":"moderate_cut","spending_schedule":[{"start_end_month":"1+","categories":{"meals":800}}],"savings_targets":[6500]}

# Simulation: moderate_cut

## Result
Cash-flow feasible: yes
- Emergency savings fund ($6,500): not achieved
- Interest-free credit: achieved By Aug 2026

## Timeline
- **By Oct 2026** (mo 6): depository $6,094, credit $0

</SIMULATION_CALLS>
""",
    "output": '{"score": 1, "notes": "Claims savings milestone from depository $6,094 while ## Result says not achieved."}',
  },
  {
    "name": "bad_only_two_scenarios",
    "batch": 1,
    "input": """<TASK_AND_NEEDS>

# Financial Needs

## Primary needs
1. **Settle debt**

</TASK_AND_NEEDS>

<STRATEGY>

# Financial Strategy

## Recommended plan
- **Scenario id:** `aggressive_cut`

## Goals and milestone fit
- **Interest-free credit**: **Achieved** (Jun 2026).

</STRATEGY>

<SIMULATION_CALLS>

# Round 1

## Invoked tools

1. **`simulate_financial_strategy`** {"scenario_id":"status_quo"}
2. **`simulate_financial_strategy`** {"scenario_id":"aggressive_cut","spending_schedule":[{"start_end_month":"1+","categories":{"meals":600}}],"credit_balance_target":0}

# Simulation: aggressive_cut

## Result
Cash-flow feasible: yes
- Interest-free credit: achieved By Jun 2026

</SIMULATION_CALLS>
""",
    "output": '{"score": 2, "notes": "Only two scenarios and no phased spending_schedule among improvements."}',
  },
  {
    "name": "good_honors_user_preferences",
    "batch": 2,
    "input": """<TASK_AND_NEEDS>

# Financial Needs

## Primary needs
1. **Save money** — Build **$6,500** emergency fund.

</TASK_AND_NEEDS>

<STRATEGY>

# Financial Strategy

## Recommended plan
- **Scenario id:** `protect_leisure_cut_meals`
- **Your preferences:** Cuts focus on `meals` while `leisure` stays at baseline **$520**/mo.
- **Parent category changes:** `meals: $974→$750 (−$224/mo)`; `leisure: $520/mo` unchanged.
- **Milestones:** Savings **$6,500** achieved **Jan 2028** per Result.

## Goals and milestone fit
- **Emergency savings fund ($6,500)**: **Achieved** (Jan 2028).

</STRATEGY>

<SIMULATION_CALLS>

# Round 1

## Invoked tools

1. **`simulate_financial_strategy`** {"scenario_id":"status_quo"}
2. **`simulate_financial_strategy`** {"scenario_id":"cut_both","spending_schedule":[{"start_end_month":"1+","categories":{"meals":700,"leisure":350}}],"savings_targets":[6500]}
3. **`simulate_financial_strategy`** {"scenario_id":"protect_leisure_cut_meals","spending_schedule":[{"start_end_month":"1-3","categories":{"meals":850}},{"start_end_month":"4+","categories":{"meals":750}}],"savings_targets":[6500],"savings_per_month":150}

# Simulation: protect_leisure_cut_meals

## Result
Cash-flow feasible: yes
- Emergency savings fund ($6,500): achieved By Jan 2028 (mo 22) ($6,520 savings)

</SIMULATION_CALLS>

<USER_PREFERENCES>

Willing to cut food but loves travel — protect leisure/travel when trimming discretionary spending.

</USER_PREFERENCES>
""",
    "output": '{"score": 5, "notes": "Three scenarios with phased preference-aligned winner and savings date matches ## Result."}',
  },
  {
    "name": "bad_invented_milestone_date",
    "batch": 2,
    "input": """<TASK_AND_NEEDS>

# Financial Needs

## Primary needs
1. **Reduce interest drag**

</TASK_AND_NEEDS>

<STRATEGY>

# Financial Strategy

## Recommended plan
- **Scenario id:** `steady_cut`
- **Milestones:** Interest-free **May 2026**.

## Goals and milestone fit
- **Interest-free credit**: **Achieved** (May 2026).

</STRATEGY>

<SIMULATION_CALLS>

# Round 1

## Invoked tools

1. **`simulate_financial_strategy`** {"scenario_id":"status_quo"}
2. **`simulate_financial_strategy`** {"scenario_id":"light_cut","spending_schedule":[{"start_end_month":"1-3","categories":{"meals":900}},{"start_end_month":"4+","categories":{"meals":800}}],"credit_balance_target":0}
3. **`simulate_financial_strategy`** {"scenario_id":"steady_cut","spending_schedule":[{"start_end_month":"1+","categories":{"meals":700}}],"credit_balance_target":0}

# Simulation: steady_cut

## Result
Cash-flow feasible: yes
- Interest-free credit: achieved By Aug 2026 (mo 5)

</SIMULATION_CALLS>
""",
    "output": '{"score": 1, "notes": "Cites interest-free May 2026 but ## Result for steady_cut says Aug 2026."}',
  },
  {
    "name": "medium_recommends_infeasible_without_tradeoff",
    "batch": 2,
    "input": """<TASK_AND_NEEDS>

# Financial Needs

## Primary needs
1. **Build emergency fund** — **$6,500**

</TASK_AND_NEEDS>

<STRATEGY>

# Financial Strategy

## Recommended plan
- **Scenario id:** `deep_cut`
- **Parent category changes:** `meals: $974→$400`.

## Goals and milestone fit
- **Emergency savings fund ($6,500)**: **Achieved** (Dec 2027).

</STRATEGY>

<SIMULATION_CALLS>

# Round 1

## Invoked tools

1. **`simulate_financial_strategy`** {"scenario_id":"status_quo"}
2. **`simulate_financial_strategy`** {"scenario_id":"moderate","spending_schedule":[{"start_end_month":"1-3","categories":{"meals":850}},{"start_end_month":"4+","categories":{"meals":750}}],"savings_targets":[6500]}
3. **`simulate_financial_strategy`** {"scenario_id":"deep_cut","spending_schedule":[{"start_end_month":"1+","categories":{"meals":400}}],"savings_targets":[6500]}

# Simulation: deep_cut

## Result
Cash-flow feasible: no
Warnings: discretionary floor breached in mo 2

# Simulation: moderate

## Result
Cash-flow feasible: yes
- Emergency savings fund ($6,500): achieved By Dec 2027

</SIMULATION_CALLS>
""",
    "output": '{"score": 3, "notes": "Recommends infeasible deep_cut without explaining tradeoff though moderate is feasible with matching milestone."}',
  },
]


class SimulateFinancialStrategyCheckerOptimizer:
  def __init__(
    self,
    model_name: str = "gemini-flash-lite-latest",
    *,
    max_output_tokens: int = 256,
    thinking_budget: int = 0,
  ):
    if genai is None or types is None:
      raise ImportError(
        "Missing dependency for Gemini client. Install `google-genai` (and ensure your venv is active) "
        "to run this script."
      )
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set.")
    self.client = genai.Client(api_key=api_key)
    self._types = types
    self.model_name = model_name
    self.temperature = 0.0
    self.top_p = 0.95
    self.max_output_tokens = max_output_tokens
    self.thinking_budget = thinking_budget
    self.system_prompt = SYSTEM_PROMPT
    self.safety_settings = [
      self._types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      self._types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      self._types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      self._types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ]
    self.output_schema = _build_output_schema(self._types)

  def grade(self, bundled_input: str) -> Dict[str, Any]:
    user_msg = (bundled_input or "").strip()
    t = self._types
    contents = [t.Content(role="user", parts=[t.Part.from_text(text=user_msg)])]
    cfg = t.GenerateContentConfig(
      temperature=self.temperature,
      top_p=self.top_p,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[t.Part.from_text(text=self.system_prompt)],
      thinking_config=t.ThinkingConfig(thinking_budget=self.thinking_budget, include_thoughts=True),
      response_mime_type="application/json",
      response_schema=self.output_schema,
    )
    out = self.client.models.generate_content(model=self.model_name, contents=contents, config=cfg)
    text = (out.text or "").strip()
    try:
      return json.loads(text)
    except Exception:
      s = text[text.find("{") : text.rfind("}") + 1] if ("{" in text and "}" in text) else "{}"
      return json.loads(s)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--test", type=str, default=None, help="Test name, index, or 'all'.")
  parser.add_argument("--batch", type=int, default=None, help="Run all tests in batch N.")
  parser.add_argument("--model", type=str, default="gemini-flash-lite-latest")
  parser.add_argument("--max-output-tokens", type=int, default=256)
  parser.add_argument("--thinking-budget", type=int, default=0)
  parser.add_argument(
    "--check",
    action="store_true",
    help="After grading, require integer score to match expected output JSON. Exits non-zero on failure.",
  )
  args = parser.parse_args()

  if args.test is None and args.batch is None:
    print("Available test cases:")
    for i, tc in enumerate(TEST_CASES):
      batch = tc.get("batch")
      batch_s = str(batch) if isinstance(batch, int) else "—"
      print(f"  {i}: {tc.get('name')} (batch {batch_s})")
    return

  opt = SimulateFinancialStrategyCheckerOptimizer(
    model_name=args.model,
    max_output_tokens=args.max_output_tokens,
    thinking_budget=args.thinking_budget,
  )

  if args.batch is not None:
    cases = [(i, tc) for i, tc in enumerate(TEST_CASES) if int(tc.get("batch") or 0) == int(args.batch)]
    if not cases:
      raise SystemExit(f"No tests found for batch={args.batch}")
  elif (args.test or "").strip().lower() == "all":
    cases = list(enumerate(TEST_CASES))
  else:
    if (args.test or "").isdigit():
      idx = int(args.test)
      if idx < 0 or idx >= len(TEST_CASES):
        raise SystemExit(f"Test index out of range: {idx}")
      cases = [(idx, TEST_CASES[idx])]
    else:
      idx_tc = next(((i, t) for i, t in enumerate(TEST_CASES) if t.get("name") == args.test), None)
      if idx_tc is None:
        raise SystemExit(f"Unknown test: {args.test!r}")
      cases = [idx_tc]

  failures: list[str] = []

  for run_i, (case_index, tc) in enumerate(cases):
    if run_i:
      print(f"\n{_TEST_SEPARATOR}\n")
    batch = tc.get("batch")
    batch_s = str(batch) if isinstance(batch, int) else "—"
    print(f"# Test: {case_index}  {tc['name']}  (batch {batch_s})\n")
    _print_section_banner("# LLM Checker Input")
    print(tc["input"])
    result = opt.grade(tc["input"])
    _print_section_banner("# LLM Checker Response")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.check:
      exp = _parse_expected_output(tc.get("output"))
      if not isinstance(exp, dict) or "score" not in exp:
        failures.append(f"{tc.get('name')}: invalid expected output JSON")
      else:
        try:
          got = int(result.get("score"))
          want = int(exp["score"])
        except Exception:
          failures.append(f"{tc.get('name')}: non-integer score")
        else:
          if got != want:
            failures.append(f"{tc.get('name')}: score {got} != expected {want}")
    if tc.get("output") is not None:
      _print_section_banner("# Expected Output")
      print(tc["output"])

  print(f"\n{_TEST_SEPARATOR}\n")
  print(f"# Total tests: {len(cases)}\n")
  if args.check and failures:
    print("# CHECK FAILURES\n")
    for line in failures:
      print(line)
    raise SystemExit(1)
  if args.check and not failures:
    print("# CHECK: all cases passed score match.\n")


if __name__ == "__main__":
  main()
