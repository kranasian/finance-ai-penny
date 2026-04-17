"""
Prompt optimizer harness for salary-detection LLM (not runtime detection).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

DEFAULT_MODEL = "gemini-flash-lite-latest"

SYSTEM_PROMPT = """Input JSON: [{id,name,transactions[]}], each tx line contains date, signed $amount, category_token.

Hard rules:
1) Direction comes ONLY from sign: `$x` inflow, `-$x` outflow. Never infer direction from name/category.
2) Salaries are inflows; outflows cannot be salary evidence.
3) Evaluate each id independently and return every id exactly once in salary_ids or excluded_ids.

Per-id checklist:
- Parse counts: inflow_count, outflow_count, inflow_amounts, date spacing of inflows.
- **If all inflows share one calendar date, that is not recurring income** (one posting day, not a paycheck schedule over time)—do not treat same-day duplicates as cadence.
- **Different calendar dates alone are not “recurring”:** recurrence needs a **rough repeating gap pattern** between **sorted** +inflow dates—e.g. ~7±3d weekly, ~14±4d biweekly, ~15–17d semimonthly pairs, ~28–35d monthly (allow a few days slack; amounts can vary modestly). **Irregular/random spacing** across dates does **not** qualify as salary cadence unless Exception A (payroll/direct-deposit name) applies with its own date rules.
- Detect cadence from inflows only (weekly/biweekly/semimonthly/monthly or near-monthly) **as approximate interval families**, not as “any multi-date scatter.”
- Detect non-wage patterns: transfer/advance loop, reversal mirror (similar +/- amounts), tiny side-gig inflows, one-off transfer/loan swings.
- Any amount without '-' is inflow. Calling it outflow is invalid.
- If all listed amounts are positive, the group is inflow-only by definition.

Decision policy:
- Highest-priority rule: if name contains "payroll" or "direct deposit", classify as salary_ids unless transactions are mostly outflows, clearly advance/transfer-reversal behavior, or **all inflows share one calendar date** (same-day clusters are not recurring income).
- If category suggests salary/income, assume salary_ids UNLESS strong contrary evidence exists (sparse inflows, **no rough paycheck gap pattern** across +dates, mixed with clear non-wage pattern, mostly outflows, or advance/reversal loop).
- If category is not salary, still classify as salary_ids when +inflows show a **rough paycheck-like gap pattern** (not merely different dates) and wage-like scale; name is only secondary.
- Recurring positive inflows must not be excluded only because merchant/category looks like spending.

Minimum salary evidence:
- **Same-day veto:** If **every** inflow line shares the **same calendar date** (YYYY-MM-DD), that pattern is **not recurring income**; **do not** classify as salary_ids on recurrence/cadence or on Exception A/B below—same-day clusters are excluded_ids for wage-recurrence purposes.
- General rule: require >=3 inflows with **recurring cadence** for salary_ids: **>=2 different calendar dates** **and** sorted +inflow gaps that **roughly** fit one paycheck rhythm (weekly/biweekly/semimonthly/monthly bands above)—**not** “different dates only.”
- Exception A: if name contains "payroll" or "direct deposit" and there are >=2 inflows (>=40) on **>=2 different calendar dates**, classify as salary_ids (name shortcut still **does not** bypass **same-day veto**).
- Exception B: if there are >=4 inflows on **>=2 different calendar dates** and **gap pattern** is near-monthly or biweekly (rough interval consistency), classify as salary_ids even when category/name look like shopping.
- If only 1-2 inflows and no strong payroll/direct-deposit name, classify as excluded_ids.
- Guardrail: names implying bill/transfer automation (e.g., "autopay", "payment", transfer-like wording) are excluded_ids unless strong payroll/direct-deposit wording is present.

Behavior anchors:
- Four monthly `+$839.18` entries labeled shopping => salary_ids (recurring inflow stream).
- Transfer-labeled "Autopay" inflow series => excluded_ids (transfer/bill automation pattern).
- Two inflows with payroll/direct-deposit naming => salary_ids.
- Two inflows mixed with several outflows and no payroll/direct-deposit naming => excluded_ids.
- Many identical inflows **all on the same calendar date** => excluded_ids (same day is not recurring income cadence).
- +Inflows on **several different dates** but **irregular gaps** (no rough weekly/biweekly/monthly family) => excluded_ids unless Exception A applies.

salary_ids = recurring wage-like inflow streams (variable amounts allowed) across **>=2 calendar dates** with a **rough interval pattern**, plus short histories with strong payroll/direct-deposit naming when dates differ; **same-date-only inflows are never recurring income** for this purpose.
excluded_ids = all other ids.
notes: <=3 sentences, names only, cite sign/gap-pattern/recurrence/amount facts briefly (note same-day-only or irregular spacing when relevant)."""

OUTPUT_SCHEMA = types.Schema(
  type=types.Type.OBJECT,
  required=["salary_transaction_ids", "excluded_transaction_ids", "notes"],
  properties={
    "salary_transaction_ids": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.INTEGER)),
    "excluded_transaction_ids": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.INTEGER)),
    "notes": types.Schema(type=types.Type.STRING),
  },
)

SAFETY_SETTINGS = [
  types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
  types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
  types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
  types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
]


def _compare_id_sets(actual_text: str, ideal_text: str) -> tuple[bool, str]:
  """Compare salary_transaction_ids and excluded_transaction_ids only (ignores notes wording)."""
  try:
    actual = json.loads(actual_text or "{}")
    ideal = json.loads(ideal_text or "{}")
  except json.JSONDecodeError as exc:
    return False, f"invalid JSON ({exc})"
  a_s = set(actual.get("salary_transaction_ids", []))
  a_e = set(actual.get("excluded_transaction_ids", []))
  i_s = set(ideal.get("salary_transaction_ids", []))
  i_e = set(ideal.get("excluded_transaction_ids", []))
  if a_s != i_s or a_e != i_e:
    return (
      False,
      f"salary_transaction_ids model={sorted(a_s)} ideal={sorted(i_s)}; excluded_transaction_ids model={sorted(a_e)} ideal={sorted(i_e)}",
    )
  return True, "salary_transaction_ids and excluded_transaction_ids match reference"


class SalaryDetectionOptimizer:
  def __init__(self, model_name: str = DEFAULT_MODEL, thinking_budget: int = 0, max_output_tokens: int = 1024):
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
      raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set.")
    self.client = genai.Client(api_key=api_key)
    self.model_name = model_name
    self.thinking_budget = thinking_budget
    self.max_output_tokens = max_output_tokens
    self.temperature = 0.3
    self.top_p = 0.95

  def generate_response(self, prompt_override: str) -> str:
    print("## LLM Input\n")
    print(prompt_override)
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt_override)])]
    config = types.GenerateContentConfig(
      temperature=self.temperature,
      top_p=self.top_p,
      max_output_tokens=self.max_output_tokens,
      response_mime_type="application/json",
      response_schema=OUTPUT_SCHEMA,
      safety_settings=SAFETY_SETTINGS,
      system_instruction=[types.Part.from_text(text=SYSTEM_PROMPT)],
      thinking_config=types.ThinkingConfig(
        thinking_budget=self.thinking_budget,
        include_thoughts=True,
      ),
    )
    output_text = ""
    thought_summary = ""
    t0 = time.perf_counter()
    try:
      for chunk in self.client.models.generate_content_stream(
        model=self.model_name,
        contents=contents,
        config=config,
      ):
        if chunk.text is not None:
          output_text += chunk.text
        if hasattr(chunk, "candidates") and chunk.candidates:
          for candidate in chunk.candidates:
            if hasattr(candidate, "content") and candidate.content:
              if hasattr(candidate.content, "parts") and candidate.content.parts:
                for part in candidate.content.parts:
                  if hasattr(part, "thought") and part.thought:
                    if hasattr(part, "text") and part.text:
                      thought_summary += part.text
    except Exception as e:
      print(f"detect_salary_optimizer generate_content_stream failed: {e}")
      _ = time.perf_counter() - t0
      return ""
    _ = time.perf_counter() - t0
    if thought_summary.strip():
      print(f"{'=' * 80}")
      print("THOUGHT SUMMARY:")
      print(thought_summary.strip())
      print("=" * 80)
    return output_text


def _normalize_optimizer_input(input_payload: str) -> str:
  try:
    raw = json.loads(input_payload or "[]")
  except Exception:
    return input_payload
  if not isinstance(raw, list):
    return input_payload
  if raw and isinstance(raw[0], dict) and "name" in raw[0] and "transactions" in raw[0]:
    normalized_grouped: list[dict[str, Any]] = []
    for idx, row in enumerate(raw, start=1):
      normalized_grouped.append(
        {
          "id": int(row.get("id", idx)),
          "name": str(row.get("name", "")).strip(),
          "transactions": list(row.get("transactions", [])),
        }
      )
    return json.dumps(normalized_grouped, indent=2)

  grouped: dict[str, list[dict[str, Any]]] = {}
  for row in raw:
    if not isinstance(row, dict):
      continue
    name = str(row.get("merchant", row.get("transaction_name", ""))).strip()
    grouped.setdefault(name, []).append(row)

  out: list[dict[str, Any]] = []
  for idx, (name, rows) in enumerate(grouped.items(), start=1):
    rows_sorted = sorted(rows, key=lambda r: str(r.get("date", "")), reverse=True)[:5]
    tx_lines: list[str] = []
    for row in rows_sorted:
      date_str = str(row.get("date", ""))[:10]
      amount = abs(float(row.get("amount", 0.0)))
      amount_str = str(int(amount)) if amount.is_integer() else f"{amount:.2f}"
      category = str(row.get("category", "")).strip()
      tx_lines.append(f"{date_str}  ${amount_str}  {category}")
    out.append({"id": idx, "name": name, "transactions": tx_lines})
  return json.dumps(out, indent=2)


_MAX_CASES_PER_OPTIMIZER_BATCH = 1

TEST_CASES = [
  {
    "batch": 1,
    "name": "photo_batch_1_keybank_axiom_payroll_streams",
    "input": """[
  {
    "id": 5487,
    "name": "ACH Credit: Keybank National Direct Deposit",
    "transactions": [
      "2026-03-13  $3809.69  income_salary",
      "2026-03-06  $20375.10  income_salary",
      "2026-02-27  $3808.50  income_salary",
      "2026-02-13  $3808.49  income_salary",
      "2026-01-30  $3859.50  income_salary"
    ]
  },
  {
    "id": 588112,
    "name": "KeyBank National Payroll",
    "transactions": [
      "2026-04-10  $4282.73  income_salary",
      "2026-03-27  $3970.91  income_salary"
    ]
  },
  {
    "id": 601905,
    "name": "Walmart",
    "transactions": [
      "2026-03-26  $549.82  shopping_clothing",
      "2026-03-12  $790.62  shopping_clothing",
      "2026-02-25  $1000.00  shopping_clothing",
      "2026-02-11  $1000.00  shopping_clothing",
      "2026-01-29  $1000.00  shopping_clothing",
      "2026-01-15  $504.57  shopping_clothing"
    ]
  }
]""",
    "output": """{"salary_transaction_ids":[5487,601905,588112],"excluded_transaction_ids":[],"notes":"5487: five +inflows on staggered dates with large amounts and biweekly-like gaps; non-transfer name. 588112: two +inflows ≥$40 on two dates; income_salary tags trusted. 601905: six +lines on six dates, three identical +$1000.00 ~14d apart—cadence from amounts/dates; categories secondary."}""",
  },
  {
    "batch": 2,
    "name": "photo_batch_2_autopay_card_loan_transfers",
    "input": """[
  {
    "id": 367,
    "name": "Autopay",
    "transactions": [
      "2026-04-13  $327.10  transfer",
      "2026-03-17  $111.30  transfer",
      "2026-03-13  $1063.74  transfer",
      "2026-02-13  $918.54  transfer",
      "2026-01-13  $1066.80  transfer"
    ]
  },
  {
    "id": 9451,
    "name": "Apple",
    "transactions": [
      "2026-04-02  $839.18  shopping_gadgets",
      "2026-03-01  $839.18  shopping_gadgets",
      "2026-02-02  $839.18  shopping_gadgets",
      "2026-01-02  $839.18  shopping_gadgets"
    ]
  },
  {
    "id": 506503,
    "name": "Walmart",
    "transactions": [
      "2026-03-15  -$23  meals_groceries",
      "2026-03-01  $839.18  meals_groceries",
      "2026-02-02  $839.18  meals_groceries",
      "2026-01-10  -$11  meals_groceries",
      "2026-01-08  -$12  meals_groceries",
      "2026-01-02  $839.18  meals_groceries"
    ]
  }
]""",
    "output": """{"salary_transaction_ids":[9451,506503],"excluded_transaction_ids":[367],"notes":"367: transfer-like Autopay name → excluded regardless of + cadence. 9451: four identical +$839.18 on four month-spaced dates; non-transfer name. 506503: same +$839.18 monthly cadence with small -$ debits; non-transfer name—wage logic from +series/dates."}""",
  },
  {
    "batch": 3,
    "name": "photo_batch_3_chase_zelle_instacash",
    "input": """[
  {
    "id": 24951,
    "name": "Chase Travel",
    "transactions": [
      "2026-04-13  $824.53  leisure_travel_vacations",
      "2026-04-13  -$824.93  leisure_travel_vacations",
      "2026-04-12  -$875.95  leisure_travel_vacations",
      "2026-04-12  $521.93  leisure_travel_vacations",
      "2026-04-12  -$824.53  leisure_travel_vacations"
    ]
  },
  {
    "id": 17306,
    "name": "McDonald's",
    "transactions": [
      "2026-04-02  $2200.00  meals_dining_out",
      "2026-03-02  $2200.00  meals_dining_out",
      "2026-02-02  $2200.00  meals_dining_out",
      "2026-01-02  $2200.00  meals_dining_out"
    ]
  },
  {
    "id": 119912,
    "name": "MoneyLion Instacash",
    "transactions": [
      "2026-04-05  $5.00  income_salary",
      "2026-03-28  $100.00  income_salary",
      "2026-03-28  $20.00  income_salary",
      "2026-03-27  -$300.00  income_salary",
      "2026-03-27  $100.00  income_salary"
    ]
  }
]""",
    "output": """{"salary_transaction_ids":[17306],"excluded_transaction_ids":[24951,119912],"notes":"24951: same-day +/- mirror amounts—reversal shape from signs/amounts. 17306: four +$2200.00 on four month-spaced dates; non-transfer name—monthly cadence from amounts/dates; dining tags secondary. 119912: transfer-like Instacash name plus large -$300 with small +—advance/repay override despite salary tags."}""",
  },
  {
    "batch": 4,
    "name": "photo_batch_4_same_day_same_amount_direct_deposit",
    "input": """[
  {
    "id": 900001,
    "name": "ACH Credit Fidelity Direct Deposit",
    "transactions": [
      "2026-04-17  $2500.00  income_salary",
      "2026-04-17  $2500.00  income_salary",
      "2026-04-17  $2500.00  income_salary",
      "2026-04-17  $2500.00  income_salary",
      "2026-04-17  $2500.00  income_salary"
    ]
  }
]""",
    "output": """{"salary_transaction_ids":[],"excluded_transaction_ids":[900001],"notes":"900001: non-transfer name but five +$2500.00 on one calendar date—no multi-date cadence; salary tags do not override same-date-only stack."}""",
  },
]

_optimizer_batch_sizes: dict[int, int] = {}
for _tc in TEST_CASES:
  _bn = int(_tc.get("batch", 1))
  _optimizer_batch_sizes[_bn] = _optimizer_batch_sizes.get(_bn, 0) + 1
assert sorted(_optimizer_batch_sizes.keys()) == [1, 2, 3, 4], _optimizer_batch_sizes
assert all(_count == _MAX_CASES_PER_OPTIMIZER_BATCH for _count in _optimizer_batch_sizes.values()), _optimizer_batch_sizes


def run_test(test_name_or_index_or_dict: int | str | dict, optimizer: SalaryDetectionOptimizer | None = None) -> str | None:
  if isinstance(test_name_or_index_or_dict, dict):
    tc = test_name_or_index_or_dict
  elif isinstance(test_name_or_index_or_dict, int):
    idx = test_name_or_index_or_dict
    if not (0 <= idx < len(TEST_CASES)):
      return None
    tc = TEST_CASES[idx]
  else:
    tc = next((t for t in TEST_CASES if t["name"] == test_name_or_index_or_dict), None)
    if not tc:
      return None

  print(f"# Test: **{tc['name']}**")
  if optimizer is None:
    optimizer = SalaryDetectionOptimizer()

  llm_input = _normalize_optimizer_input(tc["input"])
  result = optimizer.generate_response(llm_input)
  print("## LLM Output:")
  print(result)
  if tc.get("output"):
    print("## Ideal output (reference):")
    print(tc["output"])
    if result.strip():
      ok, detail = _compare_id_sets(result, tc["output"])
      print("## ID-set match:", "PASS" if ok else "FAIL")
      print(detail)
  return result


def run_all_tests_batch(optimizer: SalaryDetectionOptimizer | None = None, batch_num: int = 1):
  if optimizer is None:
    optimizer = SalaryDetectionOptimizer()
  for tc in [x for x in TEST_CASES if x.get("batch") == batch_num]:
    run_test(tc, optimizer)


def main(test: str | None = None, run_batch: bool = False, batch_num: int = 1, model: str | None = None):
  kw: dict = {}
  if model is not None:
    kw["model_name"] = model
  optimizer = SalaryDetectionOptimizer(**kw)

  if run_batch:
    run_all_tests_batch(optimizer, batch_num=batch_num)
    return
  if test is not None:
    if test.strip().lower() == "all":
      for batch in sorted({int(tc.get("batch", 1)) for tc in TEST_CASES}):
        run_all_tests_batch(optimizer, batch_num=batch)
      return
    run_test(int(test) if test.isdigit() else test, optimizer)
    return

  print("Available test cases:")
  for i, tc in enumerate(TEST_CASES):
    print(f"  {i}: {tc['name']} (batch {tc.get('batch')})")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Salary detection prompt optimizer harness.")
  parser.add_argument("--test", type=str, default=None)
  parser.add_argument("--batch", type=int, nargs="?", const=1, default=None, metavar="N", help="Run the combined multi-id harness case for batch N (1–4; default 1).")
  parser.add_argument("--model", type=str, default=None)
  args = parser.parse_args()
  main(test=args.test, run_batch=args.batch is not None, batch_num=(1 if args.batch is None else args.batch), model=args.model)
