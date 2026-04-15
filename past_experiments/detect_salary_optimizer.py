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

SYSTEM_PROMPT = """Input: JSON array of {id, name, transactions[]} — each line is date, $amount, category_token.

Primary signals (in order): (1) amount scale, shape, and pairing—stable wage-sized repeats vs micro-amount noise vs fee-flat lines vs large offsetting debits/credits in the same band; (2) calendar recurrence and cadence across lines. Category tokens are weak and often mis-tagged—do not let them override amount+rhythm evidence. The group name is secondary (employer/Payroll/DEP language, Instapay/Advance/travel/P2P hints); use it only when dollars+dates are ambiguous.

salary_ids: coherent recurring inflows at plausible take-home scale with payroll-like stability—including **variable check-to-check amounts** (hundreds to low thousands) when dates cluster like pay cycles and the name reads as employer **Payroll** (not Instapay/Advance). Do **not** demand every line be four-figure if the rhythm and scale still resemble wage deposits. **Score each group independently**—never deny a Payroll-named employer stream because another group in the same payload posts larger ACH numbers. Mid-three-digit to ~$1k swings without micro-gig flecks (<$15) can still be payroll. If **Payroll** appears in the employer-facing name (excluding Instapay/Advance) and there are **≥3** credits each **≥ ~$40** spanning **≥14 days** of calendar coverage, treat as salary_ids unless the series is unmistakably a transfer/autopay chip pattern (near-identical bill-sized repeats). excluded_ids: everything else—including identical repeating amounts from labor-agency/unemployment context (DOL/UI) even if mislabeled salary; earned-wage style names (Instapay, MyPay Advance, @Work, etc.) stay excluded unless the amount series clearly matches employer ACH pay-scale rhythm, not small transfer steps; travel-sized mirrored flows; autopay/transfer ladders; gig/rebate magnitudes; brokerage-scale business swings.

notes: ≤3 sentences; names only (no ids). Justify with amount/recurrence facts (e.g. bi-weekly ~$3.8k vs $1–$5 micro streak). Never: "is payroll", "is not payroll", "transaction is payroll", "transaction is not payroll"."""

OUTPUT_SCHEMA = types.Schema(
  type=types.Type.OBJECT,
  required=["salary_ids", "excluded_ids", "notes"],
  properties={
    "salary_ids": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.INTEGER)),
    "excluded_ids": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.INTEGER)),
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
  """Compare salary_ids and excluded_ids only (ignores notes wording)."""
  try:
    actual = json.loads(actual_text or "{}")
    ideal = json.loads(ideal_text or "{}")
  except json.JSONDecodeError as exc:
    return False, f"invalid JSON ({exc})"
  a_s = set(actual.get("salary_ids", []))
  a_e = set(actual.get("excluded_ids", []))
  i_s = set(ideal.get("salary_ids", []))
  i_e = set(ideal.get("excluded_ids", []))
  if a_s != i_s or a_e != i_e:
    return (
      False,
      f"salary_ids model={sorted(a_s)} ideal={sorted(i_s)}; excluded_ids model={sorted(a_e)} ideal={sorted(i_e)}",
    )
  return True, "salary_ids and excluded_ids match reference"


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
    "name": "photo_batch_1_justplay_instacash_axiom",
    "input": """[
  {
    "id": 601784,
    "name": "Justplay",
    "transactions": [
      "2026-03-23  $1.03  income_side_gig",
      "2026-03-23  $3.52  income_side_gig",
      "2026-03-20  $2.56  income_side_gig",
      "2026-03-18  $3.08  income_side_gig",
      "2026-03-18  $1.69  income_side_gig",
      "2026-03-17  $3.11  income_side_gig"
    ]
  },
  {
    "id": 432517,
    "name": "Instacash Turbo Fee Rebate",
    "transactions": [
      "2026-03-12  $5.00  bills_service_fees",
      "2026-02-12  $5.00  bills_service_fees",
      "2026-01-15  $5.00  bills_service_fees"
    ]
  },
  {
    "id": 601905,
    "name": "Axiom Healthcare Payroll",
    "transactions": [
      "2026-03-26  $549.82  income_salary",
      "2026-03-12  $790.62  income_salary",
      "2026-02-25  $1000.00  income_salary",
      "2026-02-11  $1000.00  income_salary",
      "2026-01-29  $1000.00  income_salary",
      "2026-01-15  $504.57  income_salary"
    ]
  }
]""",
    "output": """{"salary_ids":[601905],"excluded_ids":[601784,432517],"notes":"Axiom Healthcare is clear payroll; Justplay app payouts and Instacash fee rebates are not wage payroll."}""",
  },
  {
    "batch": 2,
    "name": "photo_batch_2_chase_travel_at_work_mypay",
    "input": """[
  {
    "id": 24951,
    "name": "Chase Travel",
    "transactions": [
      "2026-04-13  $824.53  leisure_travel_vacations",
      "2026-04-13  -$824.93  leisure_travel_vacations",
      "2026-04-12  -$875.95  leisure_travel_vacations",
      "2026-04-12  $521.93  leisure_travel_vacations",
      "2026-04-12  -$824.53  leisure_travel_vacations",
      "2026-03-10  -$721.93  leisure_travel_vacations",
      "2025-12-14  -$122.11  leisure_travel_vacations",
      "2025-10-14  $348.46  leisure_travel_vacations",
      "2025-10-14  -$164.75  leisure_travel_vacations",
      "2025-10-14  -$607.70  leisure_travel_vacations",
      "2025-10-14  $494.25  leisure_travel_vacations"
    ]
  },
  {
    "id": 606645,
    "name": "At Work Instapay",
    "transactions": [
      "2026-04-12  $84.09  income_salary",
      "2026-04-11  $127.11  income_salary",
      "2026-04-05  $49.50  uncategorized",
      "2026-04-01  $90.36  uncategorized",
      "2026-03-27  $100.00  uncategorized",
      "2026-03-23  $40.27  uncategorized",
      "2026-02-06  $20.00  income_salary",
      "2026-02-06  $200.00  income_salary",
      "2026-01-28  $83.60  income_salary"
    ]
  },
  {
    "id": 11564,
    "name": "MyPay Advance",
    "transactions": [
      "2026-04-09  $40.00  transfer",
      "2026-03-29  $20.00  transfer",
      "2026-03-24  $20.00  transfer",
      "2026-03-16  $40.00  transfer",
      "2026-03-10  $20.00  transfer",
      "2026-02-25  $40.00  transfer",
      "2026-02-14  $25.00  transfer"
    ]
  }
]""",
    "output": """{"salary_ids":[],"excluded_ids":[24951,606645,11564],"notes":"Chase Travel reward redemptions; At Work Instapay and MyPay advances are not wage payroll."}""",
  },
  {
    "batch": 3,
    "name": "photo_batch_3_labor_force_autopay_keybank_ach",
    "input": """[
  {
    "id": 590149,
    "name": "Labor Force Group Payroll",
    "transactions": [
      "2026-04-07  $100.07  income_salary",
      "2026-03-31  $124.36  income_salary",
      "2026-03-24  $344.95  income_salary",
      "2026-03-17  $627.87  income_salary",
      "2026-03-10  $505.50  income_salary"
    ]
  },
  {
    "id": 367,
    "name": "Autopay",
    "transactions": [
      "2026-03-26  $181.00  transfer",
      "2026-01-26  $188.00  transfer",
      "2025-12-26  $181.00  transfer",
      "2025-11-26  $193.00  transfer",
      "2025-10-26  $186.00  transfer",
      "2025-09-26  $194.00  transfer",
      "2025-09-11  $155.00  transfer",
      "2025-08-26  $194.00  transfer",
      "2025-08-11  $155.00  transfer",
      "2025-07-25  $188.00  transfer"
    ]
  },
  {
    "id": 5487,
    "name": "ACH Credit: Keybank National Direct Deposit",
    "transactions": [
      "2026-03-13  $3809.69  income_salary",
      "2026-03-06  $20375.10  income_salary",
      "2026-02-27  $3808.50  income_salary",
      "2026-02-13  $3808.49  income_salary",
      "2026-01-30  $3859.50  income_salary",
      "2026-01-16  $3808.50  income_salary"
    ]
  }
]""",
    "output": """{"salary_ids":[5487,590149],"excluded_ids":[367],"notes":"KeyBank ACH DIR DEP and Labor Force Group Payroll are clear payroll; card autopay is excluded."}""",
  },
  {
    "batch": 4,
    "name": "photo_batch_4_nys_keybank_mixed_fidelity",
    "input": """[
  {
    "id": 4470,
    "name": "NYS Department of Labor",
    "transactions": [
      "2024-07-10  $504.00  income_salary",
      "2024-07-03  $504.00  income_salary",
      "2024-06-28  $504.00  income_salary",
      "2024-06-20  $504.00  income_salary",
      "2024-06-12  $504.00  income_salary",
      "2024-06-05  $504.00  income_salary",
      "2024-05-30  $504.00  income_salary",
      "2024-05-23  $504.00  income_salary",
      "2024-05-15  $504.00  income_salary"
    ]
  },
  {
    "id": 5487,
    "name": "Keybank National Direct Deposit",
    "transactions": [
      "2025-10-10  -$7.41  income_salary",
      "2025-01-03  $3726.10  income_salary",
      "2024-12-20  $3715.67  income_salary",
      "2024-12-06  $3273.96  income_salary",
      "2024-11-22  $3273.95  income_salary",
      "2024-11-08  $3715.68  income_salary"
    ]
  },
  {
    "id": 4569,
    "name": "Fidelity Investments",
    "transactions": [
      "2024-06-27  $540.00  income_business",
      "2024-06-12  $1750.00  income_business",
      "2024-05-14  $1200.00  income_business",
      "2024-04-24  $3500.00  income_business",
      "2024-04-16  $7000.00  income_business"
    ]
  }
]""",
    "output": """{"salary_ids":[5487],"excluded_ids":[4470,4569],"notes":"KeyBank DIR DEP pattern is payroll; NYS DOL unemployment and Fidelity Moneyline are not employer payroll."}""",
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
