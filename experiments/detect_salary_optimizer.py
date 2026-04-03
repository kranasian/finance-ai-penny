"""
Prompt optimizer harness for salary-detection LLM (not runtime detection).
"""

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

SYSTEM_PROMPT = """You are a payroll and income specialist for a personal finance app.
Given grouped incoming transactions by name, decide which names are definitively recurring salary or payroll.

Rules:
- Mark only clear payroll: employer names, ADP, Gusto, Rippling, Paychex, state/federal payroll, "PAYROLL", "DIRECT DEP", regular wage patterns.
- Exclude: one-off transfers, investment proceeds, interest, refunds, P2P (Venmo/Zelle) unless clearly labeled payroll, gig platforms unless clearly W-2-style payroll description.

Input notes:
- Each group has an integer `id` and a `name`.
- `transactions` is a compact history for that name.

Fill the structured response:
- salary_ids: ids to mark as payroll
- excluded_ids: ids ruled out
- notes: short rationale; do not mention ids. names can appear here if needed.
"""

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
      thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
    )
    output_text = ""
    t0 = time.perf_counter()
    try:
      response = self.client.models.generate_content(model=self.model_name, contents=contents, config=config)
    except Exception as e:
      print(f"detect_salary_optimizer generate_content failed: {e}")
      _ = time.perf_counter() - t0
      return ""
    _ = time.perf_counter() - t0
    if response.candidates:
      for part in response.candidates[0].content.parts:
        if part.text:
          output_text += part.text
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


_MAX_CASES_PER_OPTIMIZER_BATCH = 4

TEST_CASES = [
  {
    "batch": 1,
    "name": "salary_adp_gusto_uncategorized",
    "input": """[
  {
    "id": 1,
    "name": "ADP PAYROLL",
    "transactions": [
      "2025-11-20  $2500  uncategorized",
      "2025-10-20  $2500  income_salary"
    ]
  }
]""",
    "output": """{"salary_ids":[1],"excluded_ids":[],"notes":"ADP PAYROLL identified as a standard payroll processor with consistent salary history."}""",
  },
  {
    "batch": 1,
    "name": "salary_exclude_interest",
    "input": """[
  {
    "id": 1,
    "name": "CHASE SAVINGS INTEREST",
    "transactions": [
       "2025-11-01  $4.12  income_interest"
    ]
  },
  {
    "id": 2,
    "name": "ACME CORP PAYROLL",
    "transactions": [
       "2025-11-15  $3200.0  uncategorized"
    ]
  }
]""",
    "output": """{"salary_ids":[2],"excluded_ids":[1],"notes":"Interest income is excluded; employer payroll deposit is included."}""",
  },
  {
    "batch": 1,
    "name": "no_salary_zelle_p2p",
    "input": """[
  {
    "id": 1,
    "name": "RANDOM ZELLE",
    "transactions": [
      "2025-11-15  $800.0  uncategorized"
    ]
  }
]""",
    "output": """{"salary_ids":[],"excluded_ids":[1],"notes":"P2P transfer lacks payroll signals and is excluded."}""",
  },
  {
    "batch": 1,
    "name": "salary_state_payroll_mislabeled_shopping",
    "input": """[
  {
    "id": 1,
    "name": "STATE PAYROLL",
    "transactions": [
      "2025-11-01  $3000.0  income_salary",
      "2025-11-15  $3000.0  shopping_clothing"
    ]
  }
]""",
    "output": """{"salary_ids":[1],"excluded_ids":[],"notes":"State payroll deposits are salary despite one incorrect category label."}""",
  },
  {
    "batch": 2,
    "name": "salary_adp_miscategorized_transfer",
    "input": """[
  {
    "id": 1,
    "name": "ADP PAYROLL",
    "transactions": [
      "2025-10-01  $2400.0  transfers"
    ]
  }
]""",
    "output": """{"salary_ids":[1],"excluded_ids":[],"notes":"ADP payroll signal overrides transfer mislabeling."}""",
  },
  {
    "batch": 2,
    "name": "salary_mixed_amazon_gusto",
    "input": """[
  {
    "id": 1,
    "name": "AMAZON MKTPLACE",
    "transactions": [
      "2025-11-02  $1200.0  shopping_gadgets"
    ]
  },
  {
    "id": 2,
    "name": "GUSTO PAYROLL",
    "transactions": [
      "2025-11-05  $2800.0  uncategorized"
    ]
  }
]""",
    "output": """{"salary_ids":[2],"excluded_ids":[1],"notes":"Retail spend is excluded; Gusto payroll is included."}""",
  },
  {
    "batch": 2,
    "name": "salary_ca_state_payroll_vs_p2p",
    "input": """[
  {
    "id": 1,
    "name": "CA ST PAYROLL",
    "transactions": [
      "2025-11-08  $4100.0  uncategorized"
    ]
  },
  {
    "id": 2,
    "name": "CHASE QUICKPAY",
    "transactions": [
      "2025-11-09  $2000.0  uncategorized"
    ]
  }
]""",
    "output": """{"salary_ids":[1],"excluded_ids":[2],"notes":"State payroll is included; informal quickpay transfer is excluded."}""",
  },
  {
    "batch": 2,
    "name": "salary_exclude_irs_refund_vs_adp",
    "input": """[
  {
    "id": 1,
    "name": "IRS TREAS TAX REFUND",
    "transactions": [
      "2025-04-10  $2400.0  uncategorized"
    ]
  },
  {
    "id": 2,
    "name": "ADP PAYROLL",
    "transactions": [
      "2025-04-12  $3100.0  uncategorized"
    ]
  }
]""",
    "output": """{"salary_ids":[2],"excluded_ids":[1],"notes":"Tax refund is excluded while ADP payroll is included."}""",
  },
  {
    "batch": 3,
    "name": "salary_rippling_w2",
    "input": """[
  {
    "id": 1,
    "name": "RIPPLING PAYROLL",
    "transactions": [
      "2025-12-01  $5200.0  uncategorized"
    ]
  },
  {
    "id": 2,
    "name": "DOORDASH DRIVER PAY",
    "transactions": [
      "2025-12-03  $180.0  income_sidegig"
    ]
  }
]""",
    "output": """{"salary_ids":[1],"excluded_ids":[2],"notes":"Rippling payroll is salary; gig payout is excluded."}""",
  },
  {
    "batch": 3,
    "name": "empty_income_candidates",
    "input": """[]""",
    "output": """{"salary_ids":[],"excluded_ids":[],"notes":"No income candidates were supplied for payroll classification."}""",
  },
]

_optimizer_batch_sizes: dict[int, int] = {}
for _tc in TEST_CASES:
  _bn = int(_tc.get("batch", 1))
  _optimizer_batch_sizes[_bn] = _optimizer_batch_sizes.get(_bn, 0) + 1
assert all(_count <= _MAX_CASES_PER_OPTIMIZER_BATCH for _count in _optimizer_batch_sizes.values()), _optimizer_batch_sizes


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
  parser.add_argument("--batch", type=int, nargs="?", const=1, default=None, metavar="N", help="Run all cases in batch N (default 1).")
  parser.add_argument("--model", type=str, default=None)
  args = parser.parse_args()
  main(test=args.test, run_batch=args.batch is not None, batch_num=(1 if args.batch is None else args.batch), model=args.model)
