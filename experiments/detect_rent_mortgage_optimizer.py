"""
Prompt optimizer harness for rent/mortgage-detection LLM (not runtime detection).
"""

import argparse
import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

DEFAULT_MODEL = "gemini-flash-lite-latest"

SYSTEM_PROMPT = """You are a housing-spend specialist for a personal finance app.
Given candidate groups, decide which group ids are true rent/mortgage shelter spend.

Rules:
- Mark shelter_home for clear rent/mortgage signals: landlord names, lease rent, mortgage servicers, P&I/escrow style labels.
- Exclude hotels/travel lodging, temporary stays, property tax payments, and non-housing recurring payments.
- For no-keyword rows, use recurrence + amount consistency + context, and be conservative when uncertain.

Return:
- shelter_ids
- excluded_ids
- notes

Notes rule:
- Do not mention any ids in notes.
- Names can be mentioned when useful.
"""

OUTPUT_SCHEMA = types.Schema(
  type=types.Type.OBJECT,
  required=["shelter_ids", "excluded_ids", "notes"],
  properties={
    "shelter_ids": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.INTEGER)),
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


class RentMortgageDetectionOptimizer:
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
      print(f"detect_rent_mortgage_optimizer generate_content failed: {e}")
      _ = time.perf_counter() - t0
      return ""
    _ = time.perf_counter() - t0
    if response.candidates:
      for part in response.candidates[0].content.parts:
        if part.text:
          output_text += part.text
    return output_text


_MAX_CASES_PER_OPTIMIZER_BATCH = 4

TEST_CASES = [
  {
    "batch": 1,
    "name": "clear_landlord_and_mortgage",
    "input": """[
  {
    "id": 1,
    "name": "LANDLORD LLC RENT",
    "transactions": [
      "2026-01-01  $1850  uncategorized",
      "2026-02-01  $1850  uncategorized"
    ]
  },
  {
    "id": 2,
    "name": "ROCKET MORTGAGE",
    "transactions": [
      "2026-01-03  $2200  uncategorized",
      "2026-02-03  $2200  uncategorized"
    ]
  }
]""",
    "output": """{"shelter_ids":[1,2],"excluded_ids":[],"notes":"Landlord and mortgage servicer names clearly indicate shelter payments."}""",
  },
  {
    "batch": 1,
    "name": "exclude_travel_lodging",
    "input": """[
  {
    "id": 1,
    "name": "MARRIOTT HOTEL",
    "transactions": [
      "2026-02-07  $620  travel_lodging"
    ]
  }
]""",
    "output": """{"shelter_ids":[],"excluded_ids":[1],"notes":"Hotel lodging is travel-related and excluded from rent or mortgage."}""",
  },
  {
    "batch": 1,
    "name": "no_keyword_fallback_positive",
    "input": """[
  {
    "id": 1,
    "name": "ACH DEBIT 8891",
    "transactions": [
      "2026-01-10  $2100  uncategorized",
      "2026-02-10  $2105  uncategorized"
    ]
  }
]""",
    "output": """{"shelter_ids":[1],"excluded_ids":[],"notes":"Large consistent recurring debit suggests a likely housing payment."}""",
  },
  {
    "batch": 1,
    "name": "no_keyword_fallback_negative",
    "input": """[
  {
    "id": 1,
    "name": "WIRE OUT 5531",
    "transactions": [
      "2026-01-12  $2100  uncategorized",
      "2026-02-12  $2098  uncategorized"
    ]
  }
]""",
    "output": """{"shelter_ids":[],"excluded_ids":[1],"notes":"Recurring wire activity lacks direct housing evidence and is excluded conservatively."}""",
  },
  {
    "batch": 2,
    "name": "mixed_split_case",
    "input": """[
  {
    "id": 1,
    "name": "MR COOPER MORTGAGE",
    "transactions": [
      "2026-03-01  $2400  uncategorized"
    ]
  },
  {
    "id": 2,
    "name": "AIRBNB",
    "transactions": [
      "2026-03-05  $700  travel_vacations"
    ]
  }
]""",
    "output": """{"shelter_ids":[1],"excluded_ids":[2],"notes":"Mortgage servicer payment is included while temporary lodging is excluded."}""",
  },
  {
    "batch": 2,
    "name": "empty_candidates",
    "input": """[]""",
    "output": """{"shelter_ids":[],"excluded_ids":[],"notes":"No candidate groups were provided for classification."}""",
  },
  {
    "batch": 2,
    "name": "hoa_and_property_tax_mixed",
    "input": """[
  {
    "id": 1,
    "name": "SUNSET HOA DUES",
    "transactions": [
      "2026-03-01  $425  uncategorized"
    ]
  },
  {
    "id": 2,
    "name": "COUNTY PROPERTY TAX",
    "transactions": [
      "2026-03-15  $980  uncategorized"
    ]
  }
]""",
    "output": """{"shelter_ids":[1],"excluded_ids":[2],"notes":"HOA payment is shelter-related while property tax is excluded in this workflow."}""",
  },
  {
    "batch": 3,
    "name": "escrow_and_utility_split",
    "input": """[
  {
    "id": 1,
    "name": "HOME LOAN ESCROW",
    "transactions": [
      "2026-02-01  $1950  uncategorized"
    ]
  },
  {
    "id": 2,
    "name": "CITY WATER BILL",
    "transactions": [
      "2026-02-14  $145  bills_utilities"
    ]
  }
]""",
    "output": """{"shelter_ids":[1],"excluded_ids":[2],"notes":"Escrow-linked mortgage payment is included; utility bill is excluded."}""",
  },
  {
    "batch": 3,
    "name": "ambiguous_recurring_transfer_exclude",
    "input": """[
  {
    "id": 1,
    "name": "INT TRANSFER 1001",
    "transactions": [
      "2026-01-09  $2000  uncategorized",
      "2026-02-09  $2000  uncategorized"
    ]
  }
]""",
    "output": """{"shelter_ids":[],"excluded_ids":[1],"notes":"Recurring internal transfer lacks housing-specific signals and is excluded."}""",
  },
]

_optimizer_batch_sizes: dict[int, int] = {}
for _tc in TEST_CASES:
  _bn = int(_tc.get("batch", 1))
  _optimizer_batch_sizes[_bn] = _optimizer_batch_sizes.get(_bn, 0) + 1
assert all(_count <= _MAX_CASES_PER_OPTIMIZER_BATCH for _count in _optimizer_batch_sizes.values()), _optimizer_batch_sizes


def run_test(test_name_or_index_or_dict: int | str | dict, optimizer: RentMortgageDetectionOptimizer | None = None) -> str | None:
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
    optimizer = RentMortgageDetectionOptimizer()
  result = optimizer.generate_response(tc["input"])
  print("## LLM Output:")
  print(result)
  if tc.get("output"):
    print("## Ideal output (reference):")
    print(tc["output"])
  return result


def run_all_tests_batch(optimizer: RentMortgageDetectionOptimizer | None = None, batch_num: int = 1):
  if optimizer is None:
    optimizer = RentMortgageDetectionOptimizer()
  for tc in [x for x in TEST_CASES if x.get("batch") == batch_num]:
    run_test(tc, optimizer)


def main(test: str | None = None, run_batch: bool = False, batch_num: int = 1, model: str | None = None):
  kw: dict = {}
  if model is not None:
    kw["model_name"] = model
  optimizer = RentMortgageDetectionOptimizer(**kw)

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
  parser = argparse.ArgumentParser(description="Rent/mortgage detection prompt optimizer harness.")
  parser.add_argument("--test", type=str, default=None)
  parser.add_argument("--batch", type=int, nargs="?", const=1, default=None, metavar="N")
  parser.add_argument("--model", type=str, default=None)
  args = parser.parse_args()
  main(test=args.test, run_batch=args.batch is not None, batch_num=(1 if args.batch is None else args.batch), model=args.model)
