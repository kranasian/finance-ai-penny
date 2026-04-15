"""
Prompt optimizer harness for rent/mortgage-detection LLM (not runtime detection).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

DEFAULT_MODEL = "gemini-flash-lite-latest"

SYSTEM_PROMPT = """JSON in: groups `{id, name, transactions[]}`; each line is `DATE  $AMOUNT  category_token`.

Return `shelter_ids` (rent or mortgage **principal** streams) and `excluded_ids` (everything else).

**Primary:** each group’s own calendar spacing + dollar bands—**judge streams independently**; never demote one group because another in the same payload also looks like housing.

Favor **~monthly** debits at **housing-scale** dollars: stable low-thousands with modest drift still counts as rent-like; **fixed** monthlies at mortgage-typical dollars count when naming or context does not scream non-housing debt.

**Caution on small fixed monthlies:** a tight recurring **few-hundred-dollar** line whose `name` is a **generic Loan** (no Mortgage/Mtg/Rent/landlord wording) is usually **consumer/installment**, not shelter principal—exclude even if perfectly periodic.

Disfavor sub-$200 erratic ladders, internal sweep patterns, one-off five-figure checks without a rent rhythm, retail-scatter sizes, issuer card-pay bands.

**Category tokens are unreliable**—do not let them override dates+dollars. **Names are secondary** tie-breakers when rhythm+size collide (Mortgage/Mtg vs Loan, Rent/Apts vs plain Transfer).

Exclude travel lodging, property-tax-only workflows, obvious internal transfers, credit-card pay streams.

**Notes:** ≤3 sentences; never numeric ids; cite **cadence + $** (name words only if timing is ambiguous). Forbidden phrasing: "is rent", "is not rent", "is mortgage", "transaction is", or close variants."""

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


def _compare_shelter_id_sets(actual_text: str, ideal_text: str) -> tuple[bool, str]:
  """Compare shelter_ids and excluded_ids only (ignores notes wording)."""
  try:
    actual = json.loads(actual_text or "{}")
    ideal = json.loads(ideal_text or "{}")
  except json.JSONDecodeError as exc:
    return False, f"invalid JSON ({exc})"
  a_s = set(actual.get("shelter_ids", []))
  a_e = set(actual.get("excluded_ids", []))
  i_s = set(ideal.get("shelter_ids", []))
  i_e = set(ideal.get("excluded_ids", []))
  if a_s != i_s or a_e != i_e:
    return (
      False,
      f"shelter_ids model={sorted(a_s)} ideal={sorted(i_s)}; excluded_ids model={sorted(a_e)} ideal={sorted(i_e)}",
    )
  return True, "shelter_ids and excluded_ids match reference"


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
      print(f"detect_rent_mortgage_optimizer generate_content_stream failed: {e}")
      _ = time.perf_counter() - t0
      return ""
    _ = time.perf_counter() - t0
    if thought_summary.strip():
      print(f"{'=' * 80}")
      print("THOUGHT SUMMARY:")
      print(thought_summary.strip())
      print("=" * 80)
    return output_text


_MAX_CASES_PER_OPTIMIZER_BATCH = 1

TEST_CASES = [
  {
    "batch": 1,
    "name": "photo_batch_1_kaiser_chase_travel_bilt_rent",
    "input": """[
  {
    "id": 3907,
    "name": "Kaiser Permanente (#3907)",
    "transactions": [
      "2026-04-01  $916.55  bills_insurance",
      "2026-03-04  $916.55  bills_insurance",
      "2026-03-01  $18.00  bills_insurance",
      "2026-01-28  $916.55  bills_insurance"
    ]
  },
  {
    "id": 24951,
    "name": "Chase Travel (#24951)",
    "transactions": [
      "2026-04-13  -$824.53  leisure_travel_vacations",
      "2026-04-13  $824.93  leisure_travel_vacations",
      "2026-04-12  $875.95  leisure_travel_vacations",
      "2026-04-12  -$521.93  leisure_travel_vacations",
      "2026-04-12  $824.53  leisure_travel_vacations"
    ]
  },
  {
    "id": 382819,
    "name": "Bilt Rent - Avalonbay (#382819)",
    "transactions": [
      "2026-01-05  $3898.52  shelter_home"
    ]
  }
]""",
    "output": """{"shelter_ids":[382819],"excluded_ids":[3907,24951],"notes":"The Bilt rent stream is housing-scale and explicitly rent-labeled; Kaiser insurance and travel reversals do not show rent or mortgage payment cadence."}""",
  },
  {
    "batch": 2,
    "name": "photo_batch_2_viewparadise_goldman_zelle",
    "input": """[
  {
    "id": 546756,
    "name": "Viewparadise (#546756)",
    "transactions": [
      "2026-03-09  $3036.00  leisure_entertainment"
    ]
  },
  {
    "id": 4672,
    "name": "ACH Hold: Goldman Sachs Bank Collection (#4672)",
    "transactions": [
      "2026-04-14  $3000.00  transfer"
    ]
  },
  {
    "id": 42849,
    "name": "Zelle from Marilyn R Velez (#42849/#17306)",
    "transactions": [
      "2026-04-03  $500.00  transfer",
      "2026-03-03  -$2200.00  transfer",
      "2026-02-13  $200.00  transfer",
      "2026-02-02  -$1600.00  transfer",
      "2026-01-23  -$70.00  transfer"
    ]
  }
]""",
    "output": """{"shelter_ids":[],"excluded_ids":[546756,4672,42849],"notes":"These lines are one-off or mixed-direction transfer activity without repeated monthly housing cadence, so no rent or mortgage principal stream is identified."}""",
  },
  {
    "batch": 3,
    "name": "photo_batch_3_ava_commons_walmart_atm",
    "input": """[
  {
    "id": 103619,
    "name": "AVA Commons (#103619)",
    "transactions": [
      "2026-03-09  $2887.62  shelter_home",
      "2026-03-05  $1007.95  shelter_home",
      "2026-02-17  $941.35  shelter_home",
      "2026-02-03  $3007.95  shelter_home",
      "2025-12-08  $3762.34  shelter_home"
    ]
  },
  {
    "id": 388,
    "name": "Walmart / Walmart Payroll (#388/#606675)",
    "transactions": [
      "2026-04-07  -$250.00  income_salary",
      "2026-04-07  $18.00  meals_groceries",
      "2026-04-06  -$301.23  uncategorized",
      "2026-04-03  $2.28  meals_groceries",
      "2026-04-01  $21.69  meals_groceries"
    ]
  },
  {
    "id": 4771,
    "name": "ATM Withdrawal / Out-of-Network Fee (#4771/#242889)",
    "transactions": [
      "2026-04-12  $40.00  uncategorized",
      "2026-03-24  $182.95  uncategorized",
      "2026-03-24  $2.50  bills_service_fees",
      "2026-03-23  $200.00  uncategorized",
      "2026-03-16  $202.50  uncategorized"
    ]
  }
]""",
    "output": """{"shelter_ids":[103619],"excluded_ids":[388,4771],"notes":"AVA Commons shows recurring housing-scale debits across months; Walmart and ATM groups are payroll/retail/cash activity without rent or mortgage recurrence."}""",
  },
  {
    "batch": 4,
    "name": "photo_batch_4_moneylion_instacash_turbo_aliexpress",
    "input": """[
  {
    "id": 119912,
    "name": "MoneyLion Instacash (#119912)",
    "transactions": [
      "2026-04-05  -$5.00  transfer",
      "2026-03-28  -$100.00  transfer",
      "2026-03-28  -$20.00  transfer",
      "2026-03-27  $300.00  transfer",
      "2026-03-27  -$100.00  transfer"
    ]
  },
  {
    "id": 617991,
    "name": "Moneylion Turbo Transfer to ***8583 (#617991)",
    "transactions": [
      "2026-03-27  $300.00  uncategorized",
      "2026-03-27  $200.00  uncategorized",
      "2026-03-27  $500.00  uncategorized",
      "2026-03-26  $500.00  uncategorized"
    ]
  },
  {
    "id": 1494,
    "name": "AliExpress (#1494)",
    "transactions": [
      "2026-04-04  $0.11  uncategorized",
      "2026-03-24  $4.50  uncategorized",
      "2026-03-24  $9.32  uncategorized"
    ]
  }
]""",
    "output": """{"shelter_ids":[],"excluded_ids":[119912,617991,1494],"notes":"MoneyLion and AliExpress activity is short-cycle transfer or micro-spend behavior, not month-over-month housing principal cadence."}""",
  },
]

_optimizer_batch_sizes: dict[int, int] = {}
for _tc in TEST_CASES:
  _bn = int(_tc.get("batch", 1))
  _optimizer_batch_sizes[_bn] = _optimizer_batch_sizes.get(_bn, 0) + 1
assert sorted(_optimizer_batch_sizes.keys()) == [1, 2, 3, 4], _optimizer_batch_sizes
assert all(_count == _MAX_CASES_PER_OPTIMIZER_BATCH for _count in _optimizer_batch_sizes.values()), _optimizer_batch_sizes


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
    if result and result.strip():
      ok, detail = _compare_shelter_id_sets(result, tc["output"])
      print("## Sandbox execution (ID sets vs reference):")
      print("PASS" if ok else "FAIL")
      print(detail)
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
  parser.add_argument("--batch", type=int, nargs="?", const=1, default=None, metavar="N", help="Run the combined multi-id harness case for batch N (1–4; default 1).")
  parser.add_argument("--model", type=str, default=None)
  args = parser.parse_args()
  main(test=args.test, run_batch=args.batch is not None, batch_num=(1 if args.batch is None else args.batch), model=args.model)
