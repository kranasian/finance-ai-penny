from __future__ import annotations

from google import genai
from google.genai import types
import json
import os
from dotenv import load_dotenv


load_dotenv()


OUTPUT_SCHEMA = types.Schema(
  type=types.Type.OBJECT,
  required=["rationale", "rules_satisfied"],
  properties={
    "rationale": types.Schema(
      type=types.Type.STRING,
      description=(
        "Fact-only string <= 140 chars. If false, lists failure reasons separated by semicolons."
      ),
    ),
    "rules_satisfied": types.Schema(
      type=types.Type.BOOLEAN,
      description="True if all lines pass all rules; false otherwise.",
    ),
  },
)


SYSTEM_PROMPT = """Return only JSON with `rationale` and `rules_satisfied`.
Set `rationale` first; `rules_satisfied` must match it (true=all clear, false=failures).

Input you receive:
- `# Categorize Request`: a categorization-style instruction aimed at a **specific group** of transactions. The **group is defined only by the filters/rules** in that section (who is in scope), not by whether the chosen category is wise.
- `# Transactions`: a **guess** at which transactions belong to that group—each bullet is a proposed member.
- Transaction line format: `$Amount Name on YYYY-MM-DD` (example: `$10 McDonald's on 2020-08-30`).

Your job: **membership check**—decide whether `# Transactions` is **correct** for the group implied by `# Categorize Request`. For every bullet, verify it satisfies **every** explicit filter that defines the group. Do not judge category sensibility beyond applying those filters literally.
- If a filter says transactions must be named as something, the transaction name must match exactly with no added/removed letters or words; comparison is case-insensitive.

Strictness — no exceptions:
- Every filter in the request is mandatory. Do not waive, blend, round away, or partially apply a rule. Do not pass a line because it is "close enough" or because most lines look fine.
- `rules_satisfied` is true only if every bullet passes every mapped filter with zero exceptions.

Rules:
1) Map every explicit filter, then validate each bullet against all of them:
   - Payee/name: Whenever the request fixes how transactions must be named (from '…', must be exactly/named/equal to a literal, or a full multi-word merchant label such as Instacart Costco delivery), extract the establishment from each bullet (strip noise like `transaction`, `on`, dates), fold case, and require a **case-insensitive exact string match** to that required name—same characters and order; PAYPAL equals PayPal; Venmo Transfer fails against Venmo. Fuzzy matching applies **only** if the request explicitly demands it (includes, contains, something like, similar to, does not have to be exactly, the name does not have to be exactly '…'); then follow that wording strictly (usually case-insensitive substring contains).
   - Amounts and dates: apply bounds literally (at least / more than / exactly / between). No unstated tolerance except where the request uses around/about/approximately/approx (amount band [0.95*X, 1.05*X]; date band Y−2..Y+2 days).
   - Account: only when the request ties filters to account_id, account, or posts to an account; then every line must show matching account info.
2) Around amount near $X: [0.95*X, 1.05*X] inclusive.
3) Around date near Y: Y−2 through Y+2 days inclusive.
4) Account optional unless the request requires it; missing required account fails.
5) If any single line breaks any single mapped filter, `rules_satisfied` is false.

Rationale:
- One line, fact-only, <=140 chars (target <=100), use $ and digits.
- If false, cite failing line(s) by date and/or amount and the strict rule violated; keep multiple failures compact.
- Never set true by relaxing a filter; never set false for category taste."""


TEST_CASES = [
  {
    "name": "chipotle_under_cap_after_cutoff_all_pass",
    "input": """# Categorize Request

Re-categorize all transactions from 'Chipotle' where each charge was $30 or less on or after 2026-03-01, setting their category to 'meals_dining_out'.

# Transactions

- $11 Chipotle on 2026-03-04.
- $18 Chipotle on 2026-03-09.
- $9 Chipotle on 2026-03-14.
- $30 Chipotle on 2026-03-19.
- $14 Chipotle on 2026-03-24.""",
    "output": {
      "rationale": "Every Chipotle line meets the $30 cap and is on or after 2026-03-01.",
      "rules_satisfied": True,
    },
  },
  {
    "name": "delta_baggage_minimum_one_shortfall",
    "input": """# Categorize Request

Re-categorize all transactions from 'Delta Air Lines' as 'travel_airfare' where each charge was at least $55.

# Transactions

- $60 Delta Air Lines on 2026-01-10.
- $72 Delta Air Lines on 2026-01-11.
- $45 Delta Air Lines on 2026-01-12.
- $80 Delta Air Lines on 2026-01-13.
- $55 Delta Air Lines on 2026-01-14.""",
    "output": {
      "rationale": "Jan 12 charge is $45 and below the $55 minimum.",
      "rules_satisfied": False,
    },
  },
  {
    "name": "freelance_invoice_contains_and_floor",
    "input": """# Categorize Request

Re-categorize transactions as 'income_freelance' where the name includes 'invoice' and the amount is at least $300.

# Transactions

- $450 Acme Design Invoice 4412 on 2026-02-03.
- $320 March invoice — Bright LLC on 2026-02-10.
- $900 INVOICE PAID Q1 on 2026-02-17.
- $305 invoice #7721 on 2026-02-24.""",
    "output": {
      "rationale": "All lines include invoice wording and are at least $300.",
      "rules_satisfied": True,
    },
  },
  {
    "name": "venmo_exact_name_one_line_differs",
    "input": """# Categorize Request

Categorize these rows as 'transfers_peer_to_peer'; for each row the name must be exactly 'Venmo'.

# Transactions

- $40 Venmo on 2026-04-02.
- $25 Venmo Transfer on 2026-04-03.
- $60 Venmo on 2026-04-04.
- $15 Venmo on 2026-04-05.
- $90 Venmo on 2026-04-06.""",
    "output": {
      "rationale": "Apr 3 line is Venmo Transfer, not exactly Venmo.",
      "rules_satisfied": False,
    },
  },
  {
    "name": "sweetgreen_wednesdays_only_all_pass",
    "input": """# Categorize Request

Re-categorize all transactions from 'Sweetgreen' as 'meals_dining_out' where the transaction date is on a Wednesday.

# Transactions

- $16 Sweetgreen on 2026-01-07.
- $19 Sweetgreen on 2026-01-14.
- $14 Sweetgreen on 2026-01-21.
- $22 Sweetgreen on 2026-01-28.""",
    "output": {
      "rationale": "All Sweetgreen charges fall on a Wednesday.",
      "rules_satisfied": True,
    },
  },
  {
    "name": "hulu_subscription_exact_price_mismatch",
    "input": """# Categorize Request

Categorize the Hulu subscription charges as 'leisure_entertainment' where each amount is exactly $14.99.

# Transactions

- $14.99 Hulu on 2026-05-01.
- $15.49 Hulu on 2026-06-01.
- $14.99 Hulu on 2026-07-01.
- $14.99 Hulu on 2026-08-01.
- $14.99 Hulu on 2026-09-01.""",
    "output": {
      "rationale": "Jun 1 charge is $15.49, not $14.99.",
      "rules_satisfied": False,
    },
  },
  {
    "name": "paypal_refunds_negative_all_pass",
    "input": """# Categorize Request

Re-categorize all transactions from 'PayPal' as 'income_refunds' when the amount is negative.

# Transactions

- -$12 PAYPAL transaction on March 3, 2026.
- -$44 PayPal transaction on March 5, 2026.
- -$8 paypal transaction on March 7, 2026.
- -$19 PayPal transaction on March 9, 2026.
- -$31 PAYPAL transaction on March 11, 2026.""",
    "output": {
      "rationale": "Every PayPal line shows a negative amount.",
      "rules_satisfied": True,
    },
  },
  {
    "name": "instacart_costco_delivery_literal_subset_fail",
    "input": """# Categorize Request

Re-categorize the Instacart Costco delivery transactions as 'groceries'.

# Transactions

- $95 Instacart Costco delivery on 2026-02-18.
- $40 Instacart on 2026-02-19.
- $88 Costco same-day delivery on 2026-02-20.
- $62 Instacart Costco on 2026-02-21.
- $55 Instacart delivery on 2026-02-22.""",
    "output": {
      "rationale": "Only Feb 18 matches Instacart Costco delivery; other payees differ.",
      "rules_satisfied": False,
    },
  },
  {
    "name": "duke_energy_under_cap_one_spike",
    "input": """# Categorize Request

Re-categorize all transactions from 'Duke Energy' as 'utilities_electric' when the charge is under $125.

# Transactions

- $98 Duke Energy on 2026-01-15.
- $135 Duke Energy on 2026-02-15.
- $110 Duke Energy on 2026-03-15.
- $88 Duke Energy on 2026-04-15.
- $102 Duke Energy on 2026-05-15.""",
    "output": {
      "rationale": "Feb 15 Duke Energy charge is $135 and not under $125.",
      "rules_satisfied": False,
    },
  },
  {
    "name": "apple_store_spend_band_one_outlier",
    "input": """# Categorize Request

Re-categorize all transactions from 'Apple Store' as 'shopping_electronics' where each charge amount is between $40 and $160.

# Transactions

- $99 Apple Store on 2026-03-02.
- $179 Apple Store on 2026-03-09.
- $45 Apple Store on 2026-03-16.
- $160 Apple Store on 2026-03-23.
- $52 Apple Store on 2026-03-30.""",
    "output": {
      "rationale": "Mar 9 Apple Store charge is $179 and outside the $40–$160 band.",
      "rules_satisfied": False,
    },
  },
  {
    "name": "doordash_min_amount_extra_request_text_fail",
    "input": """# Categorize Request

Re-categorize all 'DoorDash' charges as 'meals_dining_out' where each amount is at least $18. Also, add a note for future similar deliveries.

# Transactions

- $22 DoorDash on 2026-06-02.
- $31 DoorDash on 2026-06-03.
- $17 DoorDash on 2026-06-04.
- $26 DoorDash on 2026-06-05.
- $20 DoorDash on 2026-06-06.""",
    "output": {
      "rationale": "Jun 4 DoorDash charge is $17 and under the $18 floor.",
      "rules_satisfied": False,
    },
  },
  {
    "name": "transit_pass_contains_muni_all_pass",
    "input": """# Categorize Request

Re-categorize these transit purchases as 'transport_public_transit' where the name contains 'Muni'.

# Transactions

- $2.50 SF Muni Clipper on 2026-04-01.
- $5 MUNI DAY PASS on 2026-04-02.
- $2.50 Muni fare on 2026-04-03.
- $5 Muni Mobile on 2026-04-04.
- $2.50 SFMUNI on 2026-04-05.""",
    "output": {
      "rationale": "Every line includes Muni and matches the transit filter.",
      "rules_satisfied": True,
    },
  },
  {
    "name": "blue_bottle_around_mid_august_window",
    "input": """# Categorize Request

Re-categorize all transactions from 'Blue Bottle' on around August 20, 2026 as 'meals_dining_out'.

# Transactions

- $7 Blue Bottle on 2026-08-18.
- $9 Blue Bottle on 2026-08-19.
- $8 Blue Bottle on 2026-08-20.
- $6 Blue Bottle on 2026-08-21.
- $11 Blue Bottle on 2026-08-22.""",
    "output": {
      "rationale": "All Blue Bottle dates sit within ±2 days of August 20, 2026.",
      "rules_satisfied": True,
    },
  },
  {
    "name": "zeta_vendor_pipe_tab_and_around_amount",
    "input": """# Categorize Request

Re-categorize transactions where the name contains 'ZETA' and each charge is around $80.

# Transactions

- $82 ZETA|Cloud billing on 2026-09-01.
- $78 ZETA\tPayroll sync on 2026-09-02.
- $76 ZETA Labs on 2026-09-03.
- $84 ZETA Holdings on 2026-09-04.
- $79 ZETA API on 2026-09-05.""",
    "output": {
      "rationale": "Each line has ZETA and each amount is within ±5% of $80.",
      "rules_satisfied": True,
    },
  },
  {
    "name": "spotify_usa_exact_name_every_line_all_pass",
    "input": """# Categorize Request

Categorize each line as 'subscriptions_music'; for every line the name must be exactly 'Spotify USA'.

# Transactions

- $11 Spotify USA on 2026-10-01.
- $11 SPOTIFY USA on 2026-10-02.
- $11 spotify usa on 2026-10-03.
- $11 Spotify USA on 2026-10-04.
- $11 Spotify USA on 2026-10-05.""",
    "output": {
      "rationale": "Every payee matches Spotify USA after case folding.",
      "rules_satisfied": True,
    },
  },
  {
    "name": "chewy_includes_fuzzy_one_line_without_substring",
    "input": """# Categorize Request

Re-categorize purchases as 'shopping_pet' where the name includes 'Chewy' and each charge was under $65.

# Transactions

- $44 Chewy on 2026-11-01.
- $52 CHEWY.COM on 2026-11-02.
- $38 Autoship Chewy on 2026-11-03.
- $61 Chewy Goody Box on 2026-11-04.
- $30 Petco on 2026-11-05.""",
    "output": {
      "rationale": "Nov 5 Petco line has no Chewy substring in the name.",
      "rules_satisfied": False,
    },
  },
]


BATCHES: dict[int, dict[str, object]] = {
  1: {
    "name": "Pass / min fail / contains / exact-name",
    "tests": [0, 1, 2, 3],
  },
  2: {
    "name": "Contains + min-with-noise + transit contains",
    "tests": [2, 10, 11],
  },
  3: {
    "name": "Compound payee + utility cap + retail band",
    "tests": [7, 8, 9],
  },
  4: {
    "name": "Weekday + exact $ + refunds + compound payee",
    "tests": [4, 5, 6, 7],
  },
  5: {
    "name": "Around date + delimiter-heavy names",
    "tests": [12, 13],
  },
  6: {
    "name": "Exact-name all-pass + fuzzy includes one miss",
    "tests": [14, 15],
  },
}


class UpdateTransactionCategoryVerifyOptimizer:
  """Gemini verify client.

  Last live check (16 `TEST_CASES`, all batch indices): `gemini-flash-lite-latest`, max_output_tokens 2048 — 0 golden
  mismatches on `rules_satisfied`; notes stayed ≤100 chars without spelled-out money/years or banned filler words.

  Defaults: `gemini-flash-lite-latest`, `temperature=0.2`, `top_p=0.95`, `top_k=40`,
  `thinking_budget=512`, `max_output_tokens=2048`. If verdicts drift, try `gemini-2.0-flash`.
  """

  def __init__(self, model_name: str = "gemini-flash-lite-latest"):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set.")
    self.client = genai.Client(api_key=api_key)
    self.model_name = model_name or "gemini-flash-lite-latest"
    self.thinking_budget = 512
    self.temperature = 0.2
    self.top_p = 0.95
    self.top_k = 40
    self.max_output_tokens = 2048
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ]
    self.system_prompt = SYSTEM_PROMPT

  def generate_response(self, user_message: str) -> str:
    request_text = types.Part.from_text(text=user_message)
    contents = [types.Content(role="user", parts=[request_text])]
    config = types.GenerateContentConfig(
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
      response_schema=OUTPUT_SCHEMA,
    )
    output_text = ""
    thought_summary = ""
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
    if thought_summary.strip():
      print(f"{'=' * 80}")
      print("THOUGHT SUMMARY:")
      print(thought_summary.strip())
      print("=" * 80)
    return output_text


def get_test_case(test_name_or_index):
  """
  Get a test case by name or index.
  
  Args:
    test_name_or_index: Test case name (str) or index (int)
    
  Returns:
    Test case dict or None if not found
  """
  if isinstance(test_name_or_index, int):
    if 0 <= test_name_or_index < len(TEST_CASES):
      return TEST_CASES[test_name_or_index]
    return None
  if isinstance(test_name_or_index, str):
    for test_case in TEST_CASES:
      if test_case["name"] == test_name_or_index:
        return test_case
  return None


def _run_test_with_logging(tc: dict, optimizer: UpdateTransactionCategoryVerifyOptimizer | None = None):
  if optimizer is None:
    optimizer = UpdateTransactionCategoryVerifyOptimizer()

  print("\n" + "=" * 80)
  print(f"Running verification test: {tc['name']}")
  print("=" * 80)
  print("\nLLM INPUT:")
  print(tc["input"])

  result = optimizer.generate_response(tc["input"])

  print("\nLLM OUTPUT:")
  print(result)
  print("\nEXPECTED OUTPUT (compact):")
  print(json.dumps(tc["output"], indent=2))
  print("\n" + "=" * 80 + "\n")
  return result


def run_test(test_name_or_index_or_dict, optimizer: UpdateTransactionCategoryVerifyOptimizer | None = None):
  if isinstance(test_name_or_index_or_dict, dict):
    tc = test_name_or_index_or_dict
  else:
    tc = get_test_case(test_name_or_index_or_dict)

  if tc is None:
    print(f"Test case '{test_name_or_index_or_dict}' not found.")
    return None

  return _run_test_with_logging(tc, optimizer)


def run_tests(test_names_or_indices=None, optimizer: UpdateTransactionCategoryVerifyOptimizer | None = None):
  """
  Run multiple tests by names or indices.
  
  Args:
    test_names_or_indices: list of names/indices, or None to run all tests
  """
  if test_names_or_indices is None:
    test_names_or_indices = list(range(len(TEST_CASES)))
  results = []
  for item in test_names_or_indices:
    results.append(run_test(item, optimizer=optimizer))
  return results


def main(test: str = None, batch: int | None = None, model: str | None = None):
  """
  CLI entrypoint to inspect verification test cases.
  
  Args:
    test: optional test name or index (as string); if provided, runs only that test.
    batch: optional batch number; runs the tests listed for that batch in BATCHES.
  """
  if batch is not None:
    if batch not in BATCHES:
      print(f"Invalid batch number: {batch}. Available batches: {sorted(BATCHES.keys())}")
      print("\nBatch descriptions:")
      for b, info in BATCHES.items():
        test_names = [TEST_CASES[idx]["name"] for idx in info["tests"]]
        print(f"  Batch {b}: {info['name']} — {', '.join(test_names)}")
      return
    info = BATCHES[batch]
    print(f"\nRunning batch {batch}: {info['name']}\n")
    optimizer = UpdateTransactionCategoryVerifyOptimizer(model_name=model or "gemini-flash-lite-latest")
    run_tests(test_names_or_indices=info["tests"], optimizer=optimizer)
    return
  
  if test is not None:
    optimizer = UpdateTransactionCategoryVerifyOptimizer(model_name=model or "gemini-flash-lite-latest")
    # Accept numeric index via string
    test_key: int | str
    if test.isdigit():
      test_key = int(test)
    else:
      test_key = test
    run_test(test_key, optimizer=optimizer)
    return
  
  # No args: list available tests
  print("Available verification test cases:")
  for i, tc in enumerate(TEST_CASES):
    print(f"  {i}: {tc['name']}")


# python experiments/check_before_recategorize_optimizer.py --test 0
# python experiments/check_before_recategorize_optimizer.py --batch 1
# python past_experiments/check_before_recategorize_optimizer.py --test chipotle_under_cap_after_cutoff_all_pass
if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(
    description="Inspect update_transaction_category verification SYSTEM_PROMPT test cases."
  )
  parser.add_argument(
    "--test",
    type=str,
    help="Test name or index to run (e.g., 'chipotle_under_cap_after_cutoff_all_pass' or '0').",
  )
  parser.add_argument(
    "--batch",
    type=int,
    nargs="?",
    const=1,
    default=None,
    metavar="N",
    help="Run batch N of tests (see BATCHES in script).",
  )
  parser.add_argument(
    "--model",
    type=str,
    default=None,
    help="Gemini model name (e.g. gemini-2.0-flash).",
  )
  args = parser.parse_args()
  main(test=args.test, batch=args.batch, model=args.model)