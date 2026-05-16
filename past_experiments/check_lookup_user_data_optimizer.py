from google import genai
from google.genai import types
import os
import json
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

RUN_SETTINGS = {
  "json": False,
  "sanitize": False,
  "gen_config": {
    "top_k": 40,
    "top_p": 0.95,
    "temperature": 0.6,
    "thinking_budget": 8196,
    "max_output_tokens": 8192,
  },
  "model_name": "gemini-flash-lite-latest",
}

SYSTEM_PROMPT = """You verify lookup (`P:Func:LookupUserData`) output: `process_input()` code + `EXECUTION_RESULT` (bool, user string, trace/logs). User message has `EVAL_INPUT`, `GENERATED_CODE`, `EXECUTION_RESULT`, optional `PAST_REVIEW_OUTCOMES` (may nest under `output`).

Use only those blocks. Do not invent data, accounts, or category tiers. Penalize only what the snippet and log show. Generic "list my accounts" / net worth via depository+credit need not add investments, loans, or other types unless the user or code explicitly requires them.

**Flags** — `{"good_copy": bool, "info_correct": bool, "eval_text": string}`
- **`info_correct`:** Are **lookups, filters, date windows, category roll-ups, and math** right for the user request? `false` for wrong retrieval logic, wrong aggregation tier, double-counting categories, missing transfer exclusion on spending totals, fuzzy-band errors, empty-guard before filter (when it breaks lookup), malformed return tuple, or totals the code cannot support. **Not** `false` for bool/string mismatch when the retrieval logic itself is sound.
- **`good_copy`:** Assuming lookups/math are right, is the **user-facing text** (and returned bool vs message) well worded for the request? `false` for vague/unlabeled numbers, missing compare interpretation, traceback, bool/message mismatch, incomplete answer, or tone that misstates what was computed.
- **Tie-break:** If the **only** flaw is fuzzy-band / filter / category-tier math in code and the log shows a clean `True`, keep `good_copy` `true` even when the reply omits date, merchant, or scope labels. Fail `good_copy` for label gaps only when `info_correct` is `true`.
- **`eval_text`:** `""` iff both `true`. Else list **every** distinct issue (major and minor)—do not pick only the biggest. Keep it **concise**: terse standalone phrases, `; `-separated (~8–12 words each); aim for the whole string under ~80 words. Cover both flag types when both are `false`. No "Rule" or axis letters.

**Past reviews:** Math/lookup issue → `info_correct`; wording/log issue → `good_copy`. Re-flag only if still broken.

**Categories** (`categories.py` model; DB `category` strings like `meals_groceries`, `bills_service_fees`, top-level ids 41–46):
- **Top-level** (e.g. Food, Bills, Shopping, Income): roll-up of **parent** categories under that top level.
- **Parent** (e.g. Meals, Bills, Shelter): roll-up of **sub-categories** plus transactions tagged to the parent when no sub fits.
- **Sub-category** (leaf, e.g. `meals_groceries`, `bills_service_fees`). Some categories are **parent-only** (no children), e.g. Donations & Gifts (`donations_gifts`).
- **Same tier only** when ranking or comparing "categories" (e.g. highest spend): compare top-level to top-level, parent to parent, sub to sub—never mix (e.g. Bills vs Bills: Service Fees is invalid).
- **Totals are not additive across tiers:** total spending ≠ top + parent + sub sums mixed together. Valid wholes: **sum of all parent categories** OR **sum of all top-level categories** (pick one tier, stay consistent). Sum of **sub-categories only** **excludes** parent-only transactions (no leaf)—undercounts vs parent/total spending. Same logic for income and forecasts.
- **Forecasts:** parent forecast ≠ sum of child forecast lines; do not substitute summing every sub line for a parent total.
- **Spending dataframe includes transfers** (`transfers`, `transfers_*`). For user "spending" totals, **subtract / exclude** transfer rows from spending amounts unless user asked for transfers.

**Code must (`info_correct`)**
1. `def process_input()` → `tuple[bool, str]`; allowed APIs only: all `retrieve_*`, listed `utter_*` / account / transaction / subscription formatters, date helpers (`get_date`, `get_start_of_month`, `get_end_of_month`, `get_start_of_year`, `get_end_of_year`, `get_start_of_week`, `get_end_of_week`, `get_after_periods`, `get_date_string`), `datetime`, `pd`. No import/I/O/network.
2. Empty `df` after each `retrieve_*` before columns/filters/`.str`.
3. Partial case-insensitive name match; whole-dollar **X:** `X-0.50` to `X+0.49` (keep sign; no `abs()` unless asked).
4. Past/next n months/weeks: exclude current period; `datetime.now()` + helpers; no hardcoded dates.
5. Discretionary: `meals*`, `shopping*`, `leisure*`, `donations_gifts`. Prefer `utter_*` for totals; match requested granularity; multi-line → `chr(10).join(output_lines)`.

**Semantics** (if request needs it): **Compare** → net/delta + brief interpretation, not side-by-side only. **Subscriptions** → aggregate by establishment, not one row per charge. **Averages** → same window for sum and period count. **Labels** → each figure states what, period, scope (and ex-transfer for spending totals).

**Good copy (`good_copy`)** — Substantive answer; unsupported → `False` + clear explanation OK; covers all request parts; returned **bool matches** the message (e.g. `False` must not accompany a successful listing); no traceback. When `info_correct` is `true`, fail `good_copy` for missing headers, thin labels, or missing compare interpretation.

**Order:** Past reviews → `info_correct` → `good_copy` → `eval_text` (all issues, each brief).
"""


class CheckLookupUserDataOptimizer:
  """Gemini checker: `info_correct` = lookups/math; `good_copy` = user-facing wording."""

  def __init__(
    self,
    model_name: str = None,
    thinking_budget: int = None,
    max_output_tokens: int = None,
    json_output: bool = None,
    sanitize: bool = None,
  ):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
      raise ValueError(
        "GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment."
      )
    self.client = genai.Client(api_key=api_key)
    gc = RUN_SETTINGS["gen_config"]
    self.model_name = model_name if model_name is not None else RUN_SETTINGS["model_name"]
    self.thinking_budget = (
      thinking_budget if thinking_budget is not None else gc["thinking_budget"]
    )
    self.temperature = gc["temperature"]
    self.top_p = gc["top_p"]
    self.top_k = gc["top_k"]
    self.max_output_tokens = (
      max_output_tokens if max_output_tokens is not None else gc["max_output_tokens"]
    )
    self.json_output = json_output if json_output is not None else RUN_SETTINGS["json"]
    self.sanitize = sanitize if sanitize is not None else RUN_SETTINGS["sanitize"]
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ]
    self.system_prompt = SYSTEM_PROMPT

  def _parse_checker_json(self, output_text: str) -> dict:
    text = output_text.strip()
    if self.sanitize:
      if text.startswith("```json"):
        text = text[7:].strip()
        if text.endswith("```"):
          text = text[:-3].strip()
      elif text.startswith("```"):
        text = text[3:].strip()
        if text.endswith("```"):
          text = text[:-3].strip()
    else:
      json_start = text.find("{")
      json_end = text.rfind("}") + 1
      if json_start != -1 and json_end > json_start:
        text = text[json_start:json_end]

    parsed = json.loads(text.strip())
    if not isinstance(parsed, dict):
      raise ValueError(f"Expected dict, got {type(parsed)}")
    if "good_copy" not in parsed or "info_correct" not in parsed:
      raise ValueError("Missing required keys: good_copy, info_correct")
    if "eval_text" not in parsed:
      parsed["eval_text"] = ""
    return parsed

  def generate_response(self, request_text: str) -> dict:
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=request_text)])]
    config_kwargs = dict(
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
    )
    if self.json_output:
      config_kwargs["response_mime_type"] = "application/json"
    generate_content_config = types.GenerateContentConfig(**config_kwargs)

    output_text = ""
    thought_summary = ""
    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=generate_content_config,
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
      print("=" * 80)
      print("THOUGHT SUMMARY:")
      print(thought_summary.strip())
      print("=" * 80)
      print()

    if not output_text or not output_text.strip():
      raise ValueError("Empty response from model. Check API key and model availability.")

    try:
      parsed = self._parse_checker_json(output_text)
      parsed["thought_summary"] = thought_summary.strip()
      return parsed
    except json.JSONDecodeError as e:
      raise ValueError(
        f"Failed to parse JSON response: {e}\nResponse length: {len(output_text)}\nResponse preview: {output_text[:500]}"
      )


TEST_CASES = [
  {
    "name": "correct_past_two_months_groceries",
    "last_user_request": "How much did I spend on groceries in the past 2 months?",
    "previous_conversation": None,
    "generated_code": """def process_input():
    today = datetime.now()
    start = get_start_of_month(get_after_periods(today, 'monthly', -2))
    end = get_end_of_month(get_after_periods(today, 'monthly', -1))
    sp = retrieve_spending_transactions()
    if sp.empty:
        return True, "No spending transactions found."
    sp = sp[(sp['date'] >= start) & (sp['date'] <= end)]
    if sp.empty:
        return True, "No grocery spending found in the past 2 months."
    sp = sp[sp['category'] == 'meals_groceries']
    if sp.empty:
        return True, "No grocery spending found in the past 2 months."
    output_lines = []
    output_lines.append("Groceries — past 2 full calendar months (current month excluded).")
    output_lines.append(utter_transaction_total(sp, "Total: {spending_total_amount}"))
    return True, chr(10).join(output_lines)""",
    "execution_result": """process_input:
success: True
return: (True, "Groceries — past 2 full calendar months (current month excluded).\\nTotal: $240.00")

(sandbox logs empty)""",
    "past_review_outcomes": None,
    "output": {
      "good_copy": True,
      "info_correct": True,
      "eval_text": "",
    },
  },
  {
    "name": "bad_exact_amount_without_fuzzy_band",
    "last_user_request": "How many $50 AT&T payments have I had this year?",
    "previous_conversation": None,
    "generated_code": """def process_input():
    start = get_start_of_year(datetime.now())
    end = get_end_of_year(start)
    sp = retrieve_spending_transactions()
    if sp.empty:
        return True, "No spending transactions found this year."
    sp = sp[(sp['date'] >= start) & (sp['date'] <= end)]
    sp = sp[
        sp['transaction_name'].str.lower().str.contains('at&t', na=False)
        & (sp['amount'] == 50)
    ]
    return True, f"Count: {len(sp)}" """,
    "execution_result": """process_input:
success: True
return: (True, "Count: 0")

(sandbox logs empty)""",
    "past_review_outcomes": None,
    "output": {
      "good_copy": True,
      "info_correct": False,
      "eval_text": "Uses exact $50 match instead of the 49.50–50.49 whole-dollar band.",
    },
  },
  {
    "name": "bad_str_on_possibly_empty_frame",
    "last_user_request": "Show my transportation spending in the past 4 weeks.",
    "previous_conversation": None,
    "generated_code": """def process_input():
    start = get_start_of_week(get_after_periods(datetime.now(), 'weekly', -4))
    end = get_end_of_week(get_after_periods(datetime.now(), 'weekly', -1))
    sp = retrieve_spending_transactions()
    sp = sp[(sp['date'] >= start) & (sp['date'] <= end)]
    sp = sp[sp['category'].str.startswith('transportation')]
    if sp.empty:
        return True, "No transportation spending in the past 4 weeks."
    return True, utter_transaction_total(sp, "Total: {spending_total_amount}")""",
    "execution_result": """process_input:
success: False
**Execution Error**: cannot access str on empty frame (example)

return: (False, "**Execution Error**: ...")""",
    "past_review_outcomes": None,
    "output": {
      "good_copy": False,
      "info_correct": False,
      "eval_text": "Missing empty check before .str filter; execution crashed.",
    },
  },
  {
    "name": "unsupported_place_stock_trade",
    "last_user_request": "Place a market order for 10 shares of AAPL from my checking account.",
    "previous_conversation": None,
    "generated_code": """def process_input():
    return False, "I cannot place stock trades. I can show balances, transactions, forecasts, or subscriptions." """,
    "execution_result": """process_input:
success: True
return: (False, "I cannot place stock trades. I can show balances, transactions, forecasts, or subscriptions.")""",
    "past_review_outcomes": None,
    "output": {
      "good_copy": True,
      "info_correct": True,
      "eval_text": "",
    },
  },
  {
    "name": "incorrect_success_with_traceback",
    "last_user_request": "What is my net worth right now?",
    "previous_conversation": None,
    "generated_code": """def process_input():
    dep = retrieve_depository_accounts()
    cre = retrieve_credit_accounts()
    if dep.empty and cre.empty:
        return True, "No accounts."
    tot_d = dep['balance_current'].sum()
    tot_c = cre['balance_current'].sum()
    return True, f"Net worth error path", tot_d - tot_c  # malformed: illustrative bad pattern""",
    "execution_result": """process_input:
success: True
return: (True, "**Execution Error**: process_input returned invalid tuple length")

stderr: TypeError: ...""",
    "past_review_outcomes": None,
    "output": {
      "good_copy": False,
      "info_correct": False,
      "eval_text": "Returns a 3-tuple instead of (bool, str); execution error.",
    },
  },
  {
    "name": "incorrect_failure_on_success",
    "last_user_request": "List all my accounts.",
    "previous_conversation": None,
    "generated_code": """def process_input():
    dep = retrieve_depository_accounts()
    cre = retrieve_credit_accounts()
    if dep.empty and cre.empty:
        return False, "No accounts found."
    output_lines = []
    if not dep.empty:
        output_lines.append("Depository:")
        output_lines.append(account_names_and_balances(dep, "- {account_name}"))
    if not cre.empty:
        output_lines.append("Credit:")
        output_lines.append(account_names_and_balances(cre, "- {account_name}"))
    return True, chr(10).join(output_lines)""",
    "execution_result": """process_input:
success: True
return: (False, "- Checking ...\\n- Savings ...")

Note: execution incorrectly returned False despite successful listing.""",
    "past_review_outcomes": None,
    "output": {
      "good_copy": False,
      "info_correct": True,
      "eval_text": "Returned False while listing accounts; missing Depository/Credit headers.",
    },
  },
  {
    "name": "past_review_fuzzy_amount_still_wrong",
    "last_user_request": "How many $50 AT&T payments have I had this year?",
    "previous_conversation": None,
    "generated_code": """def process_input():
    start = get_start_of_year(datetime.now())
    end = get_end_of_year(start)
    sp = retrieve_spending_transactions()
    if sp.empty:
        return True, "No spending transactions found this year."
    sp = sp[(sp['date'] >= start) & (sp['date'] <= end)]
    sp = sp[
        sp['transaction_name'].str.lower().str.contains('at&t', na=False)
        & (sp['amount'] == 50.0)
    ]
    return True, f"There have been {len(sp)} payments."
""",
    "execution_result": """process_input:
success: True
return: (True, "There have been 0 payments.")""",
    "past_review_outcomes": "self",
    "output": {
      "good_copy": True,
      "info_correct": False,
      "eval_text": "Exact amount equality violates the whole-dollar fuzzy band for $50.",
    },
  },
]

BATCHES: dict[int, dict[str, object]] = {
  1: {
    "name": "Good grocery spend (past 2 months)",
    "tests": [0],
  },
  2: {
    "name": "Fuzzy band + empty guard",
    "tests": [1, 2],
  },
  3: {
    "name": "Unsupported + past review fuzzy",
    "tests": [3, 6],
  },
  4: {
    "name": "Tuple error + bool mismatch",
    "tests": [4, 5],
  },
}


def _normalize_past_review_item(item: dict) -> dict:
  out = item.get("output") if isinstance(item.get("output"), dict) else item
  return {
    "generated_code": item.get("generated_code", ""),
    "execution_result": item.get("execution_result", ""),
    "good_copy": out.get("good_copy"),
    "info_correct": out.get("info_correct"),
    "eval_text": out.get("eval_text", ""),
  }


def get_test_case(test_name_or_index):
  if isinstance(test_name_or_index, int):
    if 0 <= test_name_or_index < len(TEST_CASES):
      return TEST_CASES[test_name_or_index]
    return None
  if isinstance(test_name_or_index, str):
    for test_case in TEST_CASES:
      if test_case["name"] == test_name_or_index:
        return test_case
  return None


def _resolve_past_review_outcomes(tc: dict) -> Optional[list]:
  past = tc.get("past_review_outcomes")
  if not past:
    return None
  if past == "self":
    return [
      {
        "generated_code": tc["generated_code"],
        "execution_result": tc["execution_result"],
        "output": {
          "good_copy": True,
          "info_correct": False,
          "eval_text": "Exact amount equality violates the whole-dollar fuzzy band for $50.",
        },
      }
    ]
  return past


def _run_test_with_logging(tc: dict, checker: Optional[CheckLookupUserDataOptimizer] = None):
  if checker is None:
    checker = CheckLookupUserDataOptimizer()

  last_user_request = tc["last_user_request"]
  previous_conversation = tc.get("previous_conversation")
  generated_code = tc["generated_code"]
  execution_result = tc["execution_result"]
  past_review_outcomes = _resolve_past_review_outcomes(tc)

  print("\n" + "=" * 80)
  print(f"Running checker test: {tc['name']}")
  print("=" * 80)

  past_review_section = ""
  if past_review_outcomes:
    for index, raw in enumerate(past_review_outcomes):
      past = _normalize_past_review_item(raw)
      past_review_section += f"""<PAST_REVIEW_OUTCOME_{index + 1}>

## Generated Code for #{index + 1}

```python
{past['generated_code']}
```

## Execution Result for #{index + 1}

{past['execution_result']}

## Evaluation Output for #{index + 1}

```json
{json.dumps({"good_copy": past["good_copy"], "info_correct": past["info_correct"], "eval_text": past["eval_text"]}, indent=2)}
```

</PAST_REVIEW_OUTCOME_{index + 1}>

"""

  if previous_conversation:
    prev_block = f"""**Previous Conversation**:
{previous_conversation}"""
  else:
    prev_block = """**Previous Conversation**:
None"""

  request_str = f"""<EVAL_INPUT>
**Last User Request**: {last_user_request}
{prev_block}
</EVAL_INPUT>

<GENERATED_CODE>

```python
{generated_code}
```

</GENERATED_CODE>

<EXECUTION_RESULT>

{execution_result}

</EXECUTION_RESULT>

{past_review_section}

Output:"""

  print("LLM INPUT:")
  print("=" * 80)
  print(request_str)
  print("=" * 80)
  print()

  try:
    result = checker.generate_response(request_str)
    ideal = tc["output"]
    llm_output = {
      "good_copy": result.get("good_copy"),
      "info_correct": result.get("info_correct"),
      "eval_text": result.get("eval_text", ""),
    }
    print("=" * 80)
    print("LLM OUTPUT:")
    print("=" * 80)
    print(json.dumps(llm_output, indent=2))
    print("=" * 80)
    print()
    print("IDEAL OUTPUT:")
    print("=" * 80)
    print(json.dumps(ideal, indent=2))
    print("=" * 80)
    print()
    flags_match = (
      llm_output["good_copy"] == ideal["good_copy"]
      and llm_output["info_correct"] == ideal["info_correct"]
    )
    print(f"Flags match ideal: {flags_match}")
    print("=" * 80)
    print()
    return result
  except Exception as e:
    print(f"ERROR: {type(e).__name__}: {str(e)}")
    import traceback
    print(traceback.format_exc())
    print("=" * 80)
    print()
    return None


def run_test(test_name_or_index_or_dict, checker: Optional[CheckLookupUserDataOptimizer] = None):
  if isinstance(test_name_or_index_or_dict, dict):
    tc = test_name_or_index_or_dict
  else:
    tc = get_test_case(test_name_or_index_or_dict)
  if tc is None:
    print(f"Test case '{test_name_or_index_or_dict}' not found.")
    return None
  return _run_test_with_logging(tc, checker)


def run_tests(test_names_or_indices=None, checker: Optional[CheckLookupUserDataOptimizer] = None):
  if test_names_or_indices is None:
    test_names_or_indices = list(range(len(TEST_CASES)))
  results = []
  for item in test_names_or_indices:
    results.append(run_test(item, checker=checker))
  return results


def main(batch: int = None, test: str = None):
  print("Testing CheckLookupUserDataOptimizer\n")
  checker = CheckLookupUserDataOptimizer()

  if batch is not None:
    if batch not in BATCHES:
      print(f"Invalid batch: {batch}. Available: {sorted(BATCHES.keys())}")
      for b, info in BATCHES.items():
        names = [TEST_CASES[i]["name"] for i in info["tests"]]
        print(f"  Batch {b}: {info['name']} — {', '.join(names)}")
      return
    info = BATCHES[batch]
    print(f"Batch {batch}: {info['name']}\n")
    run_tests(test_names_or_indices=info["tests"], checker=checker)
    print("All tests completed!")
    return

  if test is not None:
    test_key = int(test) if test.isdigit() else test
    run_test(test_key, checker=checker)
    return

  print("Available test cases:")
  for i, tc in enumerate(TEST_CASES):
    print(f"  {i}: {tc['name']}")
  print("\nBatches:")
  for b, info in BATCHES.items():
    names = [TEST_CASES[i]["name"] for i in info["tests"]]
    print(f"  {b}: {info['name']} — {', '.join(names)}")


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Run lookup user data checker tests")
  parser.add_argument(
    "--batch",
    type=int,
    default=None,
    choices=[1, 2, 3, 4],
    help="Batch number (1–4)",
  )
  parser.add_argument(
    "--test",
    type=str,
    default=None,
    help="Test name or index (e.g. correct_past_two_months_groceries or 0)",
  )
  args = parser.parse_args()
  main(batch=args.batch, test=args.test)
