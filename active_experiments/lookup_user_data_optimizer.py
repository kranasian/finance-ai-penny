"""Lookup user data optimizer: Gemini-driven ``process_input`` generation (P:Func:LookupUserData).

Run from the ``finance-ai-penny`` repository root with ``GEMINI_API_KEY`` set (e.g. in ``.env``)::

  python active_experiments/lookup_user_data_optimizer.py --test total_spent_current_month
  python active_experiments/lookup_user_data_optimizer.py --test five_most_recent_transactions
  python active_experiments/lookup_user_data_optimizer.py --test all
  python active_experiments/lookup_user_data_optimizer.py --batch half_dollar_amount_band_boundaries
  python active_experiments/lookup_user_data_optimizer.py --test 0 --no-thinking

"""
from google import genai
from google.genai import types
from google.genai.errors import ClientError
import os
import re
import sys
from datetime import datetime
from dotenv import load_dotenv

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

import sandbox
from database import Database

load_dotenv()

RUN_SETTINGS = {
  "json": False,
  "sanitize": False,
  "gen_config": {
    "top_k": 40,
    "top_p": 0.95,
    "temperature": 0.35,
    "thinking_budget": 0,
    "max_output_tokens": 2048,
  },
  "model_name": "gemini-flash-lite-latest",
}

SYSTEM_PROMPT = """
Your name is "Penny" and you are a helpful AI specialized in code generation. **You only output python code.**

## Task & Rules
1. **Output**: Write `process_input() -> tuple[bool, str]`. Keep code concise. Minimal comments.
2. **Tools**: Use only `IMPLEMENTED_FUNCTIONS` & `IMPLEMENTED_DATE_FUNCTIONS`. **Do not** add ``import`` lines; ``pd``, ``datetime``, and the listed helpers are already in scope.
3. **Matching**
   - Use partial case-insensitive matching for Account/Subscription names.
   - **Whole-dollar *N* vs ``amount``**: The user’s dollar figure is usually a whole number; stored ``amount`` may differ by cents. **Forbidden**: ``amount == N`` (or any equality tied only to *N*). **Required**: half-dollar band, **lower inclusive / upper exclusive**: ``(df['amount'] >= N - 0.5) & (df['amount'] < N + 0.5)`` with *N* parsed as a number. Examples: *N* = 85 → ``>= 84.5`` and ``< 85.5``; *N* = 86 → ``>= 85.5`` and ``< 86.5``. **Wrong** for *N* = 85: ``>= 85.0`` and ``< 86.0`` (that is a different band shifted by $0.50).
4. **Safety**: Always check `if df.empty: ...` before accessing data.
5. **Output Collection**: Accumulate all output strings in a list and join with newlines. Return the joined string as the second element of the tuple.
6. **Date Ranges**: When referring to "past/next n months/weeks", **always exclude the current month/week**.
   - **Past n months**: Start = `get_start_of_month(get_after_periods(today, 'monthly', -n))`, End = `get_end_of_month(get_after_periods(today, 'monthly', -1))`.
   - **Past n weeks**: Start = `get_start_of_week(get_after_periods(today, 'weekly', -n))`, End = `get_end_of_week(get_after_periods(today, 'weekly', -1))`.
   - **Next n months**: Start = `get_start_of_month(get_after_periods(today, 'monthly', 1))`, End = `get_end_of_month(get_after_periods(today, 'monthly', n))`.
   - **Next n weeks**: Start = `get_start_of_week(get_after_periods(today, 'weekly', 1))`, End = `get_end_of_week(get_after_periods(today, 'weekly', n))`.

Today: |TODAY_DATE|.

<IMPLEMENTED_FUNCTIONS>

- `retrieve_depository_accounts() -> pd.DataFrame`: checking, savings. Cols: account_type, account_name, balance_available, balance_current
- `retrieve_credit_accounts() -> pd.DataFrame`: credit, loans. Cols: account_type, account_name, balance_available, balance_current, balance_limit
- `account_names_and_balances(df, template) -> str`: Format account list. Placeholders: any df col.  When listing accounts, include account_id in the template.
- `retrieve_income_transactions() -> pd.DataFrame`: Past income. Cols: date, transaction_name, amount, category, transaction_id, account_id
- `retrieve_spending_transactions() -> pd.DataFrame`: Past spending. Cols: date, transaction_name, amount, category, transaction_id, account_id
- `transaction_names_and_amounts(df, template) -> str`: List transactions. Placeholders: any df col.  When listing transactions, include account_id and transaction_id in the template.
- `utter_transaction_total(df, template) -> str`: Sum transactions then use correct placeholder for income or spending. Placeholders: {income_total_amount}, {spending_total_amount}. Example: "Total income: {income_total_amount}" or "Total spending: {spending_total_amount}"
- `retrieve_spending_forecasts(granularity='monthly') -> pd.DataFrame`: Future spending by week/month. Cols: start_date, forecasted_amount, category
- `retrieve_income_forecasts(granularity='monthly') -> pd.DataFrame`: Future income by week/month. Cols: start_date, forecasted_amount, category
- `forecast_dates_and_amount(df, template) -> str`: List forecasts. Placeholders: any df col.
- `utter_forecast_amount(amount, template) -> str`: Format forecast amount. Placeholders: {income_total_amount}, {spending_total_amount}. Example: "Total Expected Income: {income_total_amount}" or "Spending: {spending_total_amount}"
- `utter_absolute_amount(amount, template) -> str`: Format absolute amount. Placeholders: {amount}, {amount_with_direction}, {income_total_amount}, {spending_total_amount}
- `retrieve_subscriptions() -> pd.DataFrame`: Cols: transaction_name, amount, category, subscription_name, date
- `subscription_names_and_amounts(df, template) -> str`: List subscriptions. Placeholders: any df col.
- `utter_subscription_totals(df, template) -> str`: Sum subscriptions. Placeholders: {total_amount}. Example: "Total subscriptions: {total_amount}"

</IMPLEMENTED_FUNCTIONS>

<IMPLEMENTED_DATE_FUNCTIONS>

- `get_date(y, m, d)`, `get_start_of_month(date)`, `get_end_of_month(date)`
- `get_start_of_year(date)`, `get_end_of_year(date)`
- `get_start_of_week(date)`, `get_end_of_week(date)`
- `get_after_periods(date, granularity, count)`, `get_date_string(date)`

</IMPLEMENTED_DATE_FUNCTIONS>

<ACCOUNT_TYPE>

deposit_savings, deposit_money_market, deposit_checking, credit_card, loan_home_equity, loan_line_of_credit, loan_mortgage, loan_auto

</ACCOUNT_TYPE>

<CATEGORY>

- `income`: salary, bonuses, interest, side hussles. (`income_salary`, `income_sidegig`, `income_business`, `income_interest`)
- `meals`: food spending. (`meals_groceries`, `meals_dining_out`, `meals_delivered_food`)
- `leisure`: recreation/travel. (`leisure_entertainment`, `leisure_travel`)
- `bills`: recurring costs. (`bills_connectivity`, `bills_insurance`, `bills_tax`, `bills_service_fees`)
- `shelter`: housing. (`shelter_home`, `shelter_utilities`, `shelter_upkeep`)
- `education`: learning/kids. (`education_kids_activities`, `education_tuition`)
- `shopping`: discretionary. (`shopping_clothing`, `shopping_gadgets`, `shopping_kids`, `shopping_pets`)
- `transportation`: car/public. (`transportation_public`, `transportation_car`)
- `health`: medical/wellness. (`health_medical_pharmacy`, `health_gym_wellness`, `health_personal_care`)
- `donations_gifts`: charity/gifts.
- `uncategorized`: unknown.
- `transfers`: internal movements.
- `miscellaneous`: other.

</CATEGORY>

<EXAMPLES>

input: User: What is my net worth and checking balance?
output:
```python
def process_input():
    output_lines = []
    assets = retrieve_depository_accounts()
    credit = retrieve_credit_accounts()
    if assets.empty and credit.empty:
        return True, "No accounts."

    tot_a = assets['balance_current'].sum() if not assets.empty else 0
    tot_l = credit['balance_current'].sum() if not credit.empty else 0
    net_worth = tot_a - tot_l
    output_lines.append(f"Assets: {utter_absolute_amount(tot_a, '{amount_with_direction}')}")
    output_lines.append(f"Liabilities: {utter_absolute_amount(tot_l, '{amount_with_direction}')}")
    output_lines.append(f"Net Worth: {utter_absolute_amount(net_worth, '{amount_with_direction}')}")

    if not assets.empty:
        chk = assets[assets['account_type'] == 'deposit_checking']
        if not chk.empty:
            output_lines.append("Checking:")
            output_lines.append(account_names_and_balances(chk, "Account '{account_name}' (account_id: {account_id}) has {balance_current} left with {balance_available} available now."))
    return True, chr(10).join(output_lines)
```

input: User: Can I afford $200 for concert tickets next week and what was my past few fun things I spent on?
output:
```python
def process_input():
    output_lines = []
    liq = retrieve_depository_accounts()
    cash = liq['balance_available'].sum() if not liq.empty else 0
    
    today = datetime.now()
    sp = retrieve_spending_forecasts(granularity='weekly')
    
    nxt = get_start_of_week(get_after_periods(today, 'weekly', 1))
    sp = sp[sp['start_date'] == nxt] if not sp.empty else pd.DataFrame()
    obligations = sp['forecasted_amount'].sum() if not sp.empty else 0
    
    rem = cash - obligations - 200
    
    output_lines.append(f"Cash: {utter_absolute_amount(cash, '{amount_with_direction}')}")
    output_lines.append(f"Next Week Expenses: {utter_absolute_amount(obligations, '{spending_total_amount}')}")
    
    if rem >= 0:
        output_lines.append(f"Yes. You'll have {utter_absolute_amount(rem, '{amount_with_direction}')} left.")
    else:
        output_lines.append(f"No. You'll be short by {utter_absolute_amount(rem, '{amount_with_direction}')}.")
    
    past_spending = retrieve_spending_transactions()
    if not past_spending.empty:
        fun_spending = past_spending[(past_spending['category'].str.startswith('leisure'))]
        if not fun_spending.empty:
            fun_spending = fun_spending.sort_values('date', ascending=False).head(5)
            output_lines.append("Recent fun spending:")
            output_lines.append(transaction_names_and_amounts(fun_spending, "- {transaction_name}: {amount} on {date} in {category} (account_id: {account_id}, transaction_id: {transaction_id})"))
        else:
            output_lines.append("No recent fun spending found.")
    else:
        output_lines.append("No past spending data available.")
    
    return True, chr(10).join(output_lines)
```

input: User: List subscriptions paid last month.
output:
```python
def process_input():
    output_lines = []
    subs = retrieve_subscriptions()
    if subs.empty:
        return True, "No subscriptions."
    
    start = get_start_of_month(get_after_periods(datetime.now(), 'monthly', -1))
    end = get_end_of_month(start)
    
    subs = subs[(subs['date'] >= start) & (subs['date'] <= end)]
    if subs.empty:
        return True, "No subscriptions paid last month."
        
    output_lines.append("Subscriptions paid last month:")
    output_lines.append(subscription_names_and_amounts(subs, "- {amount} {subscription_name} on {date}."))
    output_lines.append(utter_subscription_totals(subs, "Total: {total_amount}"))
    return True, chr(10).join(output_lines)
```

input: User: How did my income compare to my discretionary spending last year?
output:
```python
def process_input():
    output_lines = []
    start = get_start_of_year(get_after_periods(datetime.now(), 'yearly', -1))
    end = get_end_of_year(start)
    
    inc = retrieve_income_transactions()
    sp = retrieve_spending_transactions()
    
    inc = inc[(inc['date'] >= start) & (inc['date'] <= end)] if not inc.empty else pd.DataFrame()
    sp = sp[(sp['date'] >= start) & (sp['date'] <= end)] if not sp.empty else pd.DataFrame()
    
    if not sp.empty:
        sp = sp[sp['category'].str.startswith('meals') | sp['category'].str.startswith('shopping') | sp['category'].str.startswith('leisure') | (sp['category'] == 'donations_gifts')]
    else:
        sp = pd.DataFrame()
    
    if inc.empty and sp.empty:
        return True, "No data last year."
        
    i_val = inc['amount'].sum() if not inc.empty else 0
    s_val = sp['amount'].sum() if not sp.empty else 0
    diff = i_val - s_val
    
    output_lines.append(f"Last Year ({start.year}):")
    output_lines.append(f"Income: {utter_absolute_amount(i_val, '{amount}')}")
    output_lines.append("Discretionary Spending:")
    
    if not sp.empty:
        for cat in sorted(sp['category'].unique()):
            cat_sp = sp[sp['category'] == cat]
            cat_val = cat_sp['amount'].sum()
            output_lines.append(f"  - {cat}: {utter_absolute_amount(cat_val, '{amount}')}")
    else:
        output_lines.append("  (No discretionary spending)")
    
    output_lines.append(f"Total Discretionary: {utter_absolute_amount(s_val, '{amount}')}")
    output_lines.append(f"Net: {utter_absolute_amount(diff, '{amount_with_direction}')}")
    return True, chr(10).join(output_lines)
```

input: User: Project my savings for the next 4 weeks.
output:
```python
def process_input():
    output_lines = []
    today = datetime.now()
    inc = retrieve_income_forecasts('weekly')
    sp = retrieve_spending_forecasts('weekly')
    
    if inc.empty and sp.empty:
        return True, "No forecasts."
    
    start_next = get_start_of_week(get_after_periods(today, 'weekly', 1))
    dates = [get_after_periods(start_next, 'weekly', i) for i in range(4)]
    
    total_sav = 0
    output_lines.append("Projected Savings (Next 4 Weeks):")
    
    for d in dates:
        i_wk = inc[inc['start_date'] == d]['forecasted_amount'].sum() if not inc.empty else 0
        s_wk = sp[sp['start_date'] == d]['forecasted_amount'].sum() if not sp.empty else 0
        sav = i_wk - s_wk
        total_sav += sav
        d_str = get_date_string(d)
        output_lines.append(f"- Wk of {d_str}: Income {utter_absolute_amount(i_wk, '{income_total_amount}')} - Spending {utter_absolute_amount(s_wk, '{spending_total_amount}')} = Savings {utter_absolute_amount(sav, '{amount_with_direction}')}")
        
    output_lines.append(f"Total Projected Savings: {utter_absolute_amount(total_sav, '{amount_with_direction}')}")
    return True, chr(10).join(output_lines)
```

</EXAMPLES>

<ACCOUNT_NAMES>

|ACCOUNT_NAMES|

</ACCOUNT_NAMES>

<SUBSCRIPTION_NAMES>

|SUBSCRIPTION_NAMES|

</SUBSCRIPTION_NAMES>
"""

class LookupUserDataOptimizer:
  """Gemini API for P:Func:LookupUserData (``process_input`` code generation)."""

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
        "GEMINI_API_KEY environment variable is not set. Set it in .env or environment."
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

  def _build_account_names_section(self, user_id: int) -> str:
    db = Database()
    accounts = db.get_accounts_by_user(user_id)
    if not accounts:
      return "    (No accounts found for this user)"
    names = []
    for account in accounts:
      account_name = account.get("account_name", "")
      if account_name:
        names.append(f"    - `{account_name}`")
    return "\n".join(names) if names else "    (No account names found)"

  def _build_subscription_names_section(self, user_id: int) -> str:
    db = Database()
    rows = db.get_subscription_transactions(user_id, confidence_score_bills_threshold=0.5)
    if not rows:
      return "    (No subscriptions found for this user)"
    seen = set()
    for transaction in rows:
      sn = transaction.get("subscription_name", "")
      if sn:
        seen.add(sn)
    if not seen:
      return "    (No subscription names found)"
    return "\n".join(f"    - `{name}`" for name in sorted(seen))

  def generate_response(
    self,
    last_user_request: str,
    previous_conversation: str = "",
    user_id: int = 1,
    replacements: dict = None,
    include_thoughts: bool = True,
  ) -> str:
    system_prompt = self.system_prompt
    if replacements:
      for key, value in replacements.items():
        system_prompt = system_prompt.replace(f"|{key}|", str(value))
    today = datetime.now().strftime("%B %d, %Y")
    system_prompt = system_prompt.replace("|TODAY_DATE|", today)
    account_names_section = self._build_account_names_section(user_id)
    subscription_names_section = self._build_subscription_names_section(user_id)
    system_prompt = system_prompt.replace("|ACCOUNT_NAMES|", account_names_section)
    system_prompt = system_prompt.replace("|SUBSCRIPTION_NAMES|", subscription_names_section)
    if previous_conversation and previous_conversation.strip():
      user_block = (
        f"User: {last_user_request}\n**Input Information**:\n\n{previous_conversation.strip()}"
      )
    else:
      user_block = f"User: {last_user_request}"
    request_text = types.Part.from_text(text=f"""{user_block}

output:""")
    contents = [types.Content(role="user", parts=[request_text])]
    generate_content_config = types.GenerateContentConfig(
      temperature=self.temperature,
      top_p=self.top_p,
      top_k=self.top_k,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=system_prompt)],
      thinking_config=types.ThinkingConfig(
        thinking_budget=self.thinking_budget,
        include_thoughts=include_thoughts,
      ),
    )
    output_text = ""
    thought_summary = ""
    try:
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
                  if getattr(part, "thought", False) and getattr(part, "text", None):
                    thought_summary = (
                      (thought_summary + part.text)
                      if thought_summary
                      else part.text
                    )
    except ClientError as e:
      if self.thinking_budget == 0 and "only works in thinking mode" in (str(e) or ""):
        print(
          "\n[NOTE] This model requires thinking mode; API rejected thinking_budget=0. "
          "Use --no-thinking off (default uses include_thoughts=True with budget 0) or another model.",
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
    return output_text


def extract_python_code(text: str) -> str:
  """Extract Python code from generated response (```python blocks)."""
  code_start = text.find("```python")
  if code_start != -1:
    code_start += len("```python")
    code_end = text.find("```", code_start)
    if code_end != -1:
      return text[code_start:code_end].strip()
    return text[code_start:].strip()
  return text.strip()


def _get_heavy_data_user_id() -> int:
  try:
    db = Database()
    heavy_user = db.get_user("HeavyDataUser")
    if heavy_user and "id" in heavy_user:
      return heavy_user["id"]
    print("Warning: HeavyDataUser not found, using default user_id=1")
    return 1
  except Exception as e:
    print(f"Warning: Error getting HeavyDataUser: {e}, using default user_id=1")
    return 1


# Slugs that appear in ``return ``...`` `` rubrics for amount-band fixtures (heuristic scoring).
_RUBRIC_SCORED_CATEGORY_SLUGS = frozenset(
  {
    "shopping_clothing",
    "meals_groceries",
    "meals_dining_out",
    "meals_delivered_food",
    "bills_service_fees",
    "transportation_car",
  }
)


def _rubric_expected_category_slugs(rubric: str) -> list[str]:
  """Return ordered unique category slugs the rubric says the sandbox should mention."""
  found: list[str] = []
  for m in re.finditer(r"return\s+``([a-z][a-z0-9_]{2,50})``", rubric, re.IGNORECASE):
    slug = m.group(1).lower()
    if slug in _RUBRIC_SCORED_CATEGORY_SLUGS and slug not in found:
      found.append(slug)
  return found


def _rubric_slug_token_in_text(text: str, slug: str) -> bool:
  """True if ``slug`` appears as a whole token (avoids ``car`` matching inside ``shell``)."""
  return bool(
    re.search(rf"(?<![a-z0-9_]){re.escape(slug)}(?![a-z0-9_])", text, re.IGNORECASE)
  )


def _rubric_expects_no_match_response(rubric: str) -> bool:
  """True when the rubric describes an empty / not-found outcome (no category to return)."""
  if _rubric_expected_category_slugs(rubric):
    return False
  s = rubric.lower()
  markers = (
    "no matching row",
    "no band match",
    "no match for user",
    "clear empty / not-found",
    "empty / not-found",
    "not-found or explain",
    "below the band",
    "below**",
    "outside ``[",
  )
  return any(m in s for m in markers)


def rubric_matches_sandbox_output(expected_rubric: str, sandbox_output: str):
  """Compare sandbox user-facing text to the test ``output`` rubric (heuristic).

  Returns ``True`` / ``False`` when auto-scored, or ``None`` when the rubric has no scorable signals.
  """
  if not (expected_rubric or "").strip():
    return None
  out = (sandbox_output or "").lower()
  expected_slugs = _rubric_expected_category_slugs(expected_rubric)
  for slug in expected_slugs:
    if _rubric_slug_token_in_text(out, slug):
      return True
  if expected_slugs:
    return False
  if _rubric_expects_no_match_response(expected_rubric):
    if re.search(
      r"\b(no matching|not found|no transaction|couldn'?t find|could not find|"
      r"no row|unable to find|doesn'?t match|nothing matching|was not found|"
      r"didn'?t find)\b",
      out,
    ):
      return True
    if re.match(r"^\s*no\b", out):
      return True
    if re.search(r"\bno\b[\w\s,'$.\-]{0,160}\bfound\b", out):
      return True
    for slug in _RUBRIC_SCORED_CATEGORY_SLUGS:
      if _rubric_slug_token_in_text(out, slug):
        return False
    return False
  return None


def print_rubric_vs_sandbox_emoji(expected_rubric: str, sandbox_output: str) -> None:
  """Print ✅ or ❌ when ``rubric_matches_sandbox_output`` can score this rubric."""
  verdict = rubric_matches_sandbox_output(expected_rubric, sandbox_output)
  if verdict is None:
    return
  print(f"Expected rubric vs sandbox: {'✅' if verdict else '❌'}")


def _maybe_ensure_lur_amount_band_fixtures(test_name: str) -> None:
  if not test_name.startswith("lur_band_"):
    return
  from user_seeder import ensure_lookup_amount_band_fixtures

  ensure_lookup_amount_band_fixtures()


def _run_test_with_logging(
  last_user_request: str,
  previous_conversation: str,
  optimizer: "LookupUserDataOptimizer" = None,
  user_id: int = None,
  replacements: dict = None,
  include_thoughts: bool = True,
  expected_output_rubric: str = None,
):
  if optimizer is None:
    optimizer = LookupUserDataOptimizer()
  if user_id is None:
    user_id = _get_heavy_data_user_id()
  if previous_conversation and previous_conversation.strip():
    llm_input = (
      f"User: {last_user_request}\n**Input Information**:\n\n{previous_conversation.strip()}"
    )
  else:
    llm_input = f"User: {last_user_request}"
  print("=" * 80)
  print("LLM INPUT:")
  print("=" * 80)
  print(llm_input)
  print("=" * 80 + "\n")
  print(
    f"(run settings: json={optimizer.json_output}, sanitize={optimizer.sanitize}, "
    f"model={optimizer.model_name}, thinking_budget={optimizer.thinking_budget}, "
    f"max_output_tokens={optimizer.max_output_tokens})\n"
  )
  result = optimizer.generate_response(
    last_user_request,
    previous_conversation,
    user_id=user_id,
    replacements=replacements,
    include_thoughts=include_thoughts,
  )
  print("=" * 80)
  print("LLM OUTPUT:")
  print("=" * 80)
  print(result)
  print("=" * 80 + "\n")
  code = extract_python_code(result)
  if code:
    print("=" * 80)
    print("SANDBOX EXECUTION (extracted ```python):")
    print("=" * 80)
    output_string = ""
    try:
      success, output_string, logs, goals_list = sandbox.execute_agent_with_tools(
        result, user_id
      )
      print(f"Success: {success}")
      print("Output:")
      print("-" * 80)
      print(output_string)
      print("-" * 80)
      print_rubric_vs_sandbox_emoji(expected_output_rubric or "", output_string)
    except Exception as e:
      print(f"**Sandbox Execution Error**: {str(e)}")
      import traceback
      print(traceback.format_exc())
      print_rubric_vs_sandbox_emoji(expected_output_rubric or "", str(e))
    print("=" * 80 + "\n")
  return result


def parse_planner_test_input(raw: str) -> tuple[str, str]:
  """Split static test input into last user request and prior context (Input Information)."""
  s = raw.strip()
  lur_tag = "**Last User Request**:"
  pc_tag = "**Previous Conversation**:"
  if lur_tag not in s or pc_tag not in s:
    raise ValueError(
      f"Test input must contain literal markers {lur_tag!r} and {pc_tag!r}"
    )
  _, after_lur = s.split(lur_tag, 1)
  if pc_tag not in after_lur:
    raise ValueError(f"Test input must contain {pc_tag!r} after the last user request")
  last_user_request, previous_conversation = after_lur.split(pc_tag, 1)
  return last_user_request.strip(), previous_conversation.strip()


TEST_CASES = [
  {
    "name": "total_spent_current_month",
    "batch": "spending_current_month_total",
    "input": """**Last User Request**: Calculate the total amount I have spent this current month.
**Previous Conversation**:
""",
    "output": """Expected: ``process_input`` filters ``retrieve_spending_transactions()`` to the current calendar month (start of month through today), sums amounts (e.g. ``utter_transaction_total``), handles empty df.""",
  },
  {
    "name": "active_loan_liabilities_all_accounts",
    "batch": "loan_liabilities_accounts",
    "input": """**Last User Request**: Check all user accounts for the presence of any active loan liabilities.
**Previous Conversation**:
""",
    "output": """Expected: ``retrieve_credit_accounts()`` filtered to loan account types (``loan_*``), list balances / names via ``account_names_and_balances`` or similar; clear message if none.""",
  },
  {
    "name": "macys_category_april_15_2026",
    "batch": "category_by_merchant_date_amount",
    "input": """**Last User Request**: What is the current category assigned to the Macy's transaction of $85 that occurred on April 14, 2026?
**Previous Conversation**:
""",
    "output": """Expected: filter ``retrieve_spending_transactions()`` by date and Macy's name (case-insensitive); for user dollar *N* use ``(amount >= N - 0.5) & (amount < N + 0.5)`` (e.g. 85 → 84.5..85.5), not ``==`` or ``[N, N+1)``. Return ``category`` (and ids per template).""",
  },
  {
    "name": "five_most_recent_transactions",
    "batch": "recent_transactions_top_n",
    "input": """**Last User Request**: List the 5 most recent transactions.

**Previous Conversation**:
""",
    "output": """Expected: load spending (and/or income) via ``retrieve_spending_transactions`` / ``retrieve_income_transactions``, combine if needed, sort by ``date`` descending, ``head(5)``, format with ``transaction_names_and_amounts`` (include ``account_id`` and ``transaction_id`` in template); handle empty df.""",
  },
  {
    "name": "lur_band_below_n85_jan_14_2026",
    "batch": "half_dollar_amount_band_boundaries",
    "input": """**Last User Request**: What is the current category assigned to the Microsoft 365 transaction of $85 that occurred on January 14, 2026?

**Previous Conversation**:
""",
    "output": """Expected: filter by name (e.g. Microsoft / 365) and date 2026-01-14; for user dollar *N*=85 use band ``[84.5, 85.5)``. Stored amount 84.49 is **below** the band — no matching row; clear empty / not-found (check ``df.empty``).""",
  },
  {
    "name": "lur_band_lower_inclusive_n85_jan_22_2026",
    "batch": "half_dollar_amount_band_boundaries",
    "input": """**Last User Request**: What is the current category assigned to the Whole Foods transaction of $85 that occurred on January 22, 2026?

**Previous Conversation**:
""",
    "output": """Expected: amount 84.5 on **lower inclusive** bound of ``[84.5, 85.5)`` for *N*=85; return ``meals_groceries`` (and ids per template).""",
  },
  {
    "name": "lur_band_interior_n85_feb_4_2026",
    "batch": "half_dollar_amount_band_boundaries",
    "input": """**Last User Request**: What category is assigned to the Trader Joe's transaction of $85 on February 4, 2026?

**Previous Conversation**:
""",
    "output": """Expected: amount 85.0 strictly inside ``[84.5, 85.5)``; return ``meals_groceries``.""",
  },
  {
    "name": "lur_band_upper_inside_n85_feb_18_2026",
    "batch": "half_dollar_amount_band_boundaries",
    "input": """**Last User Request**: What is the category for the Target purchase of $85 on February 18, 2026?

**Previous Conversation**:
""",
    "output": """Expected: amount 85.49 still **inside** ``[84.5, 85.5)``; return ``shopping_clothing``.""",
  },
  {
    "name": "lur_band_upper_excluded_n85_mar_2_2026",
    "batch": "half_dollar_amount_band_boundaries",
    "input": """**Last User Request**: What category is assigned to the Shell gas transaction of $85 on March 2, 2026?

**Previous Conversation**:
""",
    "output": """Expected: stored amount 85.5 is **outside** ``[84.5, 85.5)`` for user $85; no band match; not-found or explain no match.""",
  },
  {
    "name": "lur_band_n86_interior_mar_9_2026",
    "batch": "half_dollar_amount_band_boundaries",
    "input": """**Last User Request**: What category is assigned to the Uber Eats transaction of $86 on March 9, 2026?

**Previous Conversation**:
""",
    "output": """Expected: for *N*=86 use ``[85.5, 86.5)``; amount 86.0 matches; return ``meals_delivered_food``.""",
  },
  {
    "name": "lur_band_n86_upper_excluded_apr_2_2026",
    "batch": "half_dollar_amount_band_boundaries",
    "input": """**Last User Request**: What is the category for the Spotify USA charge of $86 on April 2, 2026?

**Previous Conversation**:
""",
    "output": """Expected: stored 86.5 is **outside** ``[85.5, 86.5)`` for user $86; no band match; empty / not-found handling.""",
  },
  {
    "name": "lur_band_n84_lower_inclusive_mar_21_2026",
    "batch": "half_dollar_amount_band_boundaries",
    "input": """**Last User Request**: What category is assigned to the Starbucks transaction of $84 on March 21, 2026?

**Previous Conversation**:
""",
    "output": """Expected: for *N*=84 use ``[83.5, 84.5)``; amount 83.5 is **lower inclusive**; return ``meals_dining_out``.""",
  },
  {
    "name": "lur_band_n84_below_apr_10_2026",
    "batch": "half_dollar_amount_band_boundaries",
    "input": """**Last User Request**: What is the current category assigned to the Costco Wholesale transaction of $84 on April 10, 2026?

**Previous Conversation**:
""",
    "output": """Expected: amount 83.49 is **below** ``[83.5, 84.5)``; no match for user $84; empty / not-found.""",
  },
]


def get_test_case(test_name_or_index):
  if isinstance(test_name_or_index, int):
    if 0 <= test_name_or_index < len(TEST_CASES):
      return TEST_CASES[test_name_or_index]
    return None
  if isinstance(test_name_or_index, str):
    for tc in TEST_CASES:
      if tc["name"] == test_name_or_index:
        return tc
    return None
  return None


def valid_batch_ids() -> list[str]:
  return sorted({tc["batch"] for tc in TEST_CASES})


def indices_for_batch(batch_id: str) -> list[int]:
  key = (batch_id or "").strip()
  return [i for i, tc in enumerate(TEST_CASES) if tc["batch"] == key]


def print_available_test_cases():
  print("\nAvailable test cases (by batch):")
  prev_batch = None
  for i, tc in enumerate(TEST_CASES):
    b = tc["batch"]
    if b != prev_batch:
      print(f"  Batch {b}:")
      prev_batch = b
    print(f"    {i}: {tc['name']}")
  print("  all: run all test cases")
  print(f"  --batch <id>: run every case in that batch (ids: {', '.join(valid_batch_ids())})")


def run_test(
  test_name_or_index_or_dict,
  optimizer: "LookupUserDataOptimizer" = None,
  replacements: dict = None,
  include_thoughts: bool = True,
):
  if optimizer is None:
    optimizer = LookupUserDataOptimizer()
  if isinstance(test_name_or_index_or_dict, dict):
    if "input" not in test_name_or_index_or_dict:
      print("Invalid test dict: must contain 'input'.")
      return None
    name = test_name_or_index_or_dict.get("name", "custom_test")
    repl = test_name_or_index_or_dict.get("replacements", replacements)
    last_user_request, previous_conversation = parse_planner_test_input(
      test_name_or_index_or_dict["input"]
    )
    print(f"\n{'='*80}\nRunning test: {name}\n{'='*80}\n")
    _maybe_ensure_lur_amount_band_fixtures(name)
    out = test_name_or_index_or_dict.get("output")
    result = _run_test_with_logging(
      last_user_request,
      previous_conversation,
      optimizer,
      replacements=repl,
      include_thoughts=include_thoughts,
      expected_output_rubric=out,
    )
    if out:
      print("\n" + "=" * 80 + "\nEXPECTED OUTPUT:\n" + "=" * 80 + "\n" + out + "\n" + "=" * 80 + "\n")
    return result
  tc = get_test_case(test_name_or_index_or_dict)
  if tc is None:
    print(f"Test case '{test_name_or_index_or_dict}' not found.")
    print_available_test_cases()
    return None
  print(f"\n{'='*80}\nRunning test: {tc['name']} (batch {tc['batch']})\n{'='*80}\n")
  _maybe_ensure_lur_amount_band_fixtures(tc["name"])
  last_user_request, previous_conversation = parse_planner_test_input(tc["input"])
  result = _run_test_with_logging(
    last_user_request,
    previous_conversation,
    optimizer,
    replacements=replacements,
    include_thoughts=include_thoughts,
    expected_output_rubric=tc.get("output"),
  )
  if tc.get("output"):
    print("\n" + "=" * 80 + "\nEXPECTED OUTPUT:\n" + "=" * 80 + "\n" + tc["output"] + "\n" + "=" * 80 + "\n")
  return result


def run_tests(
  test_names_or_indices_or_dicts=None,
  optimizer: "LookupUserDataOptimizer" = None,
  replacements: dict = None,
  include_thoughts: bool = True,
):
  if test_names_or_indices_or_dicts is None:
    test_names_or_indices_or_dicts = list(range(len(TEST_CASES)))
  results = []
  for item in test_names_or_indices_or_dicts:
    results.append(
      run_test(item, optimizer, replacements=replacements, include_thoughts=include_thoughts)
    )
  return results


def main(
  test: str = None,
  batch: str = None,
  no_thinking: bool = False,
  model: str = None,
  thinking_budget_cli: int = None,
  max_output_tokens_cli: int = None,
):
  gc = RUN_SETTINGS["gen_config"]
  if no_thinking:
    thinking_budget = 0
  elif thinking_budget_cli is not None:
    thinking_budget = thinking_budget_cli
  else:
    thinking_budget = gc["thinking_budget"]
  include_thoughts = not no_thinking
  optimizer = LookupUserDataOptimizer(
    model_name=model,
    thinking_budget=thinking_budget,
    max_output_tokens=max_output_tokens_cli,
  )
  if test is not None:
    if test.strip().lower() == "all":
      print(f"\n{'='*80}\nRunning ALL test cases\n{'='*80}\n")
      for i in range(len(TEST_CASES)):
        run_test(i, optimizer, include_thoughts=include_thoughts)
        if i < len(TEST_CASES) - 1:
          print("\n" + "-" * 80 + "\n")
      return
    test_val = int(test) if test.isdigit() else test
    result = run_test(test_val, optimizer, include_thoughts=include_thoughts)
    if result is None:
      print_available_test_cases()
    return
  if batch is not None:
    idxs = indices_for_batch(batch)
    if not idxs:
      print(f"No tests found for batch {batch!r}. Known: {valid_batch_ids()}")
      print_available_test_cases()
      return
    print(f"\n{'='*80}\nRunning batch {batch} ({len(idxs)} cases)\n{'='*80}\n")
    for j, i in enumerate(idxs):
      run_test(i, optimizer, include_thoughts=include_thoughts)
      if j < len(idxs) - 1:
        print("\n" + "-" * 80 + "\n")
    return
  print("Usage:")
  print("  Run a single test: --test <name_or_index>")
  print("  Run all tests: --test all")
  print("  Run one batch: --batch <id>  (e.g. half_dollar_amount_band_boundaries)")
  print("  --no-thinking: mirror planner flag (thinking_budget=0, include_thoughts=False)")
  print("  --model / --thinking-budget / --max-output-tokens: override RUN_SETTINGS for this run")
  print_available_test_cases()


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(
    description="Run lookup user data optimizer tests (P:Func:LookupUserData)"
  )
  parser.add_argument(
    "--test",
    type=str,
    help='Test name or index (e.g. "total_spent_current_month" or "0"), or "all"',
  )
  parser.add_argument(
    "--batch",
    type=str,
    help='Run all cases with this batch id (e.g. "half_dollar_amount_band_boundaries")',
  )
  parser.add_argument("--no-thinking", action="store_true", help="Set include_thoughts=False")
  parser.add_argument(
    "--model",
    type=str,
    default=None,
    help="Override Gemini model id for this run (default: RUN_SETTINGS['model_name'])",
  )
  parser.add_argument(
    "--thinking-budget",
    type=int,
    default=None,
    help="Override thinking budget for this run (ignored with --no-thinking)",
  )
  parser.add_argument(
    "--max-output-tokens",
    type=int,
    default=None,
    help="Override max_output_tokens for this run",
  )
  args = parser.parse_args()
  main(
    test=args.test,
    batch=args.batch,
    no_thinking=args.no_thinking,
    model=args.model,
    thinking_budget_cli=args.thinking_budget,
    max_output_tokens_cli=args.max_output_tokens,
  )
