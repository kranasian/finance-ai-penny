from google import genai
from google.genai import types
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path to import sandbox
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sandbox
from database import Database

# Load environment variables
load_dotenv()

RUN_SETTINGS = {
  "user_id": 3,
  "gen_config": {
    "top_k": 40,
    "top_p": 0.95,
    "temperature": 0.6,
    "thinking_budget": 0,
    "max_output_tokens": 2048,
  },
  "model_name": "gemini-flash-lite-latest",
  "replacements": {
    "TODAY_DATE": "November 8, 2025",
    "ACCOUNT_NAMES": "* chase total checking **4372\n* chase sapphire\n* amex cash magnet\n* chase savings",
    "SUBSCRIPTION_NAMES": "*spotify\n* netflix\n* disney plus\n* dominion energy\n* at&t internet\n* at&t wireless\n* sparkfun\n* property group llc\n* netflix\n* travelers insurance\n* mid-carolina electric cooperative\n* credit card interest\n* municipal water and sewer\n* crunch fitness\n* after school program",
  },
}


SYSTEM_PROMPT = """Your name is "Penny" and you are a helpful AI specialized in code generation. **You only output python code.**

## Task & Rules
1. Output `process_input() -> tuple[bool, str]`. Keep code concise. No code comments.
2. Use only `IMPLEMENTED_FUNCTIONS` and `IMPLEMENTED_DATE_FUNCTIONS`. `import datetime` is assumed.
3. Use partial case-insensitive matching for account/subscription/merchant names when filtering by name.
4. Always guard DataFrame access with `if df.empty` before column access, filters, or `.str`.
5. Collect response lines in a list and return `chr(10).join(output_lines)`.
6. **Amount matching (default rounded):** for user whole-dollar `X` in counts, totals, or “about $X” filters, use `X - 0.50 <= amount <= X + 0.49` (inclusive). Keep sign; no `abs()` unless asked.
7. **Amount matching (exact list only):** when the user asks to **list/show transactions that are exactly $X** (or “exact amount”, “equal to $X”, “match $X exactly”), filter with `amount == X` (float). Do **not** use the rounded band for that list. “How many $50 …” / “$50 payments” still uses the rounded band unless they also demand exact equality in the list request.
8. **Trust lookup data over the question:** never invent rows, categories, or balances to match user assumptions. Filter and answer from retrieved data only; if the user’s stated merchant, amount, or category does not match lookup results, report what the data shows (labeled), not the user’s premise.
9. For "past/next n months/weeks", exclude the current month/week. Use `today = datetime.now()` and date helpers—never hardcode calendar dates.
10. For year-based ranges, use `get_after_periods(..., 'yearly', ...)` with `get_start_of_year`/`get_end_of_year`; set `end = get_end_of_year(start)` when anchored on `start`.
11. Discretionary spending (parent-tier union): `startswith('meals') | startswith('shopping') | startswith('leisure') | category == 'donations_gifts'`.
12. Prefer `utter_transaction_total`, `utter_forecast_amount`, `utter_subscription_totals` over manual amount strings when summing.
13. Match requested granularity: weekly → weekly lines; monthly → monthly lines; do not collapse to one total when a breakdown was requested.
14. Net worth: sum depository `balance_current` as assets and credit/loan `balance_current` as liabilities unless the user asks for available balances. Format money with `utter_absolute_amount`, not raw numbers.
15. Category filters: chain `str.startswith('meals') | ...` — do not pass a tuple to a single `startswith`.
16. "How many" plus merchant/amount filters: state the count on its own labeled line, then list every matching row under a header (never only the list).
17. Date windows:
   - **Past n months**: `start = get_start_of_month(get_after_periods(today, 'monthly', -n))`, `end = get_end_of_month(get_after_periods(today, 'monthly', -1))`.
   - **Past n weeks**: `start = get_start_of_week(get_after_periods(today, 'weekly', -n))`, `end = get_end_of_week(get_after_periods(today, 'weekly', -1))`.
   - **Next n months**: `start = get_start_of_month(get_after_periods(today, 'monthly', 1))`, `end = get_end_of_month(get_after_periods(today, 'monthly', n))`.
   - **Next n weeks**: `start = get_start_of_week(get_after_periods(today, 'weekly', 1))`, `end = get_end_of_week(get_after_periods(today, 'weekly', n))`.
18. **Exclude transfers** on spending/forecast frames unless requested: `df = df[~df['category'].str.startswith('transfers', na=False)]` when not `df.empty`.

## Natural language → category slug (map user words to DB `category`; pick the tier they mean)
Use `<CATEGORY>` slugs and `categories.py` hierarchy. **Do not** treat the user’s label as truth—map intent to the correct slug, then filter lookup data.

| User may say | Map to slug (tier) |
|--------------|-------------------|
| groceries, grocery store | `meals_groceries` (sub) |
| dining out, restaurants, eating out | `meals_dining_out` or parent `meals` if they mean all food out |
| delivery food, DoorDash, takeout apps | `meals_delivered_food` |
| children's stuff, kids stuff, stuff for my kids, kids toys/clothes (shopping sense) | `shopping_kids` (Shopping: Kids)—**not** `education_kids_activities` |
| kids activities, after-school, sports classes, camps (activity sense) | `education_kids_activities` |
| tuition, school fees | `education_tuition` |
| clothing, clothes, fashion | `shopping_clothing` |
| gadgets, electronics, tech purchases | `shopping_gadgets` |
| pets, pet supplies | `shopping_pets` |
| utilities, electric, water bill | `shelter_utilities` |
| rent, mortgage, housing | `shelter_home` |
| gas, fuel, car maintenance | `transportation_car` |
| bus, subway, transit | `transportation_public` |
| medical, pharmacy, doctor | `health_medical_pharmacy` |
| gym, fitness | `health_gym_wellness` |
| salary, paycheck | `income_salary` |
| discretionary (unspecified) | parent union in rule 11 |

If they name a **parent** (e.g. “shopping”, “meals”), use `startswith('shopping')` etc. If they name a **sub** (e.g. “groceries”), use `== 'meals_groceries'`. Label answers with the mapped category (e.g. “Shopping: Kids (`shopping_kids`)”). **Common mistake:** “children’s stuff” → `shopping_kids`, not `education_kids_activities` (that is sports/camps/after-school).

**Merchant or amount in the question:** filter from retrieved rows only. If no row matches the user’s stated merchant or exact amount, say so with the period label—do not fabricate matches or widen filters to pretend the premise was true.

## Category tiers (`category` slugs like `meals_groceries`, `bills_connectivity`)
Three tiers—**never mix in one ranking** and **never add totals across tiers** (inflates spending):
- **Sub (leaf):** in DB (e.g. `meals_groceries`). User names the sub → `category == 'meals_groceries'`. Never use `startswith('meals')` for groceries-only.
- **Parent:** slug prefix before `_`, or parent-only slug (`donations_gifts`, `miscellaneous`, `uncategorized`). Rankings and parent totals: one `category.str.startswith('meals')` filter—**do not** sum `meals_groceries + meals_dining_out + ...` separately.
- **Top-level:** Food, Bills, Shopping, Income, Others — **not** in DB. Never rank Food alongside `meals` or `meals_dining_out`. Never add top-level + parent + sub totals.

**Top-level → parent slug prefixes** (roll-up labels only; filter real slugs at parent or sub):
- **Food:** `meals`
- **Bills:** `bills`, `shelter`, `education`, `transportation`
- **Shopping:** `shopping`
- **Income:** `income`
- **Others:** `leisure`, `health`, `donations_gifts`, `miscellaneous`, `uncategorized`

**Parent key for groupby** (rankings / highest category spend—default tier): add column `parent_cat` — use full slug for `donations_gifts`, `miscellaneous`, `uncategorized`; else `category.str.split('_').str[0]`. Then `groupby('parent_cat')['amount'].sum()`. Label as parent tier (`meals`, `shopping`). Wrong: `groupby('category')` mixing subs with different parents; wrong: Food vs `meals` vs `meals_dining_out` in one ranked list.

**Whole spending total** (one number): sum **either** all parent-tier buckets **or** all top-level buckets—never parent + sub + top in one total. Prefer parent-prefix filters per real slug.

**Forecasts:** same tier rules; parent-level forecast ≠ sum of every sub forecast line; exclude `transfers*` like spending.

## Subscriptions vs transactions
`retrieve_subscriptions()` returns **transaction rows** per charge. For any subscription list/filter request, **always** aggregate first: `subs = subs.groupby('subscription_name', as_index=False).agg(amount=('amount', 'sum'))` then call `subscription_names_and_amounts` / `utter_subscription_totals`. Never pass raw multi-month charge rows (five Prime Video payments → one Prime Video line).

## User-facing output (second return value)
Write for someone who only reads the final text—no assumed context from the question or internal systems.

**A. Complete and precise** — Answer **only** what was requested—no extra sections, categories, or metrics. State every requested fact explicitly (counts, yes/no, totals, each list item). Wrong if the user must infer an answer, or if you add data they did not ask for.

**B. Clear labels** — Every number has a plain label: what it is, the time period or scope, and dollars vs count. Use section headers (e.g. `Depository accounts:`, `Grocery spending (past 2 full months, current month excluded):`). Yes/no questions: lead with `Yes.` or `No.`, then supporting labeled figures. Empty results still name the requested period or scope (not generic "not found" alone).

**C. Full lists** — For list/show/all/every requests, include **every** matching row. Never use `.head()`, `.tail()`, or sample limits unless the user gives a number (e.g. "top 5"). When they also ask for a total, add a labeled total after the full list.

**D. Comparisons** — For compare/vs/difference/more than/less than: (1) labeled figure for each side with period in the label, (2) labeled net or delta, (3) a `Comparison:` line in plain English (which is larger and by how much, or that they are equal). Side-by-side numbers alone are insufficient.

Today: |TODAY_DATE|.

<IMPLEMENTED_FUNCTIONS>

- `retrieve_depository_accounts() -> pd.DataFrame`: checking, savings. Cols: account_type, account_name, balance_available, balance_current
- `retrieve_credit_accounts() -> pd.DataFrame`: credit, loans. Cols: account_type, account_name, balance_available, balance_current, balance_limit
- `account_names_and_balances(df, template) -> str`: Format account list. Placeholders: any df col.  When listing accounts, include account_id in the template.
- `retrieve_income_transactions() -> pd.DataFrame`: Past income. Cols: date, transaction_name, amount, category, transaction_id, account_id
- `retrieve_spending_transactions() -> pd.DataFrame`: Past spending (includes transfers—exclude `transfers*` before totals/rankings unless requested). Cols: date, transaction_name, amount, category, transaction_id, account_id
- `transaction_names_and_amounts(df, template) -> str`: List transactions. Placeholders: any df col.  When listing transactions, include account_id and transaction_id in the template.
- `utter_transaction_total(df, template) -> str`: Sum transactions then use correct placeholder for income or spending. Placeholders: {income_total_amount}, {spending_total_amount}. Example: "Total income: {income_total_amount}" or "Total spending: {spending_total_amount}"
- `retrieve_spending_forecasts(granularity='monthly') -> pd.DataFrame`: Future spending by week/month (exclude `transfers*` before totals unless requested). Cols: start_date, forecasted_amount, category
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
- `transfers`: internal movements (`transfers_*`). Exclude from spending totals unless requested.
- `miscellaneous`: other.

Parent slug prefixes for roll-ups: `meals`, `leisure`, `bills`, `shelter`, `education`, `shopping`, `transportation`, `health`, `income`.

</CATEGORY>

<EXAMPLES>

input: User: List all my accounts.
output:
```python
def process_input():
    output_lines = []
    dep = retrieve_depository_accounts()
    cre = retrieve_credit_accounts()
    if dep.empty and cre.empty:
        return True, "No accounts found."
    if not dep.empty:
        output_lines.append("Depository accounts (all):")
        output_lines.append(account_names_and_balances(dep, "- {account_name} ({account_type}): {balance_current} (account_id: {account_id})"))
    if not cre.empty:
        output_lines.append("Credit and loan accounts (all):")
        output_lines.append(account_names_and_balances(cre, "- {account_name} ({account_type}): {balance_current} (account_id: {account_id})"))
    return True, chr(10).join(output_lines)
```

input: User: How much did I spend on my children's stuff last month?
output:
```python
def process_input():
    today = datetime.now()
    start = get_start_of_month(get_after_periods(today, 'monthly', -1))
    end = get_end_of_month(start)
    sp = retrieve_spending_transactions()
    if sp.empty:
        return True, "No spending in the past calendar month."
    sp = sp[(sp['date'] >= start) & (sp['date'] <= end)]
    if sp.empty:
        return True, "No spending in the past calendar month."
    sp = sp[~sp['category'].str.startswith('transfers', na=False)]
    sp = sp[sp['category'] == 'shopping_kids'] if not sp.empty else pd.DataFrame()
    if sp.empty:
        return True, "No Shopping: Kids spending in the past calendar month."
    return True, utter_transaction_total(sp, "Shopping: Kids spending (past calendar month): {spending_total_amount}")
```

input: User: How many $50 AT&T payments have I had this year?
output:
```python
def process_input():
    output_lines = []
    start = get_start_of_year(datetime.now())
    end = get_end_of_year(start)
    sp = retrieve_spending_transactions()
    if sp.empty:
        return True, "Count of $50 AT&T payments this calendar year: 0."
    sp = sp[(sp['date'] >= start) & (sp['date'] <= end)]
    if not sp.empty:
        sp = sp[~sp['category'].str.startswith('transfers', na=False)]
    sp = sp[sp['transaction_name'].str.lower().str.contains('at&t', na=False) & (sp['amount'] >= 49.50) & (sp['amount'] <= 50.49)] if not sp.empty else pd.DataFrame()
    output_lines.append(f"Count of $50 AT&T payments this calendar year: {len(sp)}.")
    if not sp.empty:
        output_lines.append("Matching payments (all):")
        output_lines.append(transaction_names_and_amounts(sp.sort_values('date'), "- {transaction_name}: {amount} on {date} (account_id: {account_id}, transaction_id: {transaction_id})"))
    return True, chr(10).join(output_lines)
```

input: User: List every transaction that is exactly $85.00 from last month.
output:
```python
def process_input():
    today = datetime.now()
    start = get_start_of_month(get_after_periods(today, 'monthly', -1))
    end = get_end_of_month(start)
    sp = retrieve_spending_transactions()
    if sp.empty:
        return True, "No transactions exactly $85.00 in the past calendar month."
    sp = sp[(sp['date'] >= start) & (sp['date'] <= end)]
    if not sp.empty:
        sp = sp[~sp['category'].str.startswith('transfers', na=False)]
    sp = sp[sp['amount'] == 85.0] if not sp.empty else pd.DataFrame()
    if sp.empty:
        return True, "No transactions exactly $85.00 in the past calendar month."
    return True, chr(10).join([
        "Transactions exactly $85.00 (past calendar month, all):",
        transaction_names_and_amounts(sp.sort_values('date'), "- {transaction_name}: {amount} on {date} in {category} (account_id: {account_id}, transaction_id: {transaction_id})"),
    ])
```

input: User: Compare my income and discretionary spending last year.
output:
```python
def process_input():
    start = get_start_of_year(get_after_periods(datetime.now(), 'yearly', -1))
    end = get_end_of_year(start)
    inc = retrieve_income_transactions()
    sp = retrieve_spending_transactions()
    inc = inc[(inc['date'] >= start) & (inc['date'] <= end)] if not inc.empty else pd.DataFrame()
    sp = sp[(sp['date'] >= start) & (sp['date'] <= end)] if not sp.empty else pd.DataFrame()
    if not sp.empty:
        sp = sp[~sp['category'].str.startswith('transfers', na=False)]
        sp = sp[sp['category'].str.startswith('meals') | sp['category'].str.startswith('shopping') | sp['category'].str.startswith('leisure') | (sp['category'] == 'donations_gifts')]
    if inc.empty and sp.empty:
        return True, "No income or discretionary spending data for last calendar year."
    inc_total = inc['amount'].sum() if not inc.empty else 0
    sp_total = sp['amount'].sum() if not sp.empty else 0
    net = inc_total - sp_total
    output_lines = [
        "Income vs discretionary spending (last calendar year):",
        f"Total income: {utter_absolute_amount(inc_total, '{amount}')}",
        f"Total discretionary spending: {utter_absolute_amount(sp_total, '{amount}')}",
        f"Net (income minus discretionary spending): {utter_absolute_amount(net, '{amount_with_direction}')}",
    ]
    if net > 0:
        output_lines.append(f"Comparison: Income exceeded discretionary spending by {utter_absolute_amount(net, '{amount}')}.")
    elif net < 0:
        output_lines.append(f"Comparison: Discretionary spending exceeded income by {utter_absolute_amount(-net, '{amount}')}.")
    else:
        output_lines.append("Comparison: Income and discretionary spending were equal last year.")
    return True, chr(10).join(output_lines)
```

input: User: List all subscriptions paid last month and total them.
output:
```python
def process_input():
    today = datetime.now()
    start = get_start_of_month(get_after_periods(today, 'monthly', -1))
    end = get_end_of_month(start)
    subs = retrieve_subscriptions()
    if subs.empty:
        return True, "No subscriptions found."
    subs = subs[(subs['date'] >= start) & (subs['date'] <= end)]
    if subs.empty:
        return True, "No subscription payments last month."
    subs = subs.groupby('subscription_name', as_index=False).agg(amount=('amount', 'sum'))
    return True, chr(10).join([
        "Subscriptions paid last month (one row per establishment):",
        subscription_names_and_amounts(subs, "- {subscription_name}: {amount}"),
        utter_subscription_totals(subs, "Total subscription payments last month: {total_amount}"),
    ])
```

</EXAMPLES>

<ACCOUNT_NAMES>

|ACCOUNT_NAMES|

</ACCOUNT_NAMES>

<SUBSCRIPTION_NAMES>

|SUBSCRIPTION_NAMES|

</SUBSCRIPTION_NAMES>
"""


# - `retrieve_subscriptions() -> pd.DataFrame`: Cols: transaction_name, amount, category, subscription_name, date
# - `subscription_names_and_amounts(df, template) -> str`: List subscriptions.
# - `utter_subscription_totals(df, template) -> str`: Sum subscriptions.
# - `respond_to_app_inquiry(inquiry) -> str`: For general Qs.

class LookupUserDataOptimizer:
  """Handles all Gemini API interactions for financial planning and optimization"""
  
  def __init__(
    self,
    model_name: str = None,
    thinking_budget: int = None,
    max_output_tokens: int = None,
  ):
    """Initialize the Gemini agent with API configuration for financial planning"""
    self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    gc = RUN_SETTINGS["gen_config"]
    self.model_name = model_name if model_name is not None else RUN_SETTINGS["model_name"]
    self.thinking_budget = (
      thinking_budget if thinking_budget is not None else gc["thinking_budget"]
    )
    self.top_k = gc["top_k"]
    self.temperature = gc["temperature"]
    self.top_p = gc["top_p"]
    self.max_output_tokens = (
      max_output_tokens if max_output_tokens is not None else gc["max_output_tokens"]
    )
    
    # Safety Settings
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
    
    # System Prompt
    self.system_prompt = SYSTEM_PROMPT

  def _build_account_names_section(self, user_id: int) -> str:
    """
    Build the ACCOUNT_NAMES section dynamically based on the user's accounts.
    
    Args:
      user_id: The user ID to retrieve accounts for
      
    Returns:
      String containing the ACCOUNT_NAMES section
    """
    db = Database()
    accounts = db.get_accounts_by_user(user_id)
    
    if not accounts:
      return "    (No accounts found for this user)"
    
    account_names_list = []
    for account in accounts:
      account_name = account.get('account_name', '')
      if account_name:
        account_names_list.append(f"    - `{account_name}`")
    
    account_names_text = "\n".join(account_names_list) if account_names_list else "    (No account names found)"
    
    return account_names_text

  def _build_subscription_names_section(self, user_id: int) -> str:
    """
    Build the SUBSCRIPTION_NAMES section dynamically based on the user's subscriptions.
    
    Args:
      user_id: The user ID to retrieve subscriptions for
      
    Returns:
      String containing the SUBSCRIPTION_NAMES section
    """
    db = Database()
    # Get subscription transactions to extract unique subscription names
    subscription_transactions = db.get_subscription_transactions(user_id, confidence_score_bills_threshold=0.5)
    
    if not subscription_transactions:
      return "    (No subscriptions found for this user)"
    
    # Extract unique subscription names from transactions
    subscription_names_set = set()
    for transaction in subscription_transactions:
      subscription_name = transaction.get('subscription_name', '')
      if subscription_name:
        subscription_names_set.add(subscription_name)
    
    subscription_names_list = [f"    - `{name}`" for name in sorted(subscription_names_set)]
    
    subscription_names_text = "\n".join(subscription_names_list) if subscription_names_list else "    (No subscription names found)"
    
    return subscription_names_text

  def _get_today_date_string(self) -> str:
    """
    Get today's date formatted as "Day, Month DD, YYYY"
    
    Returns:
      String containing today's date in the specified format
    """
    today = datetime.now()
    return today.strftime("%A, %B %d, %Y")

  
  def generate_response(
    self,
    last_user_request: str,
    user_id: int = None,
    replacements: dict = None,
  ) -> dict:
    """
    Generate a response using Gemini API for financial planning.
    
    Args:
      last_user_request: The last user request as a string
      user_id: User ID for building dynamic sections when replacements omit names
      replacements: Optional overrides for RUN_SETTINGS prompt placeholders
      
    Returns:
      Dict with generated code and thought_summary
    """
    if user_id is None:
      user_id = RUN_SETTINGS.get("user_id", 1)
    repl = dict(RUN_SETTINGS.get("replacements") or {})
    if replacements:
      repl.update(replacements)
    today_date = repl.get("TODAY_DATE") or self._get_today_date_string()
    account_names_section = repl.get("ACCOUNT_NAMES")
    if account_names_section is None:
      account_names_section = self._build_account_names_section(user_id)
    subscription_names_section = repl.get("SUBSCRIPTION_NAMES")
    if subscription_names_section is None:
      subscription_names_section = self._build_subscription_names_section(user_id)
    full_system_prompt = self.system_prompt.replace("|TODAY_DATE|", today_date)
    full_system_prompt = full_system_prompt.replace("|ACCOUNT_NAMES|", account_names_section)
    full_system_prompt = full_system_prompt.replace("|SUBSCRIPTION_NAMES|", subscription_names_section)
    
    # Create request text with Last User Request and Previous Conversation
    request_text = types.Part.from_text(text=f"""User: {last_user_request}
output:""")
    
    # Create content and configuration
    contents = [types.Content(role="user", parts=[request_text])]
    
    generate_content_config = types.GenerateContentConfig(
      top_k=self.top_k,
      temperature=self.temperature,
      top_p=self.top_p,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=full_system_prompt)],
      thinking_config=types.ThinkingConfig(
        thinking_budget=self.thinking_budget,
        include_thoughts=True,
      ),
    )

    # Generate response
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

    clean_thought_summary = thought_summary.strip()
    if clean_thought_summary:
      print("=" * 80)
      print("THOUGHT SUMMARY:")
      print(clean_thought_summary)
      print("=" * 80)
      print()

    return {
      "response": output_text,
      "thought_summary": clean_thought_summary,
    }
  
  
  def get_available_models(self):
    """
    Get list of available Gemini models.
    
    Returns:
      List of available model names
    """
    try:
      models = genai.list_models()
      return [m.name for m in models if 'generateContent' in m.supported_generation_methods]
    except Exception as e:
      raise Exception(f"Failed to get models: {str(e)}")


def _get_heavy_data_user_id() -> int:
  """
  Get the user ID for HeavyDataUser from the database.
  
  Returns:
    The user ID for HeavyDataUser, or 1 if not found
  """
  try:
    db = Database()
    heavy_user = db.get_user("HeavyDataUser")
    if heavy_user and 'id' in heavy_user:
      return heavy_user['id']
    else:
      print("Warning: HeavyDataUser not found, using default user_id=1")
      return 1
  except Exception as e:
    print(f"Warning: Error getting HeavyDataUser: {e}, using default user_id=1")
    return 1


TEST_CASES = [
  {
    "name": "list_all_accounts",
    "user": "List all my accounts.",
    "ideal_output": """def process_input():
    output_lines = []
    dep = retrieve_depository_accounts()
    cre = retrieve_credit_accounts()

    if dep.empty and cre.empty:
        return True, "No accounts found."

    if not dep.empty:
        output_lines.append("Depository Accounts:")
        output_lines.append(account_names_and_balances(dep, "- {account_name} (account_id: {account_id}): {balance_current}"))

    if not cre.empty:
        output_lines.append("Credit/Loan Accounts:")
        output_lines.append(account_names_and_balances(cre, "- {account_name} (account_id: {account_id}): {balance_current}"))

    return True, chr(10).join(output_lines)""",
  },
  {
    "name": "net_worth_right_now",
    "user": "What is my net worth right now?",
    "ideal_output": """def process_input():
    output_lines = []
    dep = retrieve_depository_accounts()
    cre = retrieve_credit_accounts()

    if dep.empty and cre.empty:
        return True, "No accounts found."

    assets_total = dep['balance_current'].sum() if not dep.empty else 0
    liabilities_total = cre['balance_current'].sum() if not cre.empty else 0
    net_worth = assets_total - liabilities_total

    output_lines.append(f"Assets: {utter_absolute_amount(assets_total, '{amount_with_direction}')}")
    output_lines.append(f"Liabilities: {utter_absolute_amount(liabilities_total, '{amount_with_direction}')}")
    output_lines.append(f"Net Worth: {utter_absolute_amount(net_worth, '{amount_with_direction}')}")
    return True, chr(10).join(output_lines)""",
  },
  {
    "name": "groceries_past_two_months",
    "user": "How much did I spend on groceries in the past 2 months?",
    "ideal_output": """def process_input():
    start = get_start_of_month(get_after_periods(datetime.now(), 'monthly', -2))
    end = get_end_of_month(get_after_periods(datetime.now(), 'monthly', -1))
    sp = retrieve_spending_transactions()
    if sp.empty:
        return True, "No spending transactions found."

    sp = sp[(sp['date'] >= start) & (sp['date'] <= end)]
    sp = sp[sp['category'] == 'meals_groceries'] if not sp.empty else pd.DataFrame()
    if sp.empty:
        return True, "No grocery spending found in the past 2 months."
    return True, utter_transaction_total(sp, "Total grocery spending: {spending_total_amount}")""",
  },
  {
    "name": "top_five_spending_last_month",
    "user": "Show my top 5 largest spending transactions from last month.",
    "ideal_output": """def process_input():
    start = get_start_of_month(get_after_periods(datetime.now(), 'monthly', -1))
    end = get_end_of_month(start)
    sp = retrieve_spending_transactions()
    if sp.empty:
        return True, "No spending transactions found."

    sp = sp[(sp['date'] >= start) & (sp['date'] <= end)]
    if sp.empty:
        return True, "No spending transactions found last month."

    top_sp = sp.sort_values('amount', ascending=False).head(5)
    output_lines = ["Top 5 largest spending transactions from last month:"]
    output_lines.append(transaction_names_and_amounts(top_sp, "- {transaction_name}: {amount} on {date} in {category} (account_id: {account_id}, transaction_id: {transaction_id})"))
    return True, chr(10).join(output_lines)""",
  },
  {
    "name": "att_fifty_dollar_payments_this_year",
    "user": "How many $50 AT&T payments have I had this year?",
    "ideal_output": """def process_input():
    start = get_start_of_year(datetime.now())
    end = get_end_of_year(start)
    sp = retrieve_spending_transactions()
    if sp.empty:
        return True, "No spending transactions found."

    sp = sp[(sp['date'] >= start) & (sp['date'] <= end)]
    sp = sp[
        sp['transaction_name'].str.lower().str.contains('at&t', na=False) &
        (sp['amount'] >= 49.50) &
        (sp['amount'] <= 50.49)
    ] if not sp.empty else pd.DataFrame()
    return True, f"There have been {len(sp)} $50 AT&T payments so far this year." if not sp.empty else "There have been 0 $50 AT&T payments so far this year." """,
  },
  {
    "name": "subscriptions_paid_last_month",
    "user": "List all subscriptions paid last month and total them.",
    "ideal_output": """def process_input():
    subs = retrieve_subscriptions()
    if subs.empty:
        return True, "No subscriptions found."

    start = get_start_of_month(get_after_periods(datetime.now(), 'monthly', -1))
    end = get_end_of_month(start)
    subs = subs[(subs['date'] >= start) & (subs['date'] <= end)]
    if subs.empty:
        return True, "No subscriptions paid last month."

    output_lines = ["Subscriptions paid last month:"]
    output_lines.append(subscription_names_and_amounts(subs, "- {subscription_name}: {amount} on {date}"))
    output_lines.append(utter_subscription_totals(subs, "Total: {total_amount}"))
    return True, chr(10).join(output_lines)""",
  },
  {
    "name": "projected_expenses_next_three_months",
    "user": "What are my projected expenses for the next 3 months?",
    "ideal_output": """def process_input():
    sp = retrieve_spending_forecasts(granularity='monthly')
    if sp.empty:
        return True, "No spending forecasts found."

    today = datetime.now()
    start = get_start_of_month(get_after_periods(today, 'monthly', 1))
    end = get_end_of_month(get_after_periods(today, 'monthly', 3))
    sp = sp[(sp['start_date'] >= start) & (sp['start_date'] <= end)]
    if sp.empty:
        return True, "No spending forecasts for the next 3 months."

    output_lines = ["Projected expenses for the next 3 months:"]
    output_lines.append(forecast_dates_and_amount(sp.sort_values('start_date'), "- {start_date}: {forecasted_amount}"))
    output_lines.append(utter_forecast_amount(sp['forecasted_amount'].sum(), "Total projected spending: {spending_total_amount}"))
    return True, chr(10).join(output_lines)""",
  },
  {
    "name": "savings_next_four_weeks",
    "user": "Project my savings for the next 4 weeks.",
    "ideal_output": """def process_input():
    output_lines = []
    today = datetime.now()
    inc = retrieve_income_forecasts('weekly')
    sp = retrieve_spending_forecasts('weekly')
    if inc.empty and sp.empty:
        return True, "No forecasts found."

    start_next = get_start_of_week(get_after_periods(today, 'weekly', 1))
    dates = [get_after_periods(start_next, 'weekly', i) for i in range(4)]
    total_savings = 0
    for d in dates:
        i_val = inc[inc['start_date'] == d]['forecasted_amount'].sum() if not inc.empty else 0
        s_val = sp[sp['start_date'] == d]['forecasted_amount'].sum() if not sp.empty else 0
        savings = i_val - s_val
        total_savings += savings
        output_lines.append(f"{get_date_string(d)}: {utter_absolute_amount(savings, '{amount_with_direction}')}")
    output_lines.append(f"Total projected savings: {utter_absolute_amount(total_savings, '{amount_with_direction}')}")
    return True, chr(10).join(output_lines)""",
  },
  {
    "name": "income_vs_discretionary_last_year",
    "user": "Compare my income and discretionary spending last year.",
    "ideal_output": """def process_input():
    start = get_start_of_year(get_after_periods(datetime.now(), 'yearly', -1))
    end = get_end_of_year(start)
    inc = retrieve_income_transactions()
    sp = retrieve_spending_transactions()

    inc = inc[(inc['date'] >= start) & (inc['date'] <= end)] if not inc.empty else pd.DataFrame()
    sp = sp[(sp['date'] >= start) & (sp['date'] <= end)] if not sp.empty else pd.DataFrame()
    if not sp.empty:
        sp = sp[sp['category'].str.startswith('meals') | sp['category'].str.startswith('shopping') | sp['category'].str.startswith('leisure') | (sp['category'] == 'donations_gifts')]
    if inc.empty and sp.empty:
        return True, "No data found last year."

    inc_total = inc['amount'].sum() if not inc.empty else 0
    sp_total = sp['amount'].sum() if not sp.empty else 0
    net = inc_total - sp_total
    return True, chr(10).join([
        f"Income: {utter_absolute_amount(inc_total, '{amount}')}",
        f"Discretionary Spending: {utter_absolute_amount(sp_total, '{amount}')}",
        f"Net: {utter_absolute_amount(net, '{amount_with_direction}')}",
    ])""",
  },
  {
    "name": "checking_cover_utility_bill",
    "user": "Do I have enough in checking to cover a $180 utility bill this week?",
    "ideal_output": """def process_input():
    dep = retrieve_depository_accounts()
    if dep.empty:
        return True, "No depository accounts found."

    chk = dep[dep['account_type'] == 'deposit_checking']
    if chk.empty:
        return True, "No checking account found."

    available = chk['balance_available'].sum()
    remaining = available - 180
    if remaining >= 0:
        return True, f"Yes. You can cover it and still have {utter_absolute_amount(remaining, '{amount_with_direction}')}"
    return True, f"No. You are short by {utter_absolute_amount(remaining, '{amount_with_direction}')}" """,
  },
  {
    "name": "transportation_past_four_weeks",
    "user": "Show my transportation spending in the past 4 weeks.",
    "ideal_output": """def process_input():
    start = get_start_of_week(get_after_periods(datetime.now(), 'weekly', -4))
    end = get_end_of_week(get_after_periods(datetime.now(), 'weekly', -1))
    sp = retrieve_spending_transactions()
    if sp.empty:
        return True, "No spending transactions found."

    sp = sp[(sp['date'] >= start) & (sp['date'] <= end)]
    sp = sp[sp['category'].str.startswith('transportation')] if not sp.empty else pd.DataFrame()
    if sp.empty:
        return True, "No transportation spending in the past 4 weeks."

    output_lines = ["Transportation spending in the past 4 weeks:"]
    output_lines.append(transaction_names_and_amounts(sp.sort_values('date', ascending=False), "- {transaction_name}: {amount} on {date} (account_id: {account_id}, transaction_id: {transaction_id})"))
    output_lines.append(utter_transaction_total(sp, "Total: {spending_total_amount}"))
    return True, chr(10).join(output_lines)""",
  },
  {
    "name": "streaming_subscriptions",
    "user": "Which subscriptions look like streaming services?",
    "ideal_output": """def process_input():
    subs = retrieve_subscriptions()
    if subs.empty:
        return True, "No subscriptions found."

    filtered = subs[subs['subscription_name'].str.lower().str.contains('netflix|disney|hulu|spotify|youtube|max|prime', na=False)]
    if filtered.empty:
        return True, "No streaming subscriptions found."

    output_lines = ["Streaming subscriptions:"]
    output_lines.append(subscription_names_and_amounts(filtered.sort_values('subscription_name'), "- {subscription_name}: {amount}"))
    return True, chr(10).join(output_lines)""",
  },
]


BATCHES: dict[int, dict[str, object]] = {
  1: {
    "name": "Accounts, net worth, groceries",
    "tests": [0, 1, 2],
  },
  2: {
    "name": "Top spend, AT&T fuzzy band, subscriptions",
    "tests": [3, 4, 5],
  },
  3: {
    "name": "Forecasts and year-over-year compare",
    "tests": [6, 7, 8],
  },
  4: {
    "name": "Checking cover, transport, streaming",
    "tests": [9, 10, 11],
  },
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


def _run_test_with_logging(tc: dict, lookup_data: LookupUserDataOptimizer = None, user_id: int = None):
  if lookup_data is None:
    lookup_data = LookupUserDataOptimizer()

  if user_id is None:
    user_id = RUN_SETTINGS.get("user_id") or _get_heavy_data_user_id()

  last_user_request = tc["user"]
  llm_input = f"User: {last_user_request}"

  print("\n" + "=" * 80)
  print(f"Running lookup test: {tc['name']}")
  print("=" * 80)
  print("\nLLM INPUT:")
  print(llm_input)
  print()

  result = lookup_data.generate_response(last_user_request, user_id)
  response_text = result["response"]

  print("=" * 80)
  print("LLM OUTPUT:")
  print("=" * 80)
  print(response_text)
  print("=" * 80)
  print()

  print("=" * 80)
  print("SANDBOX EXECUTION:")
  print("=" * 80)
  try:
    success, output_string, logs, goals_list = sandbox.execute_agent_with_tools(
      response_text, user_id
    )
    print(f"Success: {success}")
    print()
    print("Output:")
    print("-" * 80)
    print(output_string)
    print("-" * 80)
    print()
  except Exception as e:
    print(f"**Sandbox Execution Error**: {str(e)}")
    import traceback
    print(traceback.format_exc())
  print("=" * 80)
  print()

  ideal_output = tc.get("ideal_output")
  if ideal_output:
    print("IDEAL OUTPUT:")
    print("=" * 80)
    print(ideal_output)
    print("=" * 80)
    print()

  return result


def run_test(test_name_or_index_or_dict, lookup_data: LookupUserDataOptimizer = None, user_id: int = None):
  if isinstance(test_name_or_index_or_dict, dict):
    tc = test_name_or_index_or_dict
  else:
    tc = get_test_case(test_name_or_index_or_dict)

  if tc is None:
    print(f"Test case '{test_name_or_index_or_dict}' not found.")
    return None

  return _run_test_with_logging(tc, lookup_data, user_id)


def run_tests(test_names_or_indices=None, lookup_data: LookupUserDataOptimizer = None, user_id: int = None):
  if test_names_or_indices is None:
    test_names_or_indices = list(range(len(TEST_CASES)))
  results = []
  for item in test_names_or_indices:
    results.append(run_test(item, lookup_data, user_id))
  return results


def main(batch: int = None, test: str = None):
  lookup_data = LookupUserDataOptimizer()

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
    run_tests(test_names_or_indices=info["tests"], lookup_data=lookup_data)
    return

  if test is not None:
    test_key = int(test) if test.isdigit() else test
    run_test(test_key, lookup_data=lookup_data)
    return

  print("Available lookup test cases:")
  for i, tc in enumerate(TEST_CASES):
    print(f"  {i}: {tc['name']}")
  print("\nBatches:")
  for b, info in BATCHES.items():
    names = [TEST_CASES[i]["name"] for i in info["tests"]]
    print(f"  {b}: {info['name']} — {', '.join(names)}")


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Run lookup user data optimizer tests")
  parser.add_argument(
    "--test",
    type=str,
    help="Test name or index (e.g. list_all_accounts or 0)",
  )
  parser.add_argument(
    "--batch",
    type=int,
    nargs="?",
    const=1,
    default=None,
    metavar="N",
    help="Run batch N of tests (see BATCHES in script)",
  )
  args = parser.parse_args()
  main(batch=args.batch, test=args.test)
