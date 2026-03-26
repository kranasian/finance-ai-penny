from google import genai
from google.genai import types
from google.genai.errors import ClientError
import os
import sys
from dotenv import load_dotenv
import json
from pydantic import BaseModel, Field
from typing import Literal

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

load_dotenv()

class SCHEMA(BaseModel):
    result_summary: str = Field(
        description=(
            "≤2 sentences: minimal facts from Execution Result that answer the Task Description only. "
            "Match task scope (e.g. rent/mortgage → housing-scale rows; omit unrelated merchants/categories from the same list). "
            "No transaction_id here; no meta/process; no internal rule names."
        )
    )
    next_status: Literal["COMPLETED", "PARTIALLY_COMPLETED", "FAILED", "IN_PROGRESS"] = Field(
        description=(
            "Exactly one of: COMPLETED, PARTIALLY_COMPLETED, FAILED, IN_PROGRESS — chosen only by applying the system prompt’s status list in order."
        )
    )
    outcome_reflection: str = Field(
        description=(
            "≤3 sentences; do not copy result_summary. Delta vs Previous Outcomes, why this status, next step. "
            "IN_PROGRESS: include transaction_id from Execution Result when the next step targets those rows (never invent). FAILED: short; note if payload has no ids. "
            "Housing same-day split (two shelter_home lines, same payee+date, different $): follow the system prompt **Edge** rule when applicable; never claim that pattern is missing. No internal rule names."
        )
    )


REFLECTION_WO_CODE_SYSTEM_PROMPT = """You are Strategizer AI’s reflection step.

Inputs: Task Description; Previous Outcomes; Execution Result (may look like (True, '...') — use the inner message as truth, not the outer flag alone).

Rules for reading the payload:
- Line items and tool-style text are ground truth; don’t invent merchants, amounts, categories, or ids.
- If the inner text has real errors (“unexpected error”, exceptions, contradictions), status is FAILED even if the outer tuple says True.
- “Successfully processed categorization request …” (or equivalent) counts as a successful fix **for what it names**, if it matches the task + Previous Outcomes, unless the same inner text also contains an error.

Pick next_status in this strict order (first that applies):

1) **COMPLETED** — The task objective is fully satisfied by **this** result: required labels are already correct in listed data, **or** a success message confirms the remediation the workflow was driving toward. For listing-style tasks, judge only rows in this payload (don’t assume missing rows elsewhere). For rent/mortgage reviews: different payees/amounts are normal; two shelter_home charges to one payee on one day with different dollar amounts still allow COMPLETED if both are shelter_home; a small same-day fee doesn’t block COMPLETED if main rent/mortgage lines are correct; tuition/dining/etc. in the same list don’t block COMPLETED if the housing lines that matter are correct. **Not** COMPLETED if the listing still shows miscategorized/uncategorized rows the task requires fixing.

2) **FAILED** — Hard failure: no verified successful fix in this payload, or only generic failure text.

3) **PARTIALLY_COMPLETED** — Rare: blocked on **external** human/auth/support only. Routine “more fixes needed” or partial auto-fixes → **IN_PROGRESS**, not PARTIALLY_COMPLETED.

4) **IN_PROGRESS** — Otherwise: last step gave usable evidence, work remains (e.g. salary-like deposits still uncategorized/wrong when the task says fix them; housing lines still wrong). Put transaction_id values in outcome_reflection when they appear in Execution Result and the next step should target those rows.

**Edge (housing lists):** If two lines share the same payee and the same calendar date in the text but show different dollar amounts, and both are `shelter_home`, that is still **COMPLETED** when the task is satisfied—and `outcome_reflection` must mention both amounts once (e.g. $2000 and $1050) and that the same-day split is acceptable.

Do not mention rubric numbers or internal rule nicknames in the JSON fields (follow the schema field descriptions for tone and brevity).

"""


class SelfReflectionWoCodeOptimizer:
  """Reflection component without code_executed in input: Task + Previous Outcomes + Execution Result only."""

  def __init__(self, model_name="gemini-flash-lite-latest", thinking_budget=700, max_output_tokens=1024, temperature=0.5):
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY missing.")
    self.client = genai.Client(api_key=api_key)
    self.model_name = model_name
    self.thinking_budget = thinking_budget
    self.temperature = temperature
    self.top_p = 0.95
    self.max_output_tokens = max_output_tokens
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ]
    self.system_prompt = REFLECTION_WO_CODE_SYSTEM_PROMPT

  def generate_response(
    self,
    task_description: str,
    previous_outcomes: dict[int | str, str] | list[str] | str | None,
    execution_result: str,
  ) -> str:
    previous_outcomes_block = _format_previous_outcomes(previous_outcomes)
    desc = f"""# Task Description

{task_description}"""
    if previous_outcomes_block.strip():
      desc += f"""

## Previous Outcomes

{previous_outcomes_block}"""
    request_text = types.Part.from_text(text=f"""{desc}

## Execution Result

{execution_result}
""")
    contents = [types.Content(role="user", parts=[request_text])]
    generate_content_config = types.GenerateContentConfig(
      temperature=self.temperature,
      top_p=self.top_p,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=self.system_prompt)],
      thinking_config=types.ThinkingConfig(
        thinking_budget=self.thinking_budget,
        include_thoughts=True,
      ),
      response_mime_type="application/json",
      response_schema=SCHEMA,
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
            if hasattr(candidate, "content") and candidate.content and getattr(candidate.content, "parts", None):
              for part in candidate.content.parts:
                if getattr(part, "thought", False) and getattr(part, "text", None):
                  thought_summary = (thought_summary + part.text) if thought_summary else part.text
    except ClientError as e:
      if self.thinking_budget == 0 and "only works in thinking mode" in (str(e) or ""):
        print("\n[NOTE] This model requires thinking mode.", flush=True)
        sys.exit(1)
      raise
    if thought_summary:
      print("\n## Thought Summary\n")
      print(thought_summary.strip() + "\n")
    return output_text


def _format_previous_outcomes(previous_outcomes: dict[int | str, str] | list[str] | str | None) -> str:
  if previous_outcomes is None:
    return ""
  if isinstance(previous_outcomes, str):
    return f"1. **Outcome #1**: {previous_outcomes}"
  if isinstance(previous_outcomes, dict):
    numbered_rows = []
    for key in sorted(previous_outcomes, key=lambda x: int(x) if str(x).isdigit() else str(x)):
      value = previous_outcomes[key]
      if isinstance(value, str) and value.strip():
        label_num = int(key) if str(key).isdigit() else key
        numbered_rows.append(f"{label_num}. **Outcome #{label_num}**: {value.strip()}")
    if numbered_rows:
      return "\n".join(numbered_rows)
    return ""
  clean_outcomes = [item.strip() for item in previous_outcomes if isinstance(item, str) and item.strip()]
  if not clean_outcomes:
    return ""
  return "\n".join(
    f"{idx}. **Outcome #{idx}**: {outcome}" for idx, outcome in enumerate(clean_outcomes, start=1)
  )


def _run_test_with_logging(
  task_description: str,
  previous_outcomes: dict[int | str, str] | list[str] | str | None,
  execution_result: str,
  optimizer: SelfReflectionWoCodeOptimizer | None = None,
  ideal_response: str | None = None,
):
  if optimizer is None:
    optimizer = SelfReflectionWoCodeOptimizer()
  previous_outcomes_block = _format_previous_outcomes(previous_outcomes)
  llm_input_display = f"""# Task Description

{task_description}"""
  if previous_outcomes_block.strip():
    llm_input_display += f"""

## Previous Outcomes

{previous_outcomes_block}"""
  llm_input_display += f"""

## Execution Result

{execution_result}"""
  print(f"## LLM Input\n")
  print(llm_input_display)
  print()
  result = optimizer.generate_response(task_description, previous_outcomes, execution_result)
  print(f"## LLM Output:\n")
  print(result)
  print()
  if ideal_response:
    print(f"## Ideal Response:\n\n{ideal_response}\n")
  return result


# Same salary test sequence as strategizer batch 1; each execution_result represents
# the prior strategizer run output that self-reflection receives as input.
TEST_CASES = [
  {
    "name": "salary_check_iteration_1a_reflection",
    "batch": 1,
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": None,
    "execution_result": "(True, '--- Recent Income (Last 30 Days) ---\\nRecent Income Transactions:\\n- $1440 was received from CA State Payroll on 2025-11-18 (Chase Total Checking **1563) categorized as income_salary; transaction_id=50101.\\n- $1340 was received from CA State Payroll on 2025-10-31 (Chase Total Checking **1563) categorized as income_salary; transaction_id=50102.\\nTotal recent income: earned $2780.\\n')",
    "ideal_response": "Reflection: COMPLETED; I see two recent transactions from CA State Payroll categorized as income_salary, so salary is already detected and appears correctly categorized. outcome_reflection need not foreground transaction_id values because no follow-up targeting specific rows is required.",
  },
  {
    "name": "salary_check_iteration_1b_reflection",
    "batch": 1,
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": None,
    "execution_result": "(True, '--- Recent Income (Last 30 Days) ---\\nRecent Income Transactions:\\n- $2500 was received from ADP PAYROLL on 2025-11-20 (Chase Total Checking **1563) categorized as uncategorized; transaction_id=50111.\\n- $2500 was received from Gusto on 2025-10-22 (Chase Total Checking **1563) categorized as uncategorized; transaction_id=50112.\\nTotal recent income: earned $5000.\\n')",
    "ideal_response": "Reflection: IN_PROGRESS; I see two recent large transactions from ADP PAYROLL and Gusto that are uncategorized, so salary-like uncategorized payroll deposits were found and should be recategorized. outcome_reflection should list transaction_id 50111 and 50112 so the next step can target those rows.",
  },
  {
    "name": "salary_check_iteration_2b_success_reflection",
    "batch": 1,
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": [
      "The execution successfully listed recent income transactions, confirming the presence of two deposits ($2500 from ADP PAYROLL and $2500 from Gusto) that appear to be salary but are currently uncategorized. The task requires fixing the categorization if they are salary. Since the categorization fix has not yet occurred, the status must remain IN_PROGRESS to pursue the required correction.",
    ],
    "execution_result": "(True, 'Successfully processed categorization request: Categorize all transactions originating from 'ADP PAYROLL' and 'Gusto' as 'Salary'.')",
    "ideal_response": "Reflection: COMPLETED; I see a success message that ADP PAYROLL and Gusto transactions were categorized as Salary, so the categorization fix and rule creation are done. outcome_reflection need not list transaction_id values; this payload has none and the task is finished.",
  },
  {
    "name": "salary_check_iteration_2b_fail_reflection",
    "batch": 1,
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": [
      "The execution successfully listed recent income transactions, confirming the presence of two deposits ($2500 from ADP PAYROLL and $2500 from Gusto) that appear to be salary but are currently uncategorized. The task requires fixing the categorization if they are salary. Since the categorization fix has not yet occurred, the status must remain IN_PROGRESS to pursue the required correction.",
    ],
    "execution_result": "(True, 'An unexpected error was encountered.')",
    "ideal_response": "Reflection: FAILED; I see an unexpected error in the execution result, so the categorization step failed. outcome_reflection has no transaction_id values to repeat from this payload.",
  },
  {
    "name": "salary_check_iteration_3_reflection",
    "batch": 1,
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": [
      "The execution successfully listed recent income transactions, confirming the presence of two deposits ($2500 from ADP PAYROLL and $2500 from Gusto) that appear to be salary but are currently uncategorized. The task requires fixing the categorization if they are salary. Since the categorization fix has not yet occurred, the status must remain IN_PROGRESS to pursue the required correction.",
      "The previous step successfully identified two potential salary transactions ($2500 from ADP PAYROLL and $2500 from Gusto) that needed categorization. However, the current execution step failed with a generic error message, preventing any further progress towards fixing the categorization or confirming the salary expectation. Therefore, the status is FAILED.",
    ],
    "execution_result": "(True, 'Successfully processed categorization request: Update transactions from source 'ADP PAYROLL' to category 'Salary'. Also, update transactions from source 'Gusto' to category 'Salary'.')",
    "ideal_response": "Reflection: COMPLETED; I see a success message that ADP PAYROLL and Gusto transactions were updated to Salary, so the retry succeeded. outcome_reflection need not list transaction_id values; this payload has none and the task is finished.",
  },
  # Shelter (batch 2): optional short_description summarizes scenario vs. ideal_response rubric.
  {
    "name": "shelter_check_iteration_1a_reflection",
    "batch": 2,
    "short_description": "Initial lookup: housing-sized payees already shelter_home; task satisfied.",
    "task_description": "Look into the user's rent or mortgage spending and make sure it's categorized correctly as shelter_home. Check if amount is expected, and if it isn't transactions might be in uncategorized. Look at the largest spending to see if it is indeed rent or mortgage and fix the categorization.",
    "previous_outcomes": None,
    "execution_result": "(True, '--- Top Spending ---\\nRecent Largest Spending Transactions:\\n- $2000 was paid to Apartments LLC on 2025-11-18 (Chase Total Checking **1563) categorized as shelter_home; transaction_id=60101.\\n- $2000 was paid to Apartments LLC on 2025-10-18 (Chase Total Checking **1563) categorized as shelter_home; transaction_id=60102.\\n- $2000 was paid to Apartments LLC on 2025-09-18 (Chase Total Checking **1563) categorized as shelter_home; transaction_id=60103.\\n- $1650 was paid to Bank of America on 2025-10-31 (Chase Total Checking **1563) categorized as shelter_home; transaction_id=60104.\\n- $1650 was paid to Bank of America on 2025-09-30 (Chase Total Checking **1563) categorized as shelter_home; transaction_id=60105.\\n- $1650 was paid to Bank of America on 2025-08-31 (Chase Total Checking **1563) categorized as shelter_home; transaction_id=60106.\\n- $1200 was paid to Community College on 2025-11-15 (Chase Total Checking **1563) categorized as education_tuition; transaction_id=60107.\\n- $1200 was paid to Community College on 2024-11-15 (Chase Total Checking **1563) categorized as education_tuition; transaction_id=60108.\\nTotal recent spending: spent $13350.\\n')",
    "ideal_response": "Reflection: COMPLETED; I see consistent payments of $2000 to Apartments LLC and $1650 to Bank of America correctly categorized as shelter_home, so rent and mortgage spending are present and categorized as shelter_home. outcome_reflection need not foreground transaction_id values; the task is complete and no row-specific follow-up is required.",
  },
  {
    "name": "shelter_check_iteration_1b_reflection",
    "batch": 2,
    "short_description": "Initial lookup: rent line mislabeled dining_out; mortgage line uncategorized—needs fixes.",
    "task_description": "Look into the user's rent or mortgage spending and make sure it's categorized correctly as shelter_home. Check if amount is expected, and if it isn't transactions might be in uncategorized. Look at the largest spending to see if it is indeed rent or mortgage and fix the categorization.",
    "previous_outcomes": None,
    "execution_result": "(True, '--- Top Spending ---\\nRecent Largest Spending Transactions:\\n- $2000 was paid to Apartments LLC on 2025-11-18 (Chase Total Checking **1563) categorized as meals_dining_out; transaction_id=60201.\\n- $2000 was paid to Apartments LLC on 2025-10-18 (Chase Total Checking **1563) categorized as shelter_home; transaction_id=60202.\\n- $2000 was paid to Apartments LLC on 2025-09-18 (Chase Total Checking **1563) categorized as shelter_home; transaction_id=60203.\\n- $1650 was paid to Bank of America on 2025-10-31 (Chase Total Checking **1563) categorized as uncategorized; transaction_id=60204.\\n- $1650 was paid to Bank of America on 2025-09-30 (Chase Total Checking **1563) categorized as shelter_home; transaction_id=60205.\\n- $1650 was paid to Bank of America on 2025-08-31 (Chase Total Checking **1563) categorized as shelter_home; transaction_id=60206.\\n- $1200 was paid to Community College on 2025-11-15 (Chase Total Checking **1563) categorized as education_tuition; transaction_id=60207.\\n- $1200 was paid to Community College on 2024-11-15 (Chase Total Checking **1563) categorized as education_tuition; transaction_id=60208.\\nTotal recent spending: spent $13350.\\n')",
    "ideal_response": "Reflection: IN_PROGRESS; I see a $2000 rent payment categorized as meals_dining_out and a $1650 mortgage payment categorized as uncategorized, so rent and mortgage spending are present but categorized incorrectly. outcome_reflection should list transaction_id 60201 and 60204 for the next categorization step.",
  },
  {
    "name": "shelter_check_iteration_1c_reflection",
    "batch": 2,
    "short_description": "Initial lookup: housing-sized payees already shelter_home; service fee in mix.",
    "task_description": "Look into the user's rent or mortgage spending and make sure it's categorized correctly as shelter_home. Check if amount is expected, and if it isn't transactions might be in uncategorized. Look at the largest spending to see if it is indeed rent or mortgage and fix the categorization.",
    "previous_outcomes": None,
    "execution_result": "(True, '--- Top Spending ---\\nRecent Largest Spending Transactions:\\n- $2000 was paid to Apartments LLC on 2025-11-18 (Chase Total Checking **1563) categorized as shelter_home; transaction_id=60301.\\n- $2000 was paid to Apartments LLC on 2025-10-18 (Chase Total Checking **1563) categorized as shelter_home; transaction_id=60302.\\n- $2000 was paid to Apartments LLC on 2025-09-18 (Chase Total Checking **1563) categorized as shelter_home; transaction_id=60303.\\n- $1650 was paid to Bank of America on 2025-10-31 (Chase Total Checking **1563) categorized as shelter_home; transaction_id=60304.\\n- $1650 was paid to Bank of America on 2025-09-30 (Chase Total Checking **1563) categorized as shelter_home; transaction_id=60305.\\n- $1650 was paid to Bank of America on 2025-08-31 (Chase Total Checking **1563) categorized as shelter_home; transaction_id=60306.\\n- $1200 was paid to Community College on 2025-11-15 (Chase Total Checking **1563) categorized as education_tuition; transaction_id=60307.\\n- $1200 was paid to Community College on 2024-11-15 (Chase Total Checking **1563) categorized as education_tuition; transaction_id=60308.\\n- $50 was paid to Apartments LLC on 2025-11-18 (Chase Total Checking **1563) categorized as bills_service_fees; transaction_id=60309.\\nTotal recent spending: spent $13400.\\n')",
    "ideal_response": "Reflection: COMPLETED; I see consistent payments of $2000 to Apartments LLC and $1650 to Bank of America correctly categorized as shelter_home, so rent and mortgage spending are present and categorized as shelter_home. outcome_reflection need not foreground transaction_id values; the task is complete and no row-specific follow-up is required.",
  },
  {
    "name": "shelter_check_iteration_1d_reflection",
    "batch": 2,
    "short_description": "Initial lookup: same-day landlord lines differ in amount; both shelter_home — COMPLETED.",
    "task_description": "Look into the user's rent or mortgage spending and make sure it's categorized correctly as shelter_home. Check if amount is expected, and if it isn't transactions might be in uncategorized. Look at the largest spending to see if it is indeed rent or mortgage and fix the categorization.",
    "previous_outcomes": None,
    "execution_result": "(True, '--- Top Spending ---\\nRecent Largest Spending Transactions:\\n- $2000 was paid to Apartments LLC on 2025-11-18 (Chase Total Checking **1563) categorized as shelter_home; transaction_id=60401.\\n- $2000 was paid to Apartments LLC on 2025-10-18 (Chase Total Checking **1563) categorized as shelter_home; transaction_id=60402.\\n- $2000 was paid to Apartments LLC on 2025-09-18 (Chase Total Checking **1563) categorized as shelter_home; transaction_id=60403.\\n- $1650 was paid to Bank of America on 2025-10-31 (Chase Total Checking **1563) categorized as shelter_home; transaction_id=60404.\\n- $1650 was paid to Bank of America on 2025-09-30 (Chase Total Checking **1563) categorized as shelter_home; transaction_id=60405.\\n- $1650 was paid to Bank of America on 2025-08-31 (Chase Total Checking **1563) categorized as shelter_home; transaction_id=60406.\\n- $1200 was paid to Community College on 2025-11-15 (Chase Total Checking **1563) categorized as education_tuition; transaction_id=60407.\\n- $1200 was paid to Community College on 2024-11-15 (Chase Total Checking **1563) categorized as education_tuition; transaction_id=60408.\\n- $1050 was paid to Apartments LLC on 2025-11-18 (Chase Total Checking **1563) categorized as shelter_home; transaction_id=60409.\\nTotal recent spending: spent $14400.\\n')",
    "ideal_response": "Reflection: COMPLETED; I see Apartments LLC and Bank of America housing-scale payments correctly categorized as shelter_home, including two shelter_home lines to Apartments LLC on 2025-11-18 for $2000 and $1050—different amounts on the same day are acceptable when categorization is correct. outcome_reflection need not foreground transaction_id values; the task is complete and no row-specific follow-up is required.",
  },
  {
    "name": "shelter_check_iteration_2b_success_reflection",
    "batch": 3,
    "short_description": "After prior IN_PROGRESS: categorization step fixed both bad housing rows.",
    "task_description": "Look into the user's rent or mortgage spending and make sure it's categorized correctly as shelter_home. Check if amount is expected, and if it isn't transactions might be in uncategorized. Look at the largest spending to see if it is indeed rent or mortgage and fix the categorization.",
    "previous_outcomes": [
      "The task requires checking amounts and fixing categorizations for rent/mortgage spending. The execution result successfully listed the top spending and revealed miscategorizations: a $2000 payment to Apartments LLC on 2025-11-18 is incorrectly labeled 'meals_dining_out', and a $1650 payment to Bank of America on 2025-10-31 is 'uncategorized'. Since these required fixes have not yet been applied, the workflow must continue to the remediation step. Rubric 0 does not apply as there are no two differing large housing payments on the same date.",
    ],
    "execution_result": "(True, 'Successfully processed categorization request: Categorize transactions of $2000 to 'Apartments LLC' on 2025-11-18 with category meals_dining_out to shelter_home and of $1650 to 'Bank of America' on 2025-10-31 with category uncategorized to shelter_home as 'shelter_home'.')",
    "ideal_response": "Reflection: COMPLETED; I see a success message confirming that the transactions for Apartments LLC and Bank of America have been successfully categorized as shelter_home, so the categorizations are now correct. outcome_reflection need not list transaction_id values; this payload has none and the task is finished.",
  },
  {
    "name": "shelter_check_iteration_2b_partial_reflection",
    "batch": 3,
    "short_description": "Partial fix: only Apartments LLC rule applied; BoA row still uncategorized.",
    "task_description": "Look into the user's rent or mortgage spending and make sure it's categorized correctly as shelter_home. Check if amount is expected, and if it isn't transactions might be in uncategorized. Look at the largest spending to see if it is indeed rent or mortgage and fix the categorization.",
    "previous_outcomes": [
      "The task requires checking amounts and fixing categorizations for rent/mortgage spending. The execution result successfully listed the top spending and revealed miscategorizations: a $2000 payment to Apartments LLC on 2025-11-18 is incorrectly labeled 'meals_dining_out', and a $1650 payment to Bank of America on 2025-10-31 is 'uncategorized'. Since these required fixes have not yet been applied, the workflow must continue to the remediation step. Rubric 0 does not apply as there are no two differing large housing payments on the same date.",
    ],
    "execution_result": "(True, 'Successfully processed categorization request: Categorize all transactions originating from 'Apartments LLC' as 'shelter_home'.')",
    "ideal_response": "Reflection: IN_PROGRESS or PARTIALLY_COMPLETED; I see a success message for 'Apartments LLC' only, but the Bank of America transaction is still uncategorized. This execution result has no transaction_id fields; outcome_reflection should still clearly identify the remaining $1650 Bank of America payment on 2025-10-31 (uncategorized) for the next step without inventing ids.",
  },
  {
    "name": "shelter_check_iteration_2b_fail_reflection",
    "batch": 3,
    "short_description": "Remediation step failed with generic unexpected error after issues were identified.",
    "task_description": "Look into the user's rent or mortgage spending and make sure it's categorized correctly as shelter_home. Check if amount is expected, and if it isn't transactions might be in uncategorized. Look at the largest spending to see if it is indeed rent or mortgage and fix the categorization.",
    "previous_outcomes": [
      "The task requires checking amounts and fixing categorizations for rent/mortgage spending. The execution result successfully listed the top spending and revealed miscategorizations: a $2000 payment to Apartments LLC on 2025-11-18 is incorrectly labeled 'meals_dining_out', and a $1650 payment to Bank of America on 2025-10-31 is 'uncategorized'. Since these required fixes have not yet been applied, the workflow must continue to the remediation step. Rubric 0 does not apply as there are no two differing large housing payments on the same date.",
    ],
    "execution_result": "(True, 'An unexpected error was encountered.')",
    "ideal_response": "Reflection: FAILED; I see an unexpected error in the execution result, so the categorization step failed. outcome_reflection has no transaction_id values to repeat from this payload.",
  },
  {
    "name": "shelter_check_iteration_3_reflection",
    "batch": 3,
    "short_description": "Retry after error: both miscategorized housing lines updated to shelter_home.",
    "task_description": "Look into the user's rent or mortgage spending and make sure it's categorized correctly as shelter_home. Check if amount is expected, and if it isn't transactions might be in uncategorized. Look at the largest spending to see if it is indeed rent or mortgage and fix the categorization.",
    "previous_outcomes": [
      "The task requires checking amounts and fixing categorizations for rent/mortgage spending. The execution result successfully listed the top spending and revealed miscategorizations: a $2000 payment to Apartments LLC on 2025-11-18 is incorrectly labeled 'meals_dining_out', and a $1650 payment to Bank of America on 2025-10-31 is 'uncategorized'. Since these required fixes have not yet been applied, the workflow must continue to the remediation step. Rubric 0 does not apply as there are no two differing large housing payments on the same date.",
      "The execution result indicates an unexpected error occurred during the last step, despite previous steps identifying a clear miscategorization that needed fixing. Since the error prevents the necessary correction, the task cannot proceed successfully.",
    ],
    "execution_result": "(True, 'Successfully processed categorization request: Categorize transactions of $2000 to 'Apartments LLC' on 2025-11-18 with category meals_dining_out to shelter_home and of $1650 to 'Bank of America' on 2025-10-31 with category uncategorized to shelter_home as 'shelter_home'.')",
    "ideal_response": "Reflection: COMPLETED; I see a success message confirming that the transactions for Apartments LLC and Bank of America have been successfully categorized as shelter_home, so the categorizations are now correct. outcome_reflection need not list transaction_id values; this payload has none and the task is finished.",
  },
]


def run_test(test_name_or_index_or_dict, optimizer: SelfReflectionWoCodeOptimizer = None):
  if isinstance(test_name_or_index_or_dict, int):
    tc = TEST_CASES[test_name_or_index_or_dict] if 0 <= test_name_or_index_or_dict < len(TEST_CASES) else None
  else:
    tc = next((t for t in TEST_CASES if t["name"] == test_name_or_index_or_dict), None)
  if not tc:
    return None
  print(f"\n# Test: **{tc['name']}**\n")
  return _run_test_with_logging(tc["task_description"], tc["previous_outcomes"], tc["execution_result"], optimizer, ideal_response=tc.get("ideal_response"))


def run_all_tests_batch(optimizer: SelfReflectionWoCodeOptimizer = None, batch_num: int = 1):
  if optimizer is None:
    optimizer = SelfReflectionWoCodeOptimizer()
  cases = [tc for tc in TEST_CASES if tc["batch"] == batch_num]
  batch_results = []
  label = "salary" if batch_num == 1 else "shelter" if batch_num == 2 else f"batch {batch_num}"
  for tc in cases:
    result = run_test(tc["name"], optimizer)
    batch_results.append((tc["name"], result))
  for name, result in batch_results:
    status = "unknown"
    try:
      parsed = json.loads(result)
      status = parsed.get("next_status", "unknown")
    except Exception:
      pass
    print(f"- {name}: next_status={status}")
  print()
  return batch_results


def main(
  test: str = None,
  run_batch: bool = False,
  batch_num: int = 1,
  no_thinking: bool = False,
  thinking_budget: int = None,
  max_output_tokens: int = None,
  model: str = None,
):
  tb = (0 if no_thinking else (thinking_budget if thinking_budget is not None else 700))
  kw = {"thinking_budget": tb}
  if max_output_tokens is not None:
    kw["max_output_tokens"] = max_output_tokens
  if model is not None:
    kw["model_name"] = model
  optimizer = SelfReflectionWoCodeOptimizer(**kw)

  if run_batch:
    run_all_tests_batch(optimizer, batch_num=batch_num)
    return

  if test is not None:
    if test.strip().lower() == "all":
      run_all_tests_batch(optimizer, batch_num=1)
      run_all_tests_batch(optimizer, batch_num=2)
      return
    test_val = int(test) if test.isdigit() else test
    run_test(test_val, optimizer)
    return
  print("Available test cases:")
  for i, tc in enumerate(TEST_CASES):
    line = f"  {i}: {tc['name']} (batch {tc['batch']})"
    blurb = tc.get("short_description")
    if blurb:
      line = f"{line} — {blurb}"
    print(line)


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--test", type=str)
  parser.add_argument("--no-thinking", action="store_true")
  parser.add_argument("--thinking-budget", type=int, default=None)
  parser.add_argument("--max-output-tokens", type=int, default=None)
  parser.add_argument("--model", type=str, default=None, help="Gemini model name (e.g. gemini-2.0-flash)")
  parser.add_argument(
    "--batch",
    type=int,
    nargs="?",
    const=1,
    default=None,
    metavar="N",
    help="Run test cases in group N: 1=salary, 2=shelter (omit N to use 1).",
  )
  args = parser.parse_args()
  if args.batch is not None and args.batch not in (1, 2):
    parser.error("batch N must be 1 (salary) or 2 (shelter)")
  batch_num = 1 if args.batch is None else args.batch
  run_batch = args.batch is not None
  main(
    test=args.test,
    run_batch=run_batch,
    batch_num=batch_num,
    no_thinking=args.no_thinking,
    thinking_budget=args.thinking_budget,
    max_output_tokens=args.max_output_tokens,
    model=args.model,
  )
