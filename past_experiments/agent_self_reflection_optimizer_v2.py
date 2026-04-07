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
        description="Concise summary of the execution result directly addressing the Task Description."
    )
    outcome_reflection: str = Field(
        description="Concise explanation of how the result contributes to the task, suggestions for improvement, and the exact next action to take."
    )
    next_status: Literal["COMPLETED", "PARTIALLY_COMPLETED", "FAILED", "IN_PROGRESS"] = Field(
        description="The accurate workflow state based on the evidence compared to the expected state."
    )


REFLECTION_WO_CODE_SYSTEM_PROMPT = """You are the reflection component of Strategizer AI.
Inputs:
- Task Description
- Previous Outcomes
- Execution Result (judge the inner message, not the outer wrapper like `(True, ...)`).

Goal:
Classify workflow next_status based ONLY on Execution Result.

Parse Rules:
- Explicit success/failure phrases are ground truth. Error wording overrides success flags.
- "Successfully processed" confirms requested mutations succeeded.

Rubric (stop at first match):
0) **Anomaly Check:** If checking expected amounts, multiple large transactions to the SAME payee on the SAME date with DIFFERENT amounts (ignore <$100 fees) is an anomaly → IN_PROGRESS.
1) COMPLETED: Task objective fully satisfied. For categorization, ALL targets present must be correctly categorized. Never mark COMPLETED if required fixes remain.
2) FAILED: The step errored, encountered an exception, or failed to perform the action.
3) PARTIALLY_COMPLETED: Meaningful progress made, but hard-stalled awaiting human input.
4) IN_PROGRESS: Healthy continuation. Usable evidence obtained, but clear next actions remain.

Quality Rules:
- Quote concrete facts. 
- Keep outcome_reflection highly concise.
- Ensure result_summary is brief.
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

  def generate_response(self, task_description: str, previous_outcomes: str, execution_result: str) -> str:
    request_text = types.Part.from_text(text=f"""**Task Description**: {task_description}

**Previous Outcomes**:
{previous_outcomes}

**Execution Result**:
{execution_result}

output:""")
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


def _run_test_with_logging(task_description: str, previous_outcomes: str, execution_result: str, optimizer: SelfReflectionWoCodeOptimizer | None = None, ideal_response: str | None = None):
  if optimizer is None:
    optimizer = SelfReflectionWoCodeOptimizer()
  print(f"## LLM Input\n")
  print(f"**Task Description**: {task_description}\n")
  print(f"**Execution Result**:\n```\n{execution_result}\n```\n")
  print(f"**Previous Outcomes**:\n{previous_outcomes}")
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
    "previous_outcomes": "None. This is the first attempt.",
    "execution_result": "(True, '--- Recent Income (Last 30 Days) ---\\nRecent Income Transactions:\\n- $1440 was received from CA State Payroll on 2025-11-18 (Chase Total Checking **1563) categorized as income_salary.\\n- $1340 was received from CA State Payroll on 2025-10-31 (Chase Total Checking **1563) categorized as income_salary.\\nTotal recent income: earned $2780.\\n')",
    "ideal_response": "Reflection: COMPLETED; I see two recent transactions from CA State Payroll categorized as income_salary, so salary is already detected and appears correctly categorized.",
  },
  {
    "name": "salary_check_iteration_1b_reflection",
    "batch": 1,
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "None. This is the first attempt.",
    "execution_result": "(True, '--- Recent Income (Last 30 Days) ---\\nRecent Income Transactions:\\n- $2500 was received from ADP PAYROLL on 2025-11-20 (Chase Total Checking **1563) categorized as uncategorized.\\n- $2500 was received from Gusto on 2025-10-22 (Chase Total Checking **1563) categorized as uncategorized.\\nTotal recent income: earned $5000.\\n')",
    "ideal_response": "Reflection: IN_PROGRESS; I see two recent large transactions from ADP PAYROLL and Gusto that are uncategorized, so salary-like uncategorized payroll deposits were found and should be recategorized.",
  },
  {
    "name": "salary_check_iteration_2b_success_reflection",
    "batch": 1,
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "Outcome 1: The initial step confirmed that no specific 'Salary' transactions were found, only uncategorized income. The execution result shows two recent income transactions of $2500 each, both categorized as 'uncategorized' and coming from known payroll providers (ADP PAYROLL and Gusto). This strongly suggests these are the salary payments that need to be re-categorized as 'Salary' as per the task description. The task is not fully accomplished because the re-categorization step has not yet been executed. The next logical step is to proceed with fixing the categorization of these identified transactions.",
    "execution_result": "(True, 'Successfully processed categorization request: Categorize all transactions originating from 'ADP PAYROLL' and 'Gusto' as 'Salary'.')",
    "ideal_response": "Reflection: COMPLETED; I see a success message that ADP PAYROLL and Gusto transactions were categorized as Salary, so the categorization fix and rule creation are done.",
  },
  {
    "name": "salary_check_iteration_2b_fail_reflection",
    "batch": 1,
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "Outcome 1: The initial step confirmed that no specific 'Salary' transactions were found, only uncategorized income. The execution result shows two recent income transactions of $2500 each, both categorized as 'uncategorized' and coming from known payroll providers (ADP PAYROLL and Gusto). This strongly suggests these are the salary payments that need to be re-categorized as 'Salary' as per the task description. The task is not fully accomplished because the re-categorization step has not yet been executed. The next logical step is to proceed with fixing the categorization of these identified transactions.",
    "execution_result": "(True, 'An unexpected error was encountered.')",
    "ideal_response": "Reflection: FAILED; I see an unexpected error in the execution result, so the categorization step failed.",
  },
  {
    "name": "salary_check_iteration_3_reflection",
    "batch": 1,
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "Outcome 1: The initial step confirmed that no specific 'Salary' transactions were found, only uncategorized income. The execution result shows two recent income transactions of $2500 each, both categorized as 'uncategorized' and coming from known payroll providers (ADP PAYROLL and Gusto). This strongly suggests these are the salary payments that need to be re-categorized as 'Salary' as per the task description. The task is not fully accomplished because the re-categorization step has not yet been executed. The next logical step is to proceed with fixing the categorization of these identified transactions.\nOutcome 2: The previous step identified two likely salary transactions ($2500 each from ADP PAYROLL and Gusto) that were uncategorized. The current execution step, which was intended to fix the categorization, resulted in an unexpected error: 'An unexpected error was encountered.'. This means the core objective of fixing the categorization was not achieved. Therefore, the task is incomplete, and the next step must be to retry the categorization fix, perhaps after logging or investigating the error if more context were available. Since I must provide the next logical step based only on the provided result, the next step should be to attempt the categorization fix again or report failure if retries are exhausted. Given the structure, I will mark it as IN_PROGRESS and assume the next step should be a retry or a different approach to categorization.",
    "execution_result": "(True, 'Successfully processed categorization request: Update transactions from source 'ADP PAYROLL' to category 'Salary'. Also, update transactions from source 'Gusto' to category 'Salary'.')",
    "ideal_response": "Reflection: COMPLETED; I see a success message that ADP PAYROLL and Gusto transactions were updated to Salary, so the retry succeeded.",
  },
  # Shelter (batch 2): optional short_description summarizes scenario vs. ideal_response rubric.
  {
    "name": "shelter_check_iteration_1a_reflection",
    "batch": 2,
    "short_description": "Initial lookup: housing-sized payees already shelter_home; task satisfied.",
    "task_description": "Look into the user's rent or mortgage spending and make sure it's categorized correctly as shelter_home. Check if amount is expected, and if it isn't transactions might be in uncategorized. Look at the largest spending to see if it is indeed rent or mortgage and fix the categorization.",
    "previous_outcomes": "None. This is the first attempt.",
    "execution_result": "(True, '--- Top Spending ---\\nRecent Largest Spending Transactions:\\n- $2000 was paid to Apartments LLC on 2025-11-18 (Chase Total Checking **1563) categorized as shelter_home.\\n- $2000 was paid to Apartments LLC on 2025-10-18 (Chase Total Checking **1563) categorized as shelter_home.\\n- $2000 was paid to Apartments LLC on 2025-09-18 (Chase Total Checking **1563) categorized as shelter_home.\\n- $1650 was paid to Bank of America on 2025-10-31 (Chase Total Checking **1563) categorized as shelter_home.\\n- $1650 was paid to Bank of America on 2025-09-30 (Chase Total Checking **1563) categorized as shelter_home.\\n- $1650 was paid to Bank of America on 2025-08-31 (Chase Total Checking **1563) categorized as shelter_home.\\n- $1200 was paid to Community College on 2025-11-15 (Chase Total Checking **1563) categorized as education_tuition.\\n- $1200 was paid to Community College on 2024-11-15 (Chase Total Checking **1563) categorized as education_tuition.\\nTotal recent spending: spent $13350.\\n')",
    "ideal_response": "Reflection: COMPLETED; I see consistent payments of $2000 to Apartments LLC and $1650 to Bank of America correctly categorized as shelter_home, so rent and mortgage spending are present and categorized as shelter_home.",
  },
  {
    "name": "shelter_check_iteration_1b_reflection",
    "batch": 2,
    "short_description": "Initial lookup: rent line mislabeled dining_out; mortgage line uncategorized—needs fixes.",
    "task_description": "Look into the user's rent or mortgage spending and make sure it's categorized correctly as shelter_home. Check if amount is expected, and if it isn't transactions might be in uncategorized. Look at the largest spending to see if it is indeed rent or mortgage and fix the categorization.",
    "previous_outcomes": "None. This is the first attempt.",
    "execution_result": "(True, '--- Top Spending ---\\nRecent Largest Spending Transactions:\\n- $2000 was paid to Apartments LLC on 2025-11-18 (Chase Total Checking **1563) categorized as meals_dining_out.\\n- $2000 was paid to Apartments LLC on 2025-10-18 (Chase Total Checking **1563) categorized as shelter_home.\\n- $2000 was paid to Apartments LLC on 2025-09-18 (Chase Total Checking **1563) categorized as shelter_home.\\n- $1650 was paid to Bank of America on 2025-10-31 (Chase Total Checking **1563) categorized as uncategorized.\\n- $1650 was paid to Bank of America on 2025-09-30 (Chase Total Checking **1563) categorized as shelter_home.\\n- $1650 was paid to Bank of America on 2025-08-31 (Chase Total Checking **1563) categorized as shelter_home.\\n- $1200 was paid to Community College on 2025-11-15 (Chase Total Checking **1563) categorized as education_tuition.\\n- $1200 was paid to Community College on 2024-11-15 (Chase Total Checking **1563) categorized as education_tuition.\\nTotal recent spending: spent $13350.\\n')",
    "ideal_response": "Reflection: IN_PROGRESS; I see a $2000 rent payment categorized as meals_dining_out and a $1650 mortgage payment categorized as uncategorized, so rent and mortgage spending are present but categorized incorrectly.",
  },
  {
    "name": "shelter_check_iteration_1c_reflection",
    "batch": 2,
    "short_description": "Initial lookup: housing-sized payees already shelter_home; service fee in mix.",
    "task_description": "Look into the user's rent or mortgage spending and make sure it's categorized correctly as shelter_home. Check if amount is expected, and if it isn't transactions might be in uncategorized. Look at the largest spending to see if it is indeed rent or mortgage and fix the categorization.",
    "previous_outcomes": "None. This is the first attempt.",
    "execution_result": "(True, '--- Top Spending ---\\nRecent Largest Spending Transactions:\\n- $2000 was paid to Apartments LLC on 2025-11-18 (Chase Total Checking **1563) categorized as shelter_home.\\n- $2000 was paid to Apartments LLC on 2025-10-18 (Chase Total Checking **1563) categorized as shelter_home.\\n- $2000 was paid to Apartments LLC on 2025-09-18 (Chase Total Checking **1563) categorized as shelter_home.\\n- $1650 was paid to Bank of America on 2025-10-31 (Chase Total Checking **1563) categorized as shelter_home.\\n- $1650 was paid to Bank of America on 2025-09-30 (Chase Total Checking **1563) categorized as shelter_home.\\n- $1650 was paid to Bank of America on 2025-08-31 (Chase Total Checking **1563) categorized as shelter_home.\\n- $1200 was paid to Community College on 2025-11-15 (Chase Total Checking **1563) categorized as education_tuition.\\n- $1200 was paid to Community College on 2024-11-15 (Chase Total Checking **1563) categorized as education_tuition.\\n- $50 was paid to Apartments LLC on 2025-11-18 (Chase Total Checking **1563) categorized as bills_service_fees.\\nTotal recent spending: spent $13400.\\n')",
    "ideal_response": "Reflection: COMPLETED; I see consistent payments of $2000 to Apartments LLC and $1650 to Bank of America correctly categorized as shelter_home, so rent and mortgage spending are present and categorized as shelter_home.",
  },
  {
    "name": "shelter_check_iteration_1d_reflection",
    "batch": 2,
    "short_description": "Initial lookup: housing-sized payees already shelter_home; unexpected large transaction in mix.",
    "task_description": "Look into the user's rent or mortgage spending and make sure it's categorized correctly as shelter_home. Check if amount is expected, and if it isn't transactions might be in uncategorized. Look at the largest spending to see if it is indeed rent or mortgage and fix the categorization.",
    "previous_outcomes": "None. This is the first attempt.",
    "execution_result": "(True, '--- Top Spending ---\\nRecent Largest Spending Transactions:\\n- $2000 was paid to Apartments LLC on 2025-11-18 (Chase Total Checking **1563) categorized as shelter_home.\\n- $2000 was paid to Apartments LLC on 2025-10-18 (Chase Total Checking **1563) categorized as shelter_home.\\n- $2000 was paid to Apartments LLC on 2025-09-18 (Chase Total Checking **1563) categorized as shelter_home.\\n- $1650 was paid to Bank of America on 2025-10-31 (Chase Total Checking **1563) categorized as shelter_home.\\n- $1650 was paid to Bank of America on 2025-09-30 (Chase Total Checking **1563) categorized as shelter_home.\\n- $1650 was paid to Bank of America on 2025-08-31 (Chase Total Checking **1563) categorized as shelter_home.\\n- $1200 was paid to Community College on 2025-11-15 (Chase Total Checking **1563) categorized as education_tuition.\\n- $1200 was paid to Community College on 2024-11-15 (Chase Total Checking **1563) categorized as education_tuition.\\n- $1050 was paid to Apartments LLC on 2025-11-18 (Chase Total Checking **1563) categorized as shelter_home.\\nTotal recent spending: spent $14400.\\n')",
    "ideal_response": "Reflection: IN_PROGRESS; I see two large but different amounts ($2000 and $1050) sent to Apartments LLC on the exact same date (2025-11-18), which violates the expected amount check, so further investigation is needed.",
  },
  {
    "name": "shelter_check_iteration_2b_success_reflection",
    "batch": 3,
    "short_description": "After prior IN_PROGRESS: categorization step fixed both bad housing rows.",
    "task_description": "Look into the user's rent or mortgage spending and make sure it's categorized correctly as shelter_home. Check if amount is expected, and if it isn't transactions might be in uncategorized. Look at the largest spending to see if it is indeed rent or mortgage and fix the categorization.",
    "previous_outcomes": "Outcome 1: The task requires checking rent/mortgage spending categorization and fixing incorrect ones. The execution result shows the top spending transactions. One transaction of $2000 to 'Apartments LLC' on 2025-11-18 is incorrectly categorized as 'meals_dining_out'. Other large transactions to 'Apartments LLC' and 'Bank of America' seem related to housing but one is missing categorization ($1650 to Bank of America). Since a clear miscategorization was found ('Apartments LLC' transaction) and the goal is to fix categorization, the process is not complete. A next step is needed to correct the identified miscategorization.",
    "execution_result": "(True, 'Successfully processed categorization request: Categorize transactions of $2000 to 'Apartments LLC' on 2025-11-18 with category meals_dining_out to shelter_home and of $1650 to 'Bank of America' on 2025-10-31 with category uncategorized to shelter_home as 'shelter_home'.')",
    "ideal_response": "Reflection: COMPLETED; I see a success message confirming that the transactions for Apartments LLC and Bank of America have been successfully categorized as shelter_home, so the categorizations are now correct.",
  },
  {
    "name": "shelter_check_iteration_2b_partial_reflection",
    "batch": 3,
    "short_description": "Partial fix: only Apartments LLC rule applied; BoA row still uncategorized.",
    "task_description": "Look into the user's rent or mortgage spending and make sure it's categorized correctly as shelter_home. Check if amount is expected, and if it isn't transactions might be in uncategorized. Look at the largest spending to see if it is indeed rent or mortgage and fix the categorization.",
    "previous_outcomes": "Outcome 1: The task requires checking rent/mortgage spending categorization and fixing incorrect ones. The execution result shows the top spending transactions. One transaction of $2000 to 'Apartments LLC' on 2025-11-18 is incorrectly categorized as 'meals_dining_out'. Other large transactions to 'Apartments LLC' and 'Bank of America' seem related to housing but one is missing categorization ($1650 to Bank of America). Since a clear miscategorization was found ('Apartments LLC' transaction) and the goal is to fix categorization, the process is not complete. A next step is needed to correct the identified miscategorization.",
    "execution_result": "(True, 'Successfully processed categorization request: Categorize all transactions originating from 'Apartments LLC' as 'shelter_home'.')",
    "ideal_response": "Reflection: IN_PROGRESS or PARTIALLY_COMPLETED; I see a success message for 'Apartments LLC' only, but the Bank of America transaction is still uncategorized.",
  },
  {
    "name": "shelter_check_iteration_2b_fail_reflection",
    "batch": 3,
    "short_description": "Remediation step failed with generic unexpected error after issues were identified.",
    "task_description": "Look into the user's rent or mortgage spending and make sure it's categorized correctly as shelter_home. Check if amount is expected, and if it isn't transactions might be in uncategorized. Look at the largest spending to see if it is indeed rent or mortgage and fix the categorization.",
    "previous_outcomes": "Outcome 1: The task requires checking rent/mortgage spending categorization and fixing incorrect ones. The execution result shows the top spending transactions. One transaction of $2000 to 'Apartments LLC' on 2025-11-18 is incorrectly categorized as 'meals_dining_out'. Other large transactions to 'Apartments LLC' and 'Bank of America' seem related to housing but one is missing categorization ($1650 to Bank of America). Since a clear miscategorization was found ('Apartments LLC' transaction) and the goal is to fix categorization, the process is not complete. A next step is needed to correct the identified miscategorization.",
    "execution_result": "(True, 'An unexpected error was encountered.')",
    "ideal_response": "Reflection: FAILED; I see an unexpected error in the execution result, so the categorization step failed.",
  },
  {
    "name": "shelter_check_iteration_3_reflection",
    "batch": 3,
    "short_description": "Retry after error: both miscategorized housing lines updated to shelter_home.",
    "task_description": "Look into the user's rent or mortgage spending and make sure it's categorized correctly as shelter_home. Check if amount is expected, and if it isn't transactions might be in uncategorized. Look at the largest spending to see if it is indeed rent or mortgage and fix the categorization.",
    "previous_outcomes": "Outcome 1: The task requires checking rent/mortgage spending categorization and fixing incorrect ones. The execution result shows the top spending transactions. One transaction of $2000 to 'Apartments LLC' on 2025-11-18 is incorrectly categorized as 'meals_dining_out'. Other large transactions to 'Apartments LLC' and 'Bank of America' seem related to housing but one is missing categorization ($1650 to Bank of America). Since a clear miscategorization was found ('Apartments LLC' transaction) and the goal is to fix categorization, the process is not complete. A next step is needed to correct the identified miscategorization.\nOutcome 2: The execution result indicates an unexpected error occurred during the last step, despite previous steps identifying a clear miscategorization that needed fixing. Since the error prevents the necessary correction, the task cannot proceed successfully.",
    "execution_result": "(True, 'Successfully processed categorization request: Categorize transactions of $2000 to 'Apartments LLC' on 2025-11-18 with category meals_dining_out to shelter_home and of $1650 to 'Bank of America' on 2025-10-31 with category uncategorized to shelter_home as 'shelter_home'.')",
    "ideal_response": "Reflection: COMPLETED; I see a success message confirming that the transactions for Apartments LLC and Bank of America have been successfully categorized as shelter_home, so the categorizations are now correct.",
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
