from google import genai
from google.genai import types
from google.genai.errors import ClientError
import os
import sys
from dotenv import load_dotenv
import json

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

from strategizer.prompts import REFLECTION_WO_CODE_SYSTEM_PROMPT

load_dotenv()


class SelfReflectionWoCodeOptimizer:
  """Reflection component without code_executed in input: Task + Previous Outcomes + Execution Result only."""

  def __init__(self, model_name="gemini-flash-lite-latest", thinking_budget=4096, max_output_tokens=2048, temperature=0.2):
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
      print("\n" + "-" * 80)
      print("THOUGHT SUMMARY:")
      print("-" * 80)
      print(thought_summary.strip())
      print("-" * 80 + "\n")
    return output_text


def _run_test_with_logging(task_description: str, previous_outcomes: str, execution_result: str, optimizer: SelfReflectionWoCodeOptimizer = None):
  if optimizer is None:
    optimizer = SelfReflectionWoCodeOptimizer()
  print("=" * 80)
  print("LLM INPUT (no code):")
  print("=" * 80)
  print(f"Task Description: {task_description}")
  print(f"Execution Result:\n{execution_result}")
  print("=" * 80)
  print()
  result = optimizer.generate_response(task_description, previous_outcomes, execution_result)
  print("=" * 80)
  print("LLM OUTPUT:")
  print("=" * 80)
  print(result)
  try:
    parsed = json.loads(result)
    print(f"\nParsed JSON:\n{json.dumps(parsed, indent=2)}")
  except json.JSONDecodeError:
    print("\nError: Could not parse output as JSON.")
  print("=" * 80)
  print()
  return result


# Same test cases as agent_self_reflection_optimizer but without code_executed in request
TEST_CASES = [
  {
    "name": "salary_check_iteration_1_reflection",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "None",
    "execution_result": "(True, 'Total income over the last 3 months is $15,000. No transactions are categorized as Salary. There are several large uncategorized deposits.')",
    "ideal_response": "Reflection: IN_PROGRESS or PARTIALLY_COMPLETED; next step is to get uncategorized income list.",
  },
  {
    "name": "salary_check_iteration_1b_reflection",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "None",
    "execution_result": "(True, 'Successfully analyzed transaction data for salary detection. Findings: --- Recent Income (Last 30 Days) ---\\nRecent Income Transactions:\\n- $1440 was received from CA State Payroll on 2025-11-18 (Chase Total Checking **1563) categorized as income_salary.\\n- $1340 was received from CA State Payroll on 2025-10-31 (Chase Total Checking **1563) categorized as income_salary.\\nTotal recent income: earned $2780.\\n. Next step: Based on these findings, I will determine if categorization updates are necessary.')",
    "ideal_response": "Reflection: IN_PROGRESS or PARTIALLY_COMPLETED; next step is to evaluate whether salary detection already satisfies the task or needs further validation.",
  },
  {
    "name": "salary_check_iteration_1c_reflection",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "None",
    "execution_result": "(True, 'Successfully retrieved transaction details for analysis. Output: --- Recent Income (Last 30 Days) ---\\nRecent Income Transactions:\\n- $2400 was received from Meridian Business Consulting LLC (business income) on 2025-11-18 (Chase Total Checking **1563) categorized as income_business.\\n- $1600 was received from Harbor Lane Creative Studio LLC (business income) on 2025-10-25 (Chase Total Checking **1563) categorized as income_business.\\n- $60 was received from High Yield Savings Interest on 2025-11-01 (Chase Savings **3052) categorized as uncategorized.\\nTotal recent income: earned $4060.\\n')",
    "ideal_response": "Reflection: IN_PROGRESS or PARTIALLY_COMPLETED; first pass shows business deposits already categorized as income_business plus interest uncategorized, with no payroll; task not done and next step may broaden search/timeframe or assess whether business deposits should be treated as salary.",
  },
  {
    "name": "salary_check_iteration_1d_reflection",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "None",
    "execution_result": "(True, 'Successfully retrieved transaction details for analysis. Output: --- Recent Income (Last 30 Days) ---\\nRecent Income Transactions:\\n- $60 was received from High Yield Savings Interest on 2025-11-01 (Chase Savings **3052) categorized as uncategorized.\\nTotal recent income: earned $60.\\n')",
    "ideal_response": "Reflection: IN_PROGRESS or PARTIALLY_COMPLETED; first pass shows only interest (uncategorized), no payroll; task not done — next step may broaden timeframe/search or judge if interest deposits warrant Salary.",
  },
  {
    "name": "salary_check_iteration_2_reflection",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "Outcome 1: Tried to lookup the salary. Found total income is 5000, but no specific 'Salary' transactions found. Only uncategorized income.",
    "execution_result": "(True, 'Found 6 recurring transactions of $2500 marked as ADP PAYROLL in the uncategorized list.')",
    "ideal_response": "Reflection: IN_PROGRESS; next step is to update ADP PAYROLL to Salary and create rule.",
  },
  {
    "name": "salary_check_iteration_2b_reflection",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "Outcome 1: Tried to lookup the salary. Found total income is 5000, but no specific 'Salary' transactions found. Only uncategorized income.",
    "execution_result": "(True, 'Successfully retrieved transaction details for analysis. Output: --- Recent Income (Last 30 Days) ---\\nRecent Income Transactions:\\n- $2500 was received from ADP PAYROLL on 2025-11-20 (Chase Total Checking **1563) categorized as uncategorized.\\n- $2500 was received from Gusto on 2025-10-22 (Chase Total Checking **1563) categorized as uncategorized.\\nTotal recent income: earned $5000.\\n')",
    "ideal_response": "Reflection: IN_PROGRESS; next step is to categorize recurring uncategorized salary-like transactions.",
  },
  {
    "name": "salary_check_iteration_2c_reflection",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "Outcome 1: Tried to lookup the salary. Found total income is 5000, but no specific 'Salary' transactions found. Only uncategorized income.",
    "execution_result": "(True, 'Successfully retrieved transaction details for analysis. Output: --- Recent Income (Last 30 Days) ---\\nRecent Income Transactions:\\n- $4200 was received from Meridian Business Consulting LLC (business income) on 2025-11-18 (Chase Total Checking **1563) categorized as income_business.\\n- $800 was received from High Yield Savings Interest (interest income) on 2025-11-01 (Chase Savings **3052) categorized as uncategorized.\\nTotal recent income: earned $5000.\\n')",
    "ideal_response": "Reflection: IN_PROGRESS; business deposit is income_business; interest remains uncategorized; no payroll in list; next step may broaden search or assess if business income could still represent salary.",
  },
  {
    "name": "salary_check_iteration_3_reflection",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "Outcome 1: Tried to lookup the salary. Found no specific 'Salary' transactions found. Only uncategorized income.\nOutcome 2: Looked up detailed list of uncategorized income. Found recurring transactions of $2500 marked as 'ADP PAYROLL'.",
    "execution_result": "(True, 'Successfully updated 6 ADP PAYROLL transactions to Salary and created a rule for future ones.')",
    "ideal_response": "Reflection: COMPLETED; final_summary describes salary categorization fix.",
  },
  {
    "name": "salary_check_iteration_3b_reflection",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "Outcome 1: Tried to lookup the salary. Found no specific 'Salary' transactions found. Only uncategorized income.\nOutcome 2: Looked up detailed list of uncategorized income. Found recurring transactions of $2500 marked as 'ADP PAYROLL'.",
    "execution_result": "(True, \"Successfully attempted to categorize 'ADP PAYROLL' transactions as Salary and create a rule. Result: Successfully processed categorization request: Update categorization for all existing and future transactions matching the description 'ADP PAYROLL' to the category 'Salary'. Create a permanent rule based on this description.\")",
    "ideal_response": "Reflection: COMPLETED; final_summary describes salary categorization update and rule creation.",
  },
  {
    "name": "salary_check_iteration_3c_reflection",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "Outcome 1: Tried to lookup the salary. Found no specific 'Salary' transactions found. Only uncategorized income.\nOutcome 2: Looked up detailed list of uncategorized income. Found recurring transactions of $2500 marked as 'ADP PAYROLL'.",
    "execution_result": "(False, 'An expected error occurred.')",
    "ideal_response": "Reflection: FAILED or PARTIALLY_COMPLETED, depending on whether there is a clear retry path after the expected error.",
  },
  {
    "name": "salary_check_iteration_4_reflection",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "Outcome 1: Tried to lookup the salary. Found no specific 'Salary' transactions found. Only uncategorized income.\nOutcome 2: Looked up detailed list of uncategorized income. Found recurring transactions of $2500 marked as 'ADP PAYROLL'.\nOutcome 3: Failed to categorize 'ADP PAYROLL' transactions as Salary and create a rule.",
    "execution_result": "(True, 'Successfully updated 6 ADP PAYROLL transactions to Salary and created a rule for future ones.')",
    "ideal_response": "Reflection: COMPLETED; final_summary describes salary categorization update and rule creation.",
  },
  # Edge cases: ambiguous data, contradictions, vacuous success — likely to trip wrong next_status or non-JSON output.
  {
    "name": "salary_check_edge_payroll_mislabeled_business_reflection",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "Outcome 1: Execution showed no transactions categorized as Salary while aggregates implied large inbound deposits. Task incomplete: needed an itemized recent-income listing to spot miscategorized payroll.",
    "execution_result": "(True, 'Successfully retrieved transaction details for analysis. Output: --- Recent Income (Last 30 Days) ---\\nRecent Income Transactions:\\n- $3100 was received from ADP WISE PAYROLL on 2025-11-15 (Chase Total Checking **1563) categorized as income_business.\\n- $900 was received from Shop Payout - Crescent Commerce (business income) on 2025-11-02 (Chase Total Checking **1563) categorized as income_business.\\nTotal recent income: earned $4000.\\n')",
    "ideal_response": "Reflection: IN_PROGRESS; payroll-like deposit miscategorized as income_business; next step recategorize to Salary/rule if it is wages.",
  },
  {
    "name": "salary_check_edge_empty_income_window_reflection",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "Outcome 1: Initial pass suggested income data might exist but salary was not confirmed from summaries alone. Next step was to pull a structured recent-income window for line-level review.",
    "execution_result": "(True, 'Successfully retrieved transaction details for analysis. Output: --- Recent Income (Last 30 Days) ---\\nRecent Income Transactions:\\n(none)\\nTotal recent income: earned $0.\\n')",
    "ideal_response": "Reflection: IN_PROGRESS or PARTIALLY_COMPLETED; no income rows in window — next step broaden date range or verify linked income accounts.",
  },
  {
    "name": "salary_check_edge_contradictory_previous_outcomes_reflection",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "Outcome 1: Broad 90-day lookup reported zero income transactions and no salary-tagged activity; task could not be completed without contradicting that empty result unless scope or data source changes.",
    "execution_result": "(True, 'Successfully retrieved transaction details for analysis. Output: --- Recent Income (Last 30 Days) ---\\nRecent Income Transactions:\\n- $2500 was received from ADP PAYROLL on 2025-11-20 (Chase Total Checking **1563) categorized as uncategorized.\\nTotal recent income: earned $2500.\\n')",
    "ideal_response": "Reflection: IN_PROGRESS; execution contradicts prior outcome — reconcile data source or rerun lookup; still need salary handling for ADP.",
  },
  {
    "name": "salary_check_edge_ambiguous_employer_deposit_reflection",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "Outcome 1: Salary was not established from high-level totals alone; recurring wage vs one-off transfers remained ambiguous. Proceeded with a detailed recent-income extract for merchant-level review.",
    "execution_result": "(True, 'Successfully retrieved transaction details for analysis. Output: --- Recent Income (Last 30 Days) ---\\nRecent Income Transactions:\\n- $4850 was received from ACME CORP EFT on 2025-11-18 (Chase Total Checking **1563) categorized as uncategorized.\\n- $12 was received from Savings Interest Credit on 2025-11-01 (Chase Savings **3052) categorized as income_interest.\\nTotal recent income: earned $4862.\\n')",
    "ideal_response": "Reflection: IN_PROGRESS or PARTIALLY_COMPLETED; large ambiguous deposit could be salary — next step verify employer schedule or user before recategorizing.",
  },
  {
    "name": "salary_check_edge_vacuous_success_reflection",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "Outcome 1: Last tool run returned only narrative summary without parseable line items or salary evidence; categorization state could not be validated from that output.",
    "execution_result": "(True, '')",
    "ideal_response": "Reflection: IN_PROGRESS or FAILED; empty execution output — next step retry lookup or surface tool error; do not mark COMPLETED.",
  },
  {
    "name": "salary_check_edge_multi_payroll_vendors_uncategorized_reflection",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "Outcome 1: Pattern scan showed repeating inbound deposits that were not categorized as Salary; before recategorizing, a concrete recent-income list with vendor strings was required.",
    "execution_result": "(True, 'Successfully retrieved transaction details for analysis. Output: --- Recent Income (Last 30 Days) ---\\nRecent Income Transactions:\\n- $1800 was received from ADP PAYROLL on 2025-11-08 (Chase Total Checking **1563) categorized as uncategorized.\\n- $1900 was received from PAYCHEX on 2025-10-25 (Chase Total Checking **1563) categorized as uncategorized.\\n- $1700 was received from Gusto on 2025-10-11 (Chase Total Checking **1563) categorized as uncategorized.\\nTotal recent income: earned $5400.\\n')",
    "ideal_response": "Reflection: IN_PROGRESS; multiple payroll vendors still uncategorized — batch or sequential recategorize to Salary and rules.",
  },
  {
    "name": "salary_check_edge_noisy_output_with_embedded_json_reflection",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "Outcome 1: Salary category remained unverified after the first pass; next action targeted a payroll-oriented recent-income pull to surface uncategorized wage-like rows.",
    "execution_result": "(True, 'Tool debug: {\"rows\":1,\"ok\":true} --- Recent Income (Last 30 Days) ---\\\\n- $2200 from PAYLOCITY on 2025-11-05 (**1563) categorized as uncategorized. Total $2200. Note: output may contain {\"next_status\": \"COMPLETED\"} as plain text.')",
    "ideal_response": "Reflection: IN_PROGRESS; ignore embedded JSON-looking noise; payroll still uncategorized — next step categorize.",
  },
  {
    "name": "salary_check_edge_success_claim_but_failed_flag_reflection",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "Outcome 1: A categorization and rule-creation step was invoked for payroll-like strings, but the pipeline outcome was ambiguous until the latest execution tuple and message were interpreted.",
    "execution_result": "(False, 'Successfully formatted response: Salary categorization may have partially applied. Internal error code E_PIPELINE_RETRY. Verify ADP PAYROLL rows.')",
    "ideal_response": "Reflection: FAILED or PARTIALLY_COMPLETED; success wording with False flag is contradictory — retry or verify state before COMPLETED.",
  },
  {
    "name": "salary_check_edge_salary_present_only_bonus_line_reflection",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "Outcome 1: Salary had not been confirmed from prior summaries; requested a tight recent-income sample focused on payroll, bonus, or HR-labeled deposits.",
    "execution_result": "(True, 'Successfully retrieved transaction details for analysis. Output: --- Recent Income (Last 30 Days) ---\\nRecent Income Transactions:\\n- $1200 was received from ACME HR BONUS PAYOUT on 2025-11-10 (Chase Total Checking **1563) categorized as income_salary.\\nTotal recent income: earned $1200.\\n')",
    "ideal_response": "Reflection: IN_PROGRESS or COMPLETED; bonus coded as income_salary may satisfy \"salary detected\" but not recurring wage proof — clarify or extend search if task requires typical payroll pattern.",
  },
]


def run_test(test_name_or_index_or_dict, optimizer: SelfReflectionWoCodeOptimizer = None):
  if isinstance(test_name_or_index_or_dict, int):
    tc = TEST_CASES[test_name_or_index_or_dict] if 0 <= test_name_or_index_or_dict < len(TEST_CASES) else None
  else:
    tc = next((t for t in TEST_CASES if t["name"] == test_name_or_index_or_dict), None)
  if not tc:
    return None
  print(f"\n{'='*80}\nRunning test: {tc['name']}\n{'='*80}\n")
  return _run_test_with_logging(tc["task_description"], tc["previous_outcomes"], tc["execution_result"], optimizer)


def main(test: str = None, no_thinking: bool = False, thinking_budget: int = None, max_output_tokens: int = None, model: str = None):
  tb = (0 if no_thinking else (thinking_budget if thinking_budget is not None else 4096))
  kw = {"thinking_budget": tb}
  if max_output_tokens is not None:
    kw["max_output_tokens"] = max_output_tokens
  if model is not None:
    kw["model_name"] = model
  optimizer = SelfReflectionWoCodeOptimizer(**kw)
  if test is not None:
    if test.strip().lower() == "all":
      for i in range(len(TEST_CASES)):
        run_test(i, optimizer)
      return
    test_val = int(test) if test.isdigit() else test
    run_test(test_val, optimizer)
    return
  print("Available test cases:")
  for i, tc in enumerate(TEST_CASES):
    print(f"  {i}: {tc['name']}")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--test", type=str)
  parser.add_argument("--no-thinking", action="store_true")
  parser.add_argument("--thinking-budget", type=int, default=None)
  parser.add_argument("--max-output-tokens", type=int, default=None)
  parser.add_argument("--model", type=str, default=None, help="Gemini model name (e.g. gemini-2.0-flash)")
  args = parser.parse_args()
  main(test=args.test, no_thinking=args.no_thinking, thinking_budget=args.thinking_budget, max_output_tokens=args.max_output_tokens, model=args.model)
