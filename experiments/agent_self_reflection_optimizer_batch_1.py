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

load_dotenv()

REFLECTION_WO_CODE_SYSTEM_PROMPT = """You are the reflection component of Strategizer AI.
Inputs:
- Task Description
- Previous Outcomes
- Execution Result (the most recent step output)

Goal:
Decide the correct workflow state based on concrete evidence in Execution Result + context from Previous Outcomes.

Decision rubric (apply in order):
1) COMPLETED
   - Choose COMPLETED when the task objective is satisfied by the latest result.
   - If task asks to detect salary and salary is already clearly present/categorized, that can be COMPLETED without extra speculative checks.
2) FAILED
   - Choose FAILED when the latest result indicates an error, contradiction, or blocked action and there is no verified successful remediation in that same step.
   - Error-like outputs (for example: "unexpected error", exceptions, tool failure wording) should generally map to FAILED.
3) PARTIALLY_COMPLETED
   - Choose when meaningful progress was made, but the run reached a hard stop and cannot continue from current step without external intervention.
4) IN_PROGRESS
   - Choose only when the task is incomplete and there is a clear, feasible next action.

Quality rules:
- Prefer evidence over speculation.
- Use Previous Outcomes to infer intent and avoid losing context.
- Keep reflection concise and specific to observed results.
- Do not invent facts not present in inputs.
- If Execution Result is tuple-like and contains success=True but error wording in message, treat outcome as failed execution evidence (not successful completion).

Output strict JSON object:
{
  "reflection": "Concise evidence-based reasoning.",
  "next_status": "COMPLETED | PARTIALLY_COMPLETED | FAILED | IN_PROGRESS",
  "final_summary": "User-facing summary only for terminal states (COMPLETED/FAILED/PARTIALLY_COMPLETED); otherwise null."
}
"""


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
  print(f"Previous Outcomes:\n{previous_outcomes}")
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


# Same salary test sequence as strategizer batch 1; each execution_result represents
# the prior strategizer run output that self-reflection receives as input.
TEST_CASES = [
  {
    "name": "salary_check_iteration_1a_reflection",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "None. This is the first attempt.",
    "execution_result": "(True, '--- Recent Income (Last 30 Days) ---\\nRecent Income Transactions:\\n- $1440 was received from CA State Payroll on 2025-11-18 (Chase Total Checking **1563) categorized as income_salary.\\n- $1340 was received from CA State Payroll on 2025-10-31 (Chase Total Checking **1563) categorized as income_salary.\\nTotal recent income: earned $2780.\\n')",
    "ideal_response": "Reflection: COMPLETED; salary is already detected and appears correctly categorized.",
  },
  {
    "name": "salary_check_iteration_1b_reflection",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "None. This is the first attempt.",
    "execution_result": "(True, '--- Recent Income (Last 30 Days) ---\\nRecent Income Transactions:\\n- $2500 was received from ADP PAYROLL on 2025-11-20 (Chase Total Checking **1563) categorized as uncategorized.\\n- $2500 was received from Gusto on 2025-10-22 (Chase Total Checking **1563) categorized as uncategorized.\\nTotal recent income: earned $5000.\\n')",
    "ideal_response": "Reflection: IN_PROGRESS; salary-like uncategorized payroll deposits were found and should be recategorized.",
  },
  {
    "name": "salary_check_iteration_2b_success_reflection",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "Outcome 1: The initial step confirmed that no specific 'Salary' transactions were found, only uncategorized income. The execution result shows two recent income transactions of $2500 each, both categorized as 'uncategorized' and coming from known payroll providers (ADP PAYROLL and Gusto). This strongly suggests these are the salary payments that need to be re-categorized as 'Salary' as per the task description. The task is not fully accomplished because the re-categorization step has not yet been executed. The next logical step is to proceed with fixing the categorization of these identified transactions.",
    "execution_result": "(True, 'Successfully processed categorization request: Categorize all transactions originating from 'ADP PAYROLL' and 'Gusto' as 'Salary'.')",
    "ideal_response": "Reflection: COMPLETED; salary categorization fix and rule creation are done.",
  },
  {
    "name": "salary_check_iteration_2b_fail_reflection",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "Outcome 1: The initial step confirmed that no specific 'Salary' transactions were found, only uncategorized income. The execution result shows two recent income transactions of $2500 each, both categorized as 'uncategorized' and coming from known payroll providers (ADP PAYROLL and Gusto). This strongly suggests these are the salary payments that need to be re-categorized as 'Salary' as per the task description. The task is not fully accomplished because the re-categorization step has not yet been executed. The next logical step is to proceed with fixing the categorization of these identified transactions.",
    "execution_result": "(True, 'An unexpected error was encountered.')",
    "ideal_response": "Reflection: FAILED; unexpected error encountered.",
  },
  {
    "name": "salary_check_iteration_3_reflection",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "Outcome 1: The initial step confirmed that no specific 'Salary' transactions were found, only uncategorized income. The execution result shows two recent income transactions of $2500 each, both categorized as 'uncategorized' and coming from known payroll providers (ADP PAYROLL and Gusto). This strongly suggests these are the salary payments that need to be re-categorized as 'Salary' as per the task description. The task is not fully accomplished because the re-categorization step has not yet been executed. The next logical step is to proceed with fixing the categorization of these identified transactions.\nOutcome 2: The previous step identified two likely salary transactions ($2500 each from ADP PAYROLL and Gusto) that were uncategorized. The current execution step, which was intended to fix the categorization, resulted in an unexpected error: 'An unexpected error was encountered.'. This means the core objective of fixing the categorization was not achieved. Therefore, the task is incomplete, and the next step must be to retry the categorization fix, perhaps after logging or investigating the error if more context were available. Since I must provide the next logical step based only on the provided result, the next step should be to attempt the categorization fix again or report failure if retries are exhausted. Given the structure, I will mark it as IN_PROGRESS and assume the next step should be a retry or a different approach to categorization.",
    "execution_result": "(True, 'Successfully processed categorization request: Update transactions from source 'ADP PAYROLL' to category 'Salary'. Also, update transactions from source 'Gusto' to category 'Salary'.')",
    "ideal_response": "Reflection: COMPLETED; retry succeeded and payroll transactions were categorized as Salary.",
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


def run_all_tests_batch(optimizer: SelfReflectionWoCodeOptimizer = None):
  if optimizer is None:
    optimizer = SelfReflectionWoCodeOptimizer()
  batch_results = []
  print(f"\n{'#'*80}\nBATCH RUN START\n{'#'*80}\n")
  for i, tc in enumerate(TEST_CASES):
    result = run_test(i, optimizer)
    batch_results.append((tc["name"], result))
  print(f"\n{'#'*80}\nBATCH RUN SUMMARY\n{'#'*80}")
  for name, result in batch_results:
    status = "unknown"
    try:
      parsed = json.loads(result)
      status = parsed.get("next_status", "unknown")
    except Exception:
      pass
    print(f"- {name}: next_status={status}")
  print("#" * 80 + "\n")
  return batch_results


def main(test: str = None, run_batch: bool = False, no_thinking: bool = False, thinking_budget: int = None, max_output_tokens: int = None, model: str = None):
  tb = (0 if no_thinking else (thinking_budget if thinking_budget is not None else 4096))
  kw = {"thinking_budget": tb}
  if max_output_tokens is not None:
    kw["max_output_tokens"] = max_output_tokens
  if model is not None:
    kw["model_name"] = model
  optimizer = SelfReflectionWoCodeOptimizer(**kw)

  if run_batch:
    run_all_tests_batch(optimizer)
    return

  if test is not None:
    if test.strip().lower() == "all":
      run_all_tests_batch(optimizer)
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
  parser.add_argument("--run-batch", action="store_true", help="Run all test cases as a batch with summary.")
  args = parser.parse_args()
  main(test=args.test, run_batch=args.run_batch, no_thinking=args.no_thinking, thinking_budget=args.thinking_budget, max_output_tokens=args.max_output_tokens, model=args.model)
