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
    "name": "salary_check_iteration_2_reflection",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "Outcome 1: Tried to lookup the salary. Found total income is 5000, but no specific 'Salary' transactions found. Only uncategorized income.",
    "execution_result": "(True, 'Found 6 recurring transactions of $2500 marked as ADP PAYROLL in the uncategorized list.')",
    "ideal_response": "Reflection: IN_PROGRESS; next step is to update ADP PAYROLL to Salary and create rule.",
  },
  {
    "name": "salary_check_iteration_3_reflection",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "Outcome 1: Tried to lookup the salary. Found no specific 'Salary' transactions found. Only uncategorized income.\nOutcome 2: Looked up detailed list of uncategorized income. Found recurring transactions of $2500 marked as 'ADP PAYROLL'.",
    "execution_result": "(True, 'Successfully updated 6 ADP PAYROLL transactions to Salary and created a rule for future ones.')",
    "ideal_response": "Reflection: COMPLETED; final_summary describes salary categorization fix.",
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
