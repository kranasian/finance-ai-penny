"""
Goal Update Text Verbalizer Optimizer.

Takes the output of goal_update_text_verbalizer.generate_update_text (the canonical status
string, e.g. "Past 3 days: spent $25. On track for the budget of $100.") as input and produces
a short, user-facing verbalized status. Input/output are JSON.
"""

from google import genai
from google.genai import types
from google.genai.errors import ClientError
import os
import sys
import json
from dotenv import load_dotenv

# Add the parent directory to the path so we can import from experiments
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

load_dotenv()

# Output: list of { id, verbalized_status }
SCHEMA = types.Schema(
  type=types.Type.ARRAY,
  items=types.Schema(
    type=types.Type.OBJECT,
    properties={
      "id": types.Schema(type=types.Type.NUMBER, description="Same id as input."),
      "verbalized_status": types.Schema(
        type=types.Type.STRING,
        description="Short, friendly status (e.g. 'Good job! You're on a roll in the past 3 days saving $45.'). One sentence or two with a lead-in."
      )
    },
    required=["id", "verbalized_status"]
  )
)

SYSTEM_PROMPT = """You turn canonical goal status into one short, user-facing sentence per goal.

**Input**: JSON array of goals. Each: id (number), status (canonical status, e.g. "Past 3 days: spent $25. On track for the budget of $100.").
**Output**: Reply with only the JSON array of {id, verbalized_status}. One entry per input; preserve ids. No other text, no markdown, no explanation—only the raw JSON array.

**Style**: Be brief—one short sentence, or two only if you start with a lead-in like "Good job!". Friendly when on track; clear and direct when off track. Preserve all amounts and timeframes (e.g. $25, $100, past 3 days, past 7 days).
**Samples**:
- Savings, on track: "Good job! You're on a roll in the past 3 days saving $45."
- Budget, on track: "Good job! You're on track—$25 spent in the past 3 days."
- Budget, off track: "You're off track—$10 inflow in the past 3 days (goal $50)."
- Savings, off track: "You're off track for your saving goal—overspent $5 in the past 3 days."

**Rules**:
1. One short sentence (or two with a lead-in). No filler. Always include exact amounts from the status ($X) and goal/budget amount when present.
2. Rephrase in natural language; do not echo the status. Use "in the past 3 days" / "in the past 7 days" (or "past 3/7 days") when stating activity.
3. Vary phrasing across goals—do not repeat the same sentence template for every item. Vary lead-ins, clause order, or wording (e.g. "You're on track—$0 spent...", "In the past 7 days you spent $0; on track for $200.", "Good job! On track with $0 spent in the past 7 days (goal $100).") so each verbalized_status reads distinctly.
4. Preserve meaning: "spent" = spending in period; "saved" = saved toward goal; "inflow" = credits/refunds; "overspent" = drew down savings. State "On track" or "Off track" clearly. Do not imply causation (avoid "because you spent", "having overspent").
5. Budget goals (status contains "budget of"): Use short, natural wording; do not use the word "budget" in the verbalized message.
6. Savings goals (status contains "saving goal of"): Use encouraging or neutral wording; mention saved amount and goal when relevant.
"""


class GoalUpdateTextVerbalizerOptimizer:
  """Turns canonical status (from generate_update_text) into short verbalized status."""

  def __init__(self, model_name="gemini-flash-lite-latest", thinking_budget=0):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Set it in .env or environment.")
    self.client = genai.Client(api_key=api_key)
    self.model_name = model_name
    self.thinking_budget = thinking_budget
    self.temperature = 0.8
    self.top_p = 0.95
    self.max_output_tokens = 2048
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ]
    self.system_prompt = SYSTEM_PROMPT
    # self.output_schema = SCHEMA

  def generate_response(self, goals: list) -> list:
    """
    Turn canonical status into short verbalized status per goal.

    Args:
      goals: List of dicts with keys: id (int), status (str). status = canonical status from generate_update_text.

    Returns:
      List of {"id": <int>, "verbalized_status": <str>}.
    """
    request_text = types.Part.from_text(text=f"""input:
{json.dumps([{"id": g["id"], "status": g["status"]} for g in goals], indent=2)}

output:""")

    contents = [types.Content(role="user", parts=[request_text])]
    config_kwargs = dict(
      temperature=self.temperature,
      top_p=self.top_p,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=self.system_prompt)],
    )
    if self.thinking_budget > 0:
      config_kwargs["thinking_config"] = types.ThinkingConfig(
        thinking_budget=self.thinking_budget,
        include_thoughts=True,
      )
    generate_content_config = types.GenerateContentConfig(**config_kwargs)

    output_text = ""
    thought_summary = ""
    try:
      if self.thinking_budget == 0:
        response = self.client.models.generate_content(
          model=self.model_name,
          contents=contents,
          config=generate_content_config,
        )
        if response and response.text:
          output_text = response.text
      else:
        for chunk in self.client.models.generate_content_stream(
          model=self.model_name,
          contents=contents,
          config=generate_content_config,
        ):
          if chunk.text is not None:
            output_text += chunk.text
          if hasattr(chunk, "candidates") and chunk.candidates:
            for candidate in chunk.candidates:
              if hasattr(candidate, "content") and candidate.content and hasattr(candidate.content, "parts"):
                for part in candidate.content.parts:
                  part_text = getattr(part, "text", None)
                  if not part_text:
                    continue
                  if getattr(part, "thought", False):
                    thought_summary = (thought_summary + part_text) if thought_summary else part_text
                  else:
                    output_text += part_text
    except ClientError as e:
      if self.thinking_budget == 0 and "only works in thinking mode" in (str(e) or "").lower():
        print("\n[NOTE] This model may require thinking mode; try: --thinking", flush=True)
        sys.exit(1)
      raise

    if thought_summary:
      print("\n" + "-" * 80)
      print("THOUGHT SUMMARY:")
      print("-" * 80)
      print(thought_summary.strip())
      print("-" * 80 + "\n")

    text = (output_text or "").strip()
    if not text and thought_summary:
      text = thought_summary.strip()
    if not text:
      return []
    try:
      parsed = json.loads(text)
      return _ensure_list(parsed)
    except json.JSONDecodeError:
      parsed = extract_json_from_response(text)
      if parsed is not None:
        return _ensure_list(parsed)
      if not output_text and thought_summary:
        parsed = extract_json_from_response(thought_summary)
        return _ensure_list(parsed) if parsed is not None else []
      return []


def _ensure_list(parsed):
  """If parsed is a list return it; if dict with 'verbalized' or 'verbalized_status' return that list; else return []."""
  if isinstance(parsed, list):
    return parsed
  if isinstance(parsed, dict) and "verbalized" in parsed:
    return parsed["verbalized"]
  if isinstance(parsed, dict) and "verbalized_status" in parsed:
    return parsed["verbalized_status"] if isinstance(parsed["verbalized_status"], list) else []
  return []


def extract_json_from_response(text: str):
  """Extract JSON array or object from model output (handles markdown code blocks)."""
  s = (text or "").strip()
  start_obj = s.find("{")
  start_arr = s.find("[")
  if start_arr != -1 and (start_obj == -1 or start_arr < start_obj):
    start, open_c, close_c = start_arr, "[", "]"
  elif start_obj != -1:
    start, open_c, close_c = start_obj, "{", "}"
  else:
    return None
  depth = 0
  for i in range(start, len(s)):
    if s[i] == open_c:
      depth += 1
    elif s[i] == close_c:
      depth -= 1
      if depth == 0:
        try:
          return json.loads(s[start : i + 1])
        except json.JSONDecodeError:
          pass
  return None


def run_verbalizer(goals: list, optimizer: GoalUpdateTextVerbalizerOptimizer = None) -> list:
  """
  Run the optimizer on goals whose status = canonical status from generate_update_text.

  Args:
    goals: List of {"id": int, "status": str}. status = canonical status.
    optimizer: Optional GoalUpdateTextVerbalizerOptimizer; if None, creates one.

  Returns:
    List of {id, verbalized_status}.
  """
  if optimizer is None:
    optimizer = GoalUpdateTextVerbalizerOptimizer()
  print("INPUT:")
  print(json.dumps(goals, indent=2))
  print()
  result = optimizer.generate_response(goals)
  print("OUTPUT:")
  print(json.dumps(result, indent=2))
  return result


# Test cases: status = output of goal_update_text_verbalizer.generate_update_text (finance-ai-llm-server goals/try_goal_update_text_verbalizer)
TEST_GOALS = [
  # --- budget ---
  {"id": 0, "status": "Past 3 days: spent $25. On track for the budget of $100.", "ideal_response": "Short; on track; spent $25 in 3 days; budget $100."},
  {"id": 1, "status": "Past 3 days: inflow $10. Off track for the budget of $50.", "ideal_response": "Short; off track; inflow $10; budget $50."},
  {"id": 2, "status": "Past 7 days: spent $45. On track for the budget of $200.", "ideal_response": "Short; on track; spent $45 in 7 days; budget $200."},
  {"id": 3, "status": "Past 7 days: inflow $20. Off track for the budget of $150.", "ideal_response": "Short; off track; inflow $20; budget $150."},
  {"id": 4, "status": "Past 7 days: spent $30. On track for the budget of $100.", "ideal_response": "Short; on track; spent $30; budget $100."},
  # --- save_X_amount ---
  {"id": 5, "status": "Past 3 days: saved $40. On track for the saving goal of $100.", "ideal_response": "Short; on track; saved $40 in 3 days; goal $100."},
  {"id": 6, "status": "Past 3 days: overspent $5. Off track for the saving goal of $200.", "ideal_response": "Short; off track; overspent $5; goal $200."},
  {"id": 7, "status": "Past 7 days: saved $120. On track for the saving goal of $300.", "ideal_response": "Short; on track; saved $120 in 7 days; goal $300."},
  {"id": 8, "status": "Past 7 days: overspent $15. Off track for the saving goal of $500.", "ideal_response": "Short; off track; overspent $15; goal $500."},
  # --- save_0 ---
  {"id": 9, "status": "Past 3 days: saved $30. On track for the saving goal of $80.", "ideal_response": "Short; on track; saved $30; goal $80."},
  {"id": 10, "status": "Past 7 days: saved $25. Off track for the saving goal of $100.", "ideal_response": "Short; off track; saved $25; goal $100."},
  # --- edge ---
  {"id": 11, "status": "Past 7 days: spent $10. On track for the budget of $1000.", "ideal_response": "Short; on track; spent $10 in 7 days; budget $1000."},
  {"id": 12, "status": "Past 7 days: spent $0. Off track for the budget of $0.", "ideal_response": "Short; off track; spent $0; budget $0."},
]


def get_test_goals(test_index: int):
  """Return the single goal at test_index as a one-item list for run_verbalizer (id, status)."""
  if 0 <= test_index < len(TEST_GOALS):
    g = TEST_GOALS[test_index]
    return [{"id": g["id"], "status": g["status"]}]
  return None


def main(test: str = None, thinking: bool = False):
  """Run the verbalizer. --test <index> runs one test; --test all or no --test runs all. Default: no thinking (so JSON output is returned). Use --thinking to enable thinking_budget=4096."""
  optimizer = GoalUpdateTextVerbalizerOptimizer(thinking_budget=4096 if thinking else 0)

  if test is not None:
    if test.strip().lower() == "all":
      goals = [{"id": g["id"], "status": g["status"]} for g in TEST_GOALS]
      run_verbalizer(goals, optimizer)
      for i, g in enumerate(TEST_GOALS):
        if g.get("ideal_response"):
          print(f"\n[{i}] ideal_response: {g['ideal_response']}")
      return
    idx = int(test) if test.isdigit() else None
    if idx is not None and 0 <= idx < len(TEST_GOALS):
      goals = get_test_goals(idx)
      run_verbalizer(goals, optimizer)
      if TEST_GOALS[idx].get("ideal_response"):
        print("\n" + "=" * 80 + "\nIDEAL RESPONSE:\n" + "=" * 80 + "\n" + TEST_GOALS[idx]["ideal_response"] + "\n" + "=" * 80)
      return
    print(f"Test '{test}' not found. Use 0-{len(TEST_GOALS) - 1} or 'all'.")
    for i, g in enumerate(TEST_GOALS):
      print(f"  {i}: id={g['id']}, status={g['status'][:50]}...")
    return

  # No --test: show usage and list tests
  print("Usage:")
  print("  Run one test: --test <test_index>")
  print("  Run all tests: --test all")
  print("  Enable thinking: --thinking")
  print("\nAvailable tests:")
  for i, g in enumerate(TEST_GOALS):
    print(f"  {i}: id={g['id']}, status={g['status'][:50]}...")
  print("  all: run all tests")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Goal Update Text Verbalizer Optimizer: turn canonical status into short verbalized status.")
  parser.add_argument("--test", type=str, help="Test index (0 to %d) or 'all'. Default: run all." % (len(TEST_GOALS) - 1))
  parser.add_argument("--thinking", action="store_true", help="Enable thinking mode (thinking_budget=4096); default is no thinking so JSON output is returned")
  args = parser.parse_args()
  main(test=args.test, thinking=args.thinking)
