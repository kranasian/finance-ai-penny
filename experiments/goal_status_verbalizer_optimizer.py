"""
Goal Status Verbalizer Optimizer.

Takes generic goal status strings plus type (budget/savings) and category (slug for
budget; blank for savings) and produces short user-facing verbalized status.
Input/output are JSON.
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
        description="Short, user-friendly status message. Do not repeat the category name."
      )
    },
    required=["id", "verbalized_status"]
  )
)

SYSTEM_PROMPT = """You turn generic goal status lines into one short, user-facing sentence per goal.

**Input**: JSON array of goals. Each: id (number), type ("budget" | "savings"), category (slug or ""), status (generic message).
**Output**: JSON array of {id, verbalized_status}. One entry per input; preserve ids.

**Rules**:
1. One short sentence. No filler, no greetings. Prefer compact phrasing (e.g. "Target met this period." "On track—$30 under usual.").
2. Rephrase; do not echo the status. Vary wording and structure across goals.
3. Preserve all numbers and amounts (e.g. $40, $200, 5 months). When the status gives an actionable split (e.g. "split $40 across 5 months"), keep amounts and make the message actionable.
4. Preserve meaning strictly: "stay under"/"under" = under budget (never "meet" or "hit" the limit); "over" = over; "on track" = on track; "ahead"/"reached target"/"working toward"/"tight period" = same idea in different words.
5. Budget: derive natural wording from category (e.g. meals_groceries→meal prep, transportation_car→gas) but never include the category name or slug in the message.
6. Savings: neutral wording only (on track, ahead, target, period).
"""


class GoalStatusVerbalizerOptimizer:
  """Turns generic goal status + category into verbalized, category-aware status text."""

  def __init__(self, model_name="gemini-flash-lite-latest", thinking_budget=4096):
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
    Verbalize generic status messages into category-aware status text.

    Args:
      goals: List of dicts with keys: id (int), type (str, "budget" or "savings"), category (str), status (str).
             status = generic goal status text.

    Returns:
      List of {"id": <int>, "verbalized_status": <str>}.
    """
    request_text = types.Part.from_text(text=f"""input:
{json.dumps(goals, indent=2)}

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
            if hasattr(candidate, "content") and candidate.content and hasattr(candidate.content, "parts"):
              for part in candidate.content.parts:
                if getattr(part, "thought", False) and getattr(part, "text", None):
                  thought_summary = (thought_summary + part.text) if thought_summary else part.text
    except ClientError as e:
      if self.thinking_budget == 0 and "only works in thinking mode" in (str(e) or ""):
        print("\n[NOTE] This model requires thinking mode; use default or set thinking_budget > 0.", flush=True)
        sys.exit(1)
      raise

    if thought_summary:
      print("\n" + "-" * 80)
      print("THOUGHT SUMMARY:")
      print("-" * 80)
      print(thought_summary.strip())
      print("-" * 80 + "\n")

    text = (output_text or "").strip()
    if not text:
      return []
    try:
      parsed = json.loads(text)
      return _ensure_list(parsed)
    except json.JSONDecodeError:
      parsed = extract_json_from_response(text)
      return _ensure_list(parsed) if parsed is not None else []


def _ensure_list(parsed):
  """If parsed is a list return it; if dict with 'verbalized' return that; else return []."""
  if isinstance(parsed, list):
    return parsed
  if isinstance(parsed, dict) and "verbalized" in parsed:
    return parsed["verbalized"]
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


def run_verbalizer(goals: list, optimizer: GoalStatusVerbalizerOptimizer = None) -> list:
  """
  Run the goal status verbalizer on a list of goals.

  Args:
    goals: List of {"id": int, "type": "budget"|"savings", "category": str, "status": str}.
    optimizer: Optional GoalStatusVerbalizerOptimizer; if None, creates one.

  Returns:
    List of {id, verbalized_status}.
  """
  if optimizer is None:
    optimizer = GoalStatusVerbalizerOptimizer()
  print("INPUT:")
  print(json.dumps(goals, indent=2))
  print()
  result = optimizer.generate_response(goals)
  print("OUTPUT:")
  print(json.dumps(result, indent=2))
  return result


TEST_GOALS = [
  # --- category (build_category_update_text) — type "budget" ---
  {
    "id": 0,
    "type": "budget",
    "category": "meals_groceries",
    "status": "Watch out as you're trending to exceed this by $40.",
    "ideal_response": "Short; category-aware (e.g. meal prep); do not repeat category; preserve $40.",
  },
  {
    "id": 1,
    "type": "budget",
    "category": "meals_dining_out",
    "status": "Great work spending $30 less than usual and you'll likely be under this period.",
    "ideal_response": "Short; do not repeat category; preserve $30; on track this period.",
  },
  {
    "id": 2,
    "type": "budget",
    "category": "transportation_car",
    "status": "You're past the goal and spent $50 more than expected this period.",
    "ideal_response": "Short; do not repeat category; preserve $50; over this period.",
  },
  {
    "id": 3,
    "type": "budget",
    "category": "shopping_clothes",
    "status": "You're on track to stay under the goal this period.",
    "ideal_response": "Short; do not repeat category; convey on track under goal.",
  },
  {
    "id": 4,
    "type": "budget",
    "category": "entertainment",
    "status": "You're within your budget for this period.",
    "ideal_response": "Short; do not repeat category; within budget.",
  },
  # --- save_X_amount (build_save_x_amount_update_text) — type "savings" ---
  {
    "id": 5,
    "type": "savings",
    "category": "",
    "status": "At this rate, you'll miss the goal by $200 this period. You can split $40 across the 5 months left to reach the goal.",
    "ideal_response": "Short; preserve $200, $40, 5 months; actionable.",
  },
  {
    "id": 6,
    "type": "savings",
    "category": "",
    "status": "At this rate, you'll miss the goal by $100 this period.",
    "ideal_response": "Short; preserve $100; miss goal this period.",
  },
  {
    "id": 7,
    "type": "savings",
    "category": "",
    "status": "You're ahead: added $50 more than last month; keep it up to reach the goal sooner.",
    "ideal_response": "Short; preserve $50; ahead, keep it up.",
  },
  {
    "id": 8,
    "type": "savings",
    "category": "",
    "status": "Must have been tight this period—you've put in around $60 less than expected.",
    "ideal_response": "Short; preserve $60; tight period, less than expected.",
  },
  {
    "id": 9,
    "type": "savings",
    "category": "",
    "status": "You've reached your savings target. Great job!",
    "ideal_response": "Short; reached target.",
  },
  {
    "id": 10,
    "type": "savings",
    "category": "",
    "status": "Keep saving; you're working toward your target.",
    "ideal_response": "Short; working toward target.",
  },
  # --- save_0 (build_save_0_update_text) — type "savings" ---
  {
    "id": 11,
    "type": "savings",
    "category": "",
    "status": "You're ahead by $25 and you'll likely hit the goal.",
    "ideal_response": "Short; preserve $25; ahead, hit goal.",
  },
  {
    "id": 12,
    "type": "savings",
    "category": "",
    "status": "You're on track to hit the goal this period.",
    "ideal_response": "Short; on track this period.",
  },
  {
    "id": 13,
    "type": "savings",
    "category": "",
    "status": "Must have been tight this period—you've put in around $30 less than needed.",
    "ideal_response": "Short; preserve $30; less than needed.",
  },
  {
    "id": 14,
    "type": "savings",
    "category": "",
    "status": "You've hit your savings target for this period.",
    "ideal_response": "Short; hit target this period.",
  },
  {
    "id": 15,
    "type": "savings",
    "category": "",
    "status": "Keep going; you're working toward this period's target.",
    "ideal_response": "Short; working toward period target.",
  },
]


def get_test_goals(test_index: int):
  """Return the single goal at test_index as a one-item list for run_verbalizer."""
  if 0 <= test_index < len(TEST_GOALS):
    g = TEST_GOALS[test_index].copy()
    g.pop("ideal_response", None)
    return [g]
  return None


def main(test: str = None, no_thinking: bool = False):
  """Run the verbalizer. --test <index> runs one test; --test all or no --test runs all. Use --no-thinking to set thinking_budget=0."""
  optimizer = GoalStatusVerbalizerOptimizer(thinking_budget=0 if no_thinking else 4096)

  if test is not None:
    if test.strip().lower() == "all":
      goals = [{"id": g["id"], "type": g["type"], "category": g["category"], "status": g["status"]} for g in TEST_GOALS]
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
      print(f"  {i}: id={g['id']}, type={g['type']}, category={g['category']}")
    return

  # No --test: show usage and list tests
  print("Usage:")
  print("  Run one test: --test <test_index>")
  print("  Run all tests: --test all")
  print("  Disable thinking: --no-thinking")
  print("\nAvailable tests:")
  for i, g in enumerate(TEST_GOALS):
    print(f"  {i}: id={g['id']}, type={g['type']}, category={g['category']}")
  print("  all: run all tests")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Goal Status Verbalizer: turn generic status + category into verbalized text.")
  parser.add_argument("--test", type=str, help="Test index (0 to %d) or 'all'. Default: run all." % (len(TEST_GOALS) - 1))
  parser.add_argument("--no-thinking", action="store_true", help="Set thinking_budget=0")
  args = parser.parse_args()
  main(test=args.test, no_thinking=args.no_thinking)
