from __future__ import annotations

from google import genai
from google.genai import types
import os
import json
import re
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CONFIG = {
  "json": False,
  "sanitize": True,
  "model_name": "gemini-flash-lite-latest",
  "check_template": "Chk:VerbalizerTextWithMemoryJson",
  "gen_config": {
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 2048,
    "thinking_budget": 2304,
  },
  "replacements": {
    "WEEK_DAY": "Tuesday",
    "TODAY_DATE": "September 30, 2025",
    "MARKDOWN_LINE": "    -   **No Markdown:** Output must be plain text. Do NOT use markdown formatting (bold, italics, lists, etc.).",
  },
  "post_process": True,
}


def _apply_replacements(prompt: str, replacements: Optional[Dict[str, str]] = None) -> str:
  repl = dict(CONFIG.get("replacements") or {})
  if replacements:
    repl.update(replacements)
  for key, value in repl.items():
    prompt = prompt.replace(f"|{key}|", value)
  return prompt


def _sanitize_text(text: str) -> str:
  text = text.strip()
  if text.startswith("```json"):
    text = text[7:-3].strip()
  elif text.startswith("```"):
    text = text[3:-3].strip()
  return text


def _post_process_text(text: str) -> str:
  text = _sanitize_text(text)
  text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
  text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", text)
  text = re.sub(r"_([^_]+)_", r"\1", text)
  text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
  text = re.sub(r"^[-*]\s+", "", text, flags=re.MULTILINE)
  return text.strip()


SYSTEM_PROMPT = """You are Penny—warm, concise SMS (2–3 sentences) continuing the thread. Use only `answer` facts (+ light `user_memories` ties). Today: |WEEK_DAY|, |TODAY_DATE|.

**Open:** Start with You / Your / Yes / No / I cannot / Sorry / $amount. Never Hi, Hey, So, Here's, Found, Looking at.
|MARKDOWN_LINE|
**Money:** Match `answer` exactly (keep cents). **Emojis:** 0–2 real glyphs only—never \\u escapes.

**Slugs** (`parent_subcategory`): say the **leaf** only—drop parent prefix, underscores→spaces, title-case (meals_dining_out→Dining Out; shelter_home→Home; health_gym_wellness→Gym and Wellness). Never echo snake_case or colon labels. Plain English in `answer`→keep. Merchants: Shell_Gas→Shell Gas.

**Accounts:** Include masked *** tails from `answer`; omit internal account/transaction IDs even if `answer` lists them.

**Thread (`past_conversations`):** Always verbalize `answer`. Correction in `answer`→Sorry/My mistake + corrected numbers. **Repeat:** if Penny already answered the same User question, open with "As I mentioned—" or "To confirm again—" before restating `answer`. **Inability:** if `answer` says you cannot do something, open with **Sorry** (+ name from memories), state the limit, offer one alternative from `answer` (match the Jen example). No extra coaching unless `answer` includes it. If `answer` asks a question, end with it.

input: {"user_memories":["User's preferred name is Jen."],"past_conversations":[{"speaker":"User","message":"Predict my stocks for next year."}],"answer":"I cannot predict stock market performance. I can only analyze past performance."}
output: Sorry Jen, I cannot predict stock market performance, but I can analyze your past performance—want me to pull that history? 📈

input: {"user_memories":["User is detail-oriented."],"past_conversations":[{"speaker":"User","message":"You said $200, app says $250."},{"speaker":"Penny","message":"Let me double check that for you."}],"answer":"My previous calculation missed a $50 transaction at 'Shell_Gas'. Correct total is $250."}
output: Sorry about that—the correct total is $250 after a missed $50 Shell Gas charge. I can scan for other Shell Gas charges this month if you want. 💸
"""

class VerbalizerTextWithMemory:
  """Handles all Gemini API interactions for VerbalizerText with memory generation."""

  def __init__(self, model_name=None, replacements: Optional[Dict[str, str]] = None):
    """Initialize the Gemini agent with API configuration from CONFIG."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
      raise ValueError(
        "GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment."
      )
    self.client = genai.Client(api_key=api_key)

    gc = CONFIG["gen_config"]
    self.model_name = model_name if model_name is not None else CONFIG["model_name"]
    self.thinking_budget = gc["thinking_budget"]
    self.response_json = CONFIG["json"]
    self.sanitize = CONFIG["sanitize"]
    self.post_process = CONFIG["post_process"]
    self.check_template = CONFIG["check_template"]
    self.replacements = CONFIG["replacements"]

    self.temperature = gc["temperature"]
    self.top_p = gc["top_p"]
    self.top_k = gc["top_k"]
    self.max_output_tokens = gc["max_output_tokens"]

    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ]

    self.system_prompt = _apply_replacements(SYSTEM_PROMPT, replacements)

  
  def generate_response(self, input_json: str) -> str:
    """
    Generate a response using Gemini API for VerbalizerText with memory.
    
    Args:
      input_json: JSON string containing user_memories, past_conversations, and answer.
      
    Returns:
      String containing the verbalized response from Penny
    """
    # Create request text with the new input structure
    request_text_str = f"""input: {input_json}
output: """
    
    print(f"\n{'='*80}")
    print("INPUT JSON:")
    print(input_json)
    print("="*80)
    
    request_text = types.Part.from_text(text=request_text_str)
    
    # Create content and configuration
    contents = [types.Content(role="user", parts=[request_text])]
    
    generate_content_config = types.GenerateContentConfig(
      temperature=self.temperature,
      top_p=self.top_p,
      top_k=self.top_k,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=self.system_prompt)],
      thinking_config=types.ThinkingConfig(
        thinking_budget=self.thinking_budget,
        include_thoughts=True
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
      
      # Extract thought summary from chunk
      if hasattr(chunk, 'candidates') and chunk.candidates:
        for candidate in chunk.candidates:
          if hasattr(candidate, 'content') and candidate.content:
            if hasattr(candidate.content, 'parts') and candidate.content.parts:
              for part in candidate.content.parts:
                if hasattr(part, 'thought') and part.thought:
                  if hasattr(part, 'text') and part.text:
                    if thought_summary:
                      thought_summary += part.text
                    else:
                      thought_summary = part.text
    
    # Check if response is empty
    if not output_text or not output_text.strip():
      raise ValueError(f"Empty response from model. Check API key and model availability.")
    
    if thought_summary:
      print(f"{'='*80}")
      print("THOUGHT SUMMARY:")
      print(thought_summary.strip())
      print("="*80)

    text = output_text.strip()
    if self.sanitize:
      text = _sanitize_text(text)
    if self.post_process:
      text = _post_process_text(text)

    print(f"{'='*80}")
    print("RESPONSE OUTPUT:")
    print(text)
    print("="*80)

    return text


def _format_eval_input(input_json: Dict[str, Any]) -> str:
  """Build checker EVAL_INPUT from verbalizer test payload."""
  user_request = ""
  for turn in reversed(input_json.get("past_conversations") or []):
    if turn.get("speaker") == "User":
      user_request = turn.get("message", "")
      break
  memories = input_json.get("user_memories") or []
  answer = input_json.get("answer", "")
  past_conversations = input_json.get("past_conversations") or []
  info_lines = []
  if past_conversations:
    info_lines.append("Past conversations:")
    for turn in past_conversations:
      speaker = turn.get("speaker", "Unknown")
      message = turn.get("message", "")
      info_lines.append(f"- {speaker}: {message}")
  if memories:
    info_lines.append("User memories:")
    info_lines.extend(f"- {m}" for m in memories)
  info_lines.append(f"Answer from skill:\n{answer}")
  return (
    f"**User request**: {user_request}\n"
    f"**Input Information from previous skill**:\n"
    + "\n".join(info_lines)
  )


def _run_sandbox_check(
  input_json: Dict[str, Any],
  review_needed: str,
  label: str,
  ideal_output: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
  """Run Chk:VerbalizerTextWithMemoryJson on verbalizer output."""
  if not review_needed:
    return None
  try:
    from check_verbalizer_text_with_memory import (
      CheckVerbalizerTextWithMemory,
      run_test_case,
      _compare_checker_result,
    )

    checker = CheckVerbalizerTextWithMemory()
    result = run_test_case(
      label,
      _format_eval_input(input_json),
      review_needed,
      [],
      checker,
      ideal_output=ideal_output,
    )
    if ideal_output is not None and result is not None:
      ok, _ = _compare_checker_result(result, ideal_output)
      return {"checker": result, "matches_ideal": ok}
    return {"checker": result, "matches_ideal": None}
  except Exception as e:
    print(f"Sandbox check failed: {e}")
    import traceback

    print(traceback.format_exc())
    return None


def _payload_from_test_case(tc: Dict[str, Any]) -> Dict[str, Any]:
  return {
    "user_memories": tc.get("user_memories", []),
    "past_conversations": tc.get("past_conversations", []),
    "answer": tc["answer"],
  }


def test_with_inputs(
  input_json: Dict[str, Any],
  verbalizer: VerbalizerTextWithMemory = None,
  *,
  test_name: str = "Test",
  ideal_review_needed: Optional[str] = None,
  ideal_output: Optional[Dict[str, Any]] = None,
  run_sandbox: bool = True,
) -> str:
  """
  Convenient method to test the verbalizer with custom inputs.

  Args:
    input_json: Dictionary containing user_memories, past_conversations, and answer.
    verbalizer: Optional VerbalizerTextWithMemory instance. If None, creates a new one.
    test_name: Label for sandbox checker output.
    ideal_review_needed: Reference SMS-style message (printed for human comparison).
    ideal_output: Expected checker JSON (`good_copy`, `info_correct`, `eval_text`).
    run_sandbox: If True, run checker after generation.

  Returns:
    Tuple of (verbalizer output string, optional sandbox result dict).
  """
  if verbalizer is None:
    verbalizer = VerbalizerTextWithMemory()

  if ideal_review_needed:
    print(f"\n{'='*80}")
    print("IDEAL REVIEW_NEEDED:")
    print(ideal_review_needed)
    print("=" * 80)

  output = verbalizer.generate_response(json.dumps(input_json, indent=2))
  sandbox_result = None
  if run_sandbox:
    print(f"\n{'='*80}")
    print("SANDBOX EXECUTION (Checker):")
    print("=" * 80)
    sandbox_result = _run_sandbox_check(
      input_json, output, test_name, ideal_output=ideal_output
    )
  return output, sandbox_result


# ideal_review_needed: target SMS copy for the verbalizer prompt.
# ideal_output: expected Chk:VerbalizerTextWithMemoryJson result (good_copy, info_correct, eval_text).
TEST_CASES = [
  {
    "name": "high_spending_alert_no_name",
    "user_memories": ["User loves concerts."],
    "past_conversations": [
      {"speaker": "User", "message": "How much did I spend on entertainment?"}
    ],
    "answer": "You spent $800 on Entertainment last month. This is $300 higher than your average.",
    "ideal_review_needed": (
      "You spent $800 on Entertainment last month, which is $300 higher than your average. "
      "Those concert nights might be adding up! 🎸"
    ),
    "ideal_output": {"good_copy": True, "info_correct": True, "eval_text": ""},
  },
  {
    "name": "subscription_check_with_name",
    "user_memories": [
      "User's preferred name is Sarah.",
      "User hates unused subscriptions.",
    ],
    "past_conversations": [
      {"speaker": "User", "message": "Check for recurring charges."}
    ],
    "answer": "Found a recurring charge of $12.99 for 'Digital_Magazine_Sub' on the 1st.",
    "ideal_review_needed": (
      "You have a $12.99 recurring charge for Digital Magazine Sub on the 1st, Sarah. "
      "Since you hate unused subscriptions, want help canceling it? 📉"
    ),
    "ideal_output": {"good_copy": True, "info_correct": True, "eval_text": ""},
  },
  {
    "name": "income_update_no_name",
    "user_memories": ["User is a consultant."],
    "past_conversations": [
      {"speaker": "User", "message": "Did Client Y pay?"}
    ],
    "answer": "Yes, a deposit of $4,000 from 'Client Y Corp' was received today.",
    "ideal_review_needed": (
      "Yes, a $4,000 deposit from Client Y Corp landed today. "
      "Great news for your consulting work! 💰"
    ),
    "ideal_output": {"good_copy": True, "info_correct": True, "eval_text": ""},
  },
  {
    "name": "budget_overflow_with_name",
    "user_memories": [
      "User's preferred name is Mike.",
      "User is saving for a car.",
    ],
    "past_conversations": [
      {"speaker": "User", "message": "How is my dining budget?"}
    ],
    "answer": (
      "meals_dining_out this month: $600 spent, budget $450, over by $150."
    ),
    "ideal_review_needed": (
      "You spent $600 on dining out this month, $150 over your $450 budget. "
      "Pulling that back could help your car savings, Mike! 🚗"
    ),
    "ideal_output": {"good_copy": True, "info_correct": True, "eval_text": ""},
  },
  {
    "name": "goal_progress_no_name",
    "user_memories": ["User calls their savings 'Freedom Fund'."],
    "past_conversations": [
      {"speaker": "User", "message": "Status of my freedom fund?"}
    ],
    "answer": "Your 'Freedom Fund' is at $15,000. You are 75% of the way to your $20,000 goal.",
    "ideal_review_needed": (
      "Your Freedom Fund is at $15,000, about 75% of the way to your $20,000 goal. "
      "Keep it up! ✨"
    ),
    "ideal_output": {"good_copy": True, "info_correct": True, "eval_text": ""},
  },
  {
    "name": "new_budget_question_no_name",
    "user_memories": ["User loves gadgets."],
    "past_conversations": [
      {"speaker": "User", "message": "I need a budget for electronics."}
    ],
    "answer": "I can help set a budget for 'Electronics'. What is your limit?",
    "ideal_review_needed": (
      "I can help set an Electronics budget for your gadget spending. "
      "What limit do you want? 📱"
    ),
    "ideal_output": {"good_copy": True, "info_correct": True, "eval_text": ""},
  },
  {
    "name": "merchant_spend_with_name",
    "user_memories": [
      "User's preferred name is Leo.",
      "User shops at Amazon.",
    ],
    "past_conversations": [
      {"speaker": "User", "message": "Amazon spend this year?"}
    ],
    "answer": "You have spent $2,500.50 at Amazon year-to-date across 20 transactions.",
    "ideal_review_needed": (
      "You have spent $2,500.50 at Amazon year-to-date across 20 transactions, Leo. "
      "Quite a few packages! 📦"
    ),
    "ideal_output": {"good_copy": True, "info_correct": True, "eval_text": ""},
  },
  {
    "name": "rent_masked_account_no_name",
    "user_memories": ["User went to Paris."],
    "past_conversations": [
      {"speaker": "User", "message": "Which account paid my rent?"}
    ],
    "answer": (
      "shelter_home rent $1,800 paid from Chase Savings ***1242 (internal Account ID 231)."
    ),
    "ideal_review_needed": (
      "Your $1,800 rent was paid from Chase Savings ***1242. 🏠"
    ),
    "ideal_output": {"good_copy": True, "info_correct": True, "eval_text": ""},
  },
  {
    "name": "category_breakdown_no_name",
    "user_memories": ["User is into fitness."],
    "past_conversations": [
      {"speaker": "User", "message": "Health spend lately?"}
    ],
    "answer": (
      "health_gym_wellness last 30 days: $400 total. "
      "Top merchants: Gym_Shark ($200) and Whole_Foods ($150)."
    ),
    "ideal_review_needed": (
      "You spent $400 on gym and wellness in the last 30 days, "
      "including $200 at Gym Shark and $150 at Whole Foods. Keep crushing those goals! 💪"
    ),
    "ideal_output": {"good_copy": True, "info_correct": True, "eval_text": ""},
  },
  {
    "name": "unable_to_fulfill_with_name",
    "user_memories": ["User's preferred name is Jen."],
    "past_conversations": [
      {"speaker": "User", "message": "Predict my stocks for next year."}
    ],
    "answer": "I cannot predict stock market performance. I can only analyze past performance.",
    "ideal_review_needed": (
      "Sorry Jen, I cannot predict stock market performance, but I can analyze your past "
      "performance—want me to pull that history? 📈"
    ),
    "ideal_output": {"good_copy": True, "info_correct": True, "eval_text": ""},
  },
  {
    "name": "data_discrepancy_no_name",
    "user_memories": ["User is detail-oriented."],
    "past_conversations": [
      {"speaker": "User", "message": "You said $200, app says $250."},
      {"speaker": "Penny", "message": "Let me double check that for you."},
    ],
    "answer": (
      "My previous calculation missed a $50 transaction at 'Shell_Gas'. Correct total is $250."
    ),
    "ideal_review_needed": (
      "Sorry about that—the correct total is $250 after a missed $50 Shell Gas charge. "
      "I can scan for other Shell Gas charges this month if you want. 💸"
    ),
    "ideal_output": {"good_copy": True, "info_correct": True, "eval_text": ""},
  },
  {
    "name": "groceries_repeat_ack_no_name",
    "user_memories": [],
    "past_conversations": [
      {"speaker": "User", "message": "How much on meals_groceries last month?"},
      {"speaker": "Penny", "message": "You spent $420 on groceries last month."},
      {"speaker": "User", "message": "How much on meals_groceries last month?"},
    ],
    "answer": "meals_groceries last month: $420.",
    "ideal_review_needed": (
      "As I mentioned—you spent $420 on groceries last month. 🛒"
    ),
    "ideal_output": {"good_copy": True, "info_correct": True, "eval_text": ""},
  },
]

# Batches aligned to prompt optimization axes (indices into TEST_CASES).
TEST_BATCHES = {
  1: [0, 1, 2],  # spending analysis and insights
  2: [3, 4, 5],  # budgeting and goals
  3: [6, 7, 8],  # merchant and transaction details
  4: [9, 10, 11],  # complex reasoning and apologies
}


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


def run_test(test_name_or_index_or_dict, verbalizer: VerbalizerTextWithMemory = None) -> Dict[str, Any]:
  """Run a single test. Returns dict with name, output, and optional sandbox result."""
  if isinstance(test_name_or_index_or_dict, dict):
    if "answer" not in test_name_or_index_or_dict:
      print("Invalid test dict: must contain 'answer'.")
      return {"name": "custom_test", "output": None}
    tc = test_name_or_index_or_dict
    name = tc.get("name", "custom_test")
  else:
    tc = get_test_case(test_name_or_index_or_dict)
    if tc is None:
      print(f"Test case '{test_name_or_index_or_dict}' not found.")
      return {"name": str(test_name_or_index_or_dict), "output": None}
    name = tc["name"]

  print(f"\n{'='*80}\nRunning test: {name}\n{'='*80}\n")
  payload = _payload_from_test_case(tc)
  output, sandbox_result = test_with_inputs(
    payload,
    verbalizer,
    test_name=name,
    ideal_review_needed=tc.get("ideal_review_needed"),
    ideal_output=tc.get("ideal_output"),
  )
  matches_ideal = (
    sandbox_result.get("matches_ideal") if sandbox_result else None
  )
  return {"name": name, "output": output, "matches_ideal": matches_ideal}


def run_batch(batch_num: int, verbalizer: VerbalizerTextWithMemory = None):
  """Run all tests in a batch."""
  if batch_num not in TEST_BATCHES:
    print(f"Batch {batch_num} not found. Available: {sorted(TEST_BATCHES.keys())}")
    return []
  indices = TEST_BATCHES[batch_num]
  print(f"\n{'='*80}\nRunning BATCH {batch_num} ({len(indices)} tests)\n{'='*80}\n")
  results = []
  passed = 0
  for i, idx in enumerate(indices):
    outcome = run_test(idx, verbalizer)
    results.append(outcome)
    if outcome.get("matches_ideal") is True:
      passed += 1
    if i < len(indices) - 1:
      print("\n" + "-" * 80 + "\n")
  print(f"\nBatch {batch_num}: {passed}/{len(indices)} checker matches ideal_output")
  return results


def main(test: Optional[str] = None, batch: Optional[int] = None, round_num: int = 1):
  """Run single test (--test), batch (--batch 1-4), or list usage."""
  verbalizer = VerbalizerTextWithMemory()

  if batch is not None:
    print(f"\n{'#'*80}\nOPTIMIZATION ROUND {round_num} — BATCH {batch}\n{'#'*80}")
    run_batch(batch, verbalizer)
    return 0

  if test is not None:
    if test.strip().lower() == "all":
      print(f"\n{'='*80}\nRunning ALL test cases\n{'='*80}\n")
      for i in range(len(TEST_CASES)):
        run_test(i, verbalizer)
        if i < len(TEST_CASES) - 1:
          print("\n" + "-" * 80 + "\n")
      return 0
    test_val = int(test) if test.isdigit() else test
    outcome = run_test(test_val, verbalizer)
    if outcome.get("output") is None and outcome.get("name"):
      print("\nAvailable test cases:")
      for i, tc in enumerate(TEST_CASES):
        print(f"  {i}: {tc['name']}")
      print("  all: run all test cases")
    return 0

  print("Usage:")
  print("  Run a single test: --test <name_or_index>")
  print("  Run all tests: --test all")
  print("  Run a batch: --batch 1-4")
  print("\nAvailable test cases:")
  for i, tc in enumerate(TEST_CASES):
    print(f"  {i}: {tc['name']}")
  print("  all: run all test cases")
  return 0


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Run verbalizer tests in batches")
  parser.add_argument(
    "--test",
    type=str,
    help='Test name or index (e.g. "0" or "high_spending_alert_no_name" or "all")',
  )
  parser.add_argument("--batch", type=int, choices=[1, 2, 3, 4], help="Run test batch 1-4")
  parser.add_argument(
    "--round",
    type=int,
    default=1,
    choices=[1, 2, 3],
    help="Optimization round label (logging only)",
  )
  args = parser.parse_args()
  raise SystemExit(main(test=args.test, batch=args.batch, round_num=args.round))
