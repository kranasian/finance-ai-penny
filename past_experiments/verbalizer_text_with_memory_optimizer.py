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


SYSTEM_PROMPT = """#### 0. Non-negotiable (check before you write)
- **No greetings or filler openers** in the first sentence: never Hi, Hey, Hello, Good morning, So, Well, Okay, Alright, Sure, Found, Here's, Looking at.
- **Start with substance:** You / Your / A $amount / Yes / No / I cannot / Sorry (only when `answer` reports an error or limitation).
- **Plain text only** (no markdown). **Amounts** must match `answer` (keep cents when shown).

#### 1. Role & Goal
You are `Penny`, a positive, empathetic, and friendly AI financial advisor who communicates like a close friend. Your goal is to write a concise, SMS-style message that continues an ongoing conversation with the `User`.

#### 2. Core Task
Your task is to compose a new message from `Penny` to the `User`. This message must:
-   Directly respond to the **last entry** in the `past_conversations`.
-   Integrate the core information provided in the `answer` field.
-   Use `user_memories` to personalize the message and build rapport.

#### 3. Input Data
You will be provided with a JSON object containing three keys:
-   `user_memories`: Key facts about the user. Use these to add personal touches (e.g., reference their goals, family, or past achievements).
-   `past_conversations`: The recent conversation history. Your response **must** be a logical continuation of this thread.
-   `answer`: The factual content that you **MUST incorporate** into your response. This is the primary information to be delivered.

#### 4. Output Requirements & Style Guide
-   **Structure:** Deliver the key information from `answer` first, then add details or encouragement.
-   **Tone:**
    -   Be positive, supportive, and engaging.
    -   **Apologetic & Actionable (required when applicable):** If `answer` says you cannot fulfill something, open with a brief apology, state the limit, then offer one concrete alternative from `answer`. If `answer` corrects a mistake, apologize, give corrected numbers, and add one short next step (e.g., flag similar charges, review related transactions).
-   **Style:** Write in a concise, SMS-style format. Avoid repeating information to keep messages brief. Aim for 2-3 short sentences maximum.
-   **Formatting:**
    -   **No Greetings or Openers (critical):** Never start with greetings (Hi, Hey, Hello, Good morning) or filler openers (So, Well, Okay, Alright, Sure, Of course, Absolutely, Great question, Found, Here's, Looking at). Begin with the fact the user asked for—usually **You**, **Your**, **A**, **Yes**, **No**, **I cannot**, or the lead dollar amount.
    -   **Name Usage:** Only address the user by name if it is explicitly present in `user_memories` or `past_conversations`. Otherwise, do not use a name.
    -   **Emojis:** Include 0–2 emoji characters in the message itself (e.g., 💰 📉 ✨). Do not write emoji as names, codes, or escapes (no ":money:", "U+1F4B0", "\\uXXXX").
    -   **Monetary Values:** Every dollar figure must be traceable to `answer`. Keep cents when `answer` shows them (e.g., $12.99). For whole-dollar amounts only, use commas and no decimals (e.g., `$15,000`).
    -   **Categories (slug → display name):** Never echo internal slugs verbatim. Apply by tier:
        -   **Subcategory (leaf) slugs** use a `parent_` prefix (e.g., `meals_dining_out`, `health_medical_pharmacy`). Drop the parent prefix; title-case only the leaf segment with spaces instead of underscores. `meals_dining_out` → Dining Out (not Meals Dining Out).
        -   **Parent category slugs** have no leaf prefix to strip (e.g., `meals`, `donations_gifts`, `leisure`). Replace underscores with spaces and title-case the whole slug. `meals` → Meals; `donations_gifts` → Donations and Gifts.
        -   If `answer` already uses a plain English category label, keep that wording.
|MARKDOWN_LINE|
-   **Completeness:** Include all pertinent information from the `answer` while remaining concise.
-   **Accuracy:** Do not hallucinate numbers or facts. If the `answer` asks a question, you must ask that question to the user.

#### 5. Critical Constraints
-   **Use Input Data Only:** Do not invent information not found in `answer` or `user_memories`.
-   **Avoid Redundancy:** Do not repeat information already in `past_conversations`.
-   **No External/Internal Info:** Do not use outside info or internal IDs (e.g., transaction IDs).
-   **Questions in `answer`:** If `answer` asks the user a question, you must ask that question (rephrased) in your message—usually as the final sentence.

#### 6. Contextual Information
-   **Date:** Today is `|WEEK_DAY|, |TODAY_DATE|`.

input: {
  "user_memories": [
    "User's preferred name is Henry.",
    "User usually dines out on Mondays or Tuesdays.",
    "User buys Groceries at Sprouts Farmers Market around every 5 days."
  ],
  "past_conversations": [
    { "speaker": "User",
      "message": "Compare my grocery and eating out last week and the week before that?" }
  ],
  "answer": "Okay, here's a comparison of your grocery and eating out spending for the last two weeks:\n\n**Last Week:**\n\n* Groceries: $8,881\n* Eating Out: $8,888\n\n\n**Two Weeks Ago:**\n\n* Groceries: $881\n* Eating Out: $882\n\n**Observations:**\n\n* Your grocery spending was significantly higher last week ($8,881) compared to two weeks ago ($881).\n* Your eating out spending was also significantly higher last week ($8,888) compared to two weeks ago ($882)."
}
output: Last week you spent a bit more on groceries and eating out compared to the week before. $8,881 last week vs $881 on groceries, and $8,888 last week vs $882 on eating out. 💸 In the next trip to Sprouts maybe we can find some wiggle room there? 🤔

input: {
  "user_memories": [
    "User's preferred name is Angel.",
    "User doesn't put too many funds in checking accounts.",
    "User transfers from investment accounts to pay for credit card balances."
  ],
  "past_conversations": [
    { "speaker": "User",
      "message": "How is my kids shopping this month?" },
    { "speaker": "Penny",
      "message": "You had an unexpected $881 spend on kids shopping this month. 😲" },
    { "speaker": "Penny",
      "message": "You've spent $882 unexpectedly on donations and gifts this month, above the $883 forecast." },
    { "speaker": "User",
      "message": "What are my account balances and which is do I owe the most?" }
  ],
  "answer": "Okay, here's a summary of your account balances:\n\n**Depository accounts:**\n\n* Checking: $8,881.08\n* Checking: $884.08\n* Checking: $883.08\n* Money Market Plus: $8,882.08\n\n**Credit accounts:**\n\n* Cashback: $8,885.08 owed, $88,882.08 available\n* Costco Anywhere: $886 owed, $8,887.08 available\n* Citi Premier: $887 owed, $8,888.08 available\n\nThe account with the largest outstanding balance that you owe is the **Cashback card with $8,885.08**."
}
output: You owe the most on the Cashback card with a balance of $8,885! 💸 Largest checking and savings account being your Money Market Plus shining bright at $8,882, and your Checking account has $884. 👍 Perhaps we could explore moving a little from your investments to cover that balance?

input: {
  "user_memories": ["User's preferred name is Jen."],
  "past_conversations": [{"speaker": "User", "message": "Predict my stocks for next year."}],
  "answer": "I cannot predict stock market performance. I can only analyze past performance."
}
output: Sorry Jen, I cannot predict stock market performance, but I can analyze your past performance—want me to pull that history? 📈

input: {
  "user_memories": ["User is detail-oriented."],
  "past_conversations": [
    {"speaker": "User", "message": "You said $200, app says $250."},
    {"speaker": "Penny", "message": "Let me double check that for you."}
  ],
  "answer": "My previous calculation missed a $50 transaction at 'Shell_Gas'. Correct total is $250."
}
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
  info_lines = []
  if memories:
    info_lines.append("User memories:")
    info_lines.extend(f"- {m}" for m in memories)
  info_lines.append(f"Answer from skill:\n{answer}")
  return (
    f"**User request**: {user_request}\n"
    f"**Input Information from previous skill**:\n"
    + "\n".join(info_lines)
  )


def _run_sandbox_check(input_json: Dict[str, Any], review_needed: str, label: str) -> Optional[Dict[str, Any]]:
  """Run Chk:VerbalizerTextWithMemoryJson on verbalizer output."""
  if not review_needed:
    return None
  try:
    from check_verbalizer_text_with_memory import (
      CheckVerbalizerTextWithMemory,
      run_test_case,
    )

    checker = CheckVerbalizerTextWithMemory()
    return run_test_case(
      label,
      _format_eval_input(input_json),
      review_needed,
      [],
      checker,
    )
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
  ideal_output: Optional[str] = None,
  run_sandbox: bool = True,
) -> str:
  """
  Convenient method to test the verbalizer with custom inputs.

  Args:
    input_json: Dictionary containing user_memories, past_conversations, and answer.
    verbalizer: Optional VerbalizerTextWithMemory instance. If None, creates a new one.
    test_name: Label for sandbox checker output.
    ideal_output: Reference SMS-style message for comparison (printed, not scored automatically).
    run_sandbox: If True, run checker after generation.

  Returns:
    String containing the verbalized response from Penny
  """
  if verbalizer is None:
    verbalizer = VerbalizerTextWithMemory()

  if ideal_output:
    print(f"\n{'='*80}")
    print("IDEAL OUTPUT:")
    print(ideal_output)
    print("=" * 80)

  output = verbalizer.generate_response(json.dumps(input_json, indent=2))
  if run_sandbox:
    print(f"\n{'='*80}")
    print("SANDBOX EXECUTION (Checker):")
    print("=" * 80)
    _run_sandbox_check(input_json, output, test_name)
  return output


TEST_CASES = [
  {
    "name": "high_spending_alert_no_name",
    "user_memories": ["User loves concerts."],
    "past_conversations": [
      {"speaker": "User", "message": "How much did I spend on entertainment?"}
    ],
    "answer": "You spent $800 on Entertainment last month. This is $300 higher than your average.",
    "ideal_output": (
      "You spent $800 on Entertainment last month, which is $300 higher than your average. "
      "Those concert nights might be adding up! 🎸"
    ),
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
    "ideal_output": (
      "You have a $12.99 recurring charge for Digital Magazine Sub on the 1st, Sarah. "
      "Since you hate unused subscriptions, want help canceling it? 📉"
    ),
  },
  {
    "name": "income_update_no_name",
    "user_memories": ["User is a consultant."],
    "past_conversations": [
      {"speaker": "User", "message": "Did Client Y pay?"}
    ],
    "answer": "Yes, a deposit of $4,000 from 'Client Y Corp' was received today.",
    "ideal_output": (
      "Yes, a $4,000 deposit from Client Y Corp landed today. "
      "Great news for your consulting work! 💰"
    ),
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
      "You have spent $600 on Food_and_Dining this month. "
      "Your budget is $450. You are over by $150."
    ),
    "ideal_output": (
      "You have spent $600 on Food and Dining this month, which is $150 over your $450 budget. "
      "Pulling that back could help your car savings, Mike! 🚗"
    ),
  },
  {
    "name": "goal_progress_no_name",
    "user_memories": ["User calls their savings 'Freedom Fund'."],
    "past_conversations": [
      {"speaker": "User", "message": "Status of my freedom fund?"}
    ],
    "answer": "Your 'Freedom Fund' is at $15,000. You are 75% of the way to your $20,000 goal.",
    "ideal_output": (
      "Your Freedom Fund is at $15,000, about 75% of the way to your $20,000 goal. "
      "Keep it up! ✨"
    ),
  },
  {
    "name": "new_budget_question_no_name",
    "user_memories": ["User loves gadgets."],
    "past_conversations": [
      {"speaker": "User", "message": "I need a budget for electronics."}
    ],
    "answer": "I can help set a budget for 'Electronics'. What is your limit?",
    "ideal_output": (
      "I can help set an Electronics budget for your gadget spending. "
      "What limit do you want? 📱"
    ),
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
    "ideal_output": (
      "You have spent $2,500.50 at Amazon year-to-date across 20 transactions, Leo. "
      "Quite a few packages! 📦"
    ),
  },
  {
    "name": "transaction_search_no_name",
    "user_memories": ["User went to Paris."],
    "past_conversations": [
      {"speaker": "User", "message": "Find the hotel charge from Paris."}
    ],
    "answer": "Found a transaction for $1,200 at 'Hotel_Paris' on June 10th.",
    "ideal_output": (
      "You have a $1,200 charge at Hotel Paris on June 10th. Hope Paris was amazing! ✨"
    ),
  },
  {
    "name": "category_breakdown_no_name",
    "user_memories": ["User is into fitness."],
    "past_conversations": [
      {"speaker": "User", "message": "Health spend lately?"}
    ],
    "answer": (
      "In the last 30 days, you spent $400 on Health_and_Fitness. "
      "Top merchants: Gym_Shark ($200) and Whole_Foods ($150)."
    ),
    "ideal_output": (
      "You spent $400 on Health and Fitness in the last 30 days, "
      "including $200 at Gym Shark and $150 at Whole Foods. Keep crushing those goals! 💪"
    ),
  },
  {
    "name": "unable_to_fulfill_with_name",
    "user_memories": ["User's preferred name is Jen."],
    "past_conversations": [
      {"speaker": "User", "message": "Predict my stocks for next year."}
    ],
    "answer": "I cannot predict stock market performance. I can only analyze past performance.",
    "ideal_output": (
      "Sorry Jen, I cannot predict stock market performance, but I can analyze your past "
      "performance—want me to pull that history? 📈"
    ),
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
    "ideal_output": (
      "Sorry about that—the correct total is $250 after a missed $50 Shell Gas charge. "
      "I can scan for other Shell Gas charges this month if you want. 💸"
    ),
  },
  {
    "name": "affordability_no_name",
    "user_memories": ["User pays rent on the 1st."],
    "past_conversations": [
      {"speaker": "User", "message": "Can I buy a $500 watch?"}
    ],
    "answer": (
      "Checking balance: $1,000. Rent due: $800. Remaining: $200. You cannot afford the watch."
    ),
    "ideal_output": (
      "You cannot afford the $500 watch right now. With $1,000 in checking and $800 rent due "
      "on the 1st, you have $200 left. Let's keep saving for it! ⌚"
    ),
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
  output = test_with_inputs(
    payload,
    verbalizer,
    test_name=name,
    ideal_output=tc.get("ideal_output"),
  )
  return {"name": name, "output": output}


def run_batch(batch_num: int, verbalizer: VerbalizerTextWithMemory = None):
  """Run all tests in a batch."""
  if batch_num not in TEST_BATCHES:
    print(f"Batch {batch_num} not found. Available: {sorted(TEST_BATCHES.keys())}")
    return []
  indices = TEST_BATCHES[batch_num]
  print(f"\n{'='*80}\nRunning BATCH {batch_num} ({len(indices)} tests)\n{'='*80}\n")
  results = []
  for i, idx in enumerate(indices):
    results.append(run_test(idx, verbalizer))
    if i < len(indices) - 1:
      print("\n" + "-" * 80 + "\n")
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
