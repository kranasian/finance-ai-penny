from google import genai
from google.genai import types
import os
import json
import argparse
from typing import Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Also: rationale matches actual issues; advice requires recommendation early; no IDs.

SYSTEM_PROMPT = """You validate **REVIEW_NEEDED** (VerbalizedResponseReviewer JSON) against **EVAL_INPUT** (`conversation_turns`, `ai_review_response`) and **PAST_REVIEW_OUTCOMES** (`output`, `good_copy`, `info_correct`, `eval_text`).

**Your output** `{"good_copy": boolean, "info_correct": boolean, "eval_text": string}` — `eval_text` only when a boolean is false; short, specific.

**PAST_REVIEW_OUTCOMES**: If a past `eval_text` issue still appears in REVIEW_NEEDED → `info_correct: false`.

**Allowed verdict keys** (each maps to one non-empty rationale string). These six only:

- **`good_response`**: The response is appropriate for the conversation. This includes straightforwardly answering the Human's last inquiry, or clearly stating limitations (e.g., technical error, out of scope) in a well-worded and helpful manner, even if the user's request cannot be fulfilled. If no other red flags apply, then this *must* be `good_response`. This must be the only flag used if no other red flags apply.

- **`non_sense`**: Irrelevant or illogical response, or response with inconsistent computations, or if the `ai_review_response` contradicts information previously provided by the AI in the conversation. If `ai_review_response` contradicts information previously provided by the AI in the conversation, then this *must* be `non_sense`, and `good_response` CANNOT be applied.

- **`repetitive_information`**: Repeats information from within its own response without value/context change.

- **`incoherent_flow`**: Leads to a dead end or breaks the conversation flow.

- **`verbose`**: Does not straightforwardly answer the Human last inquiry because of too many details or information unnecessary to the question.

- **`incomplete`**: Too brief or lacks necessary information.

**Applying the definitions**: Weight the last Human turn. Modest relevant extra that still straightforwardly answers (e.g. one prior period) can stay `good_response`. Use the full thread **only** for **non_sense** when this reply **repeats, depends on, or overrides** earlier AI facts and conflicts with them—**do not** mark `info_correct: false` using unrelated recalculations or topics the `ai_review_response` never invokes. **Treat figures and facts stated in `ai_review_response` as given** for this check unless **REVIEW_NEEDED** asserts a mismatch or the reply **directly contradicts** an earlier AI line on the **same** fact (do not invent “hallucination” from missing earlier disclosure alone).

**Axes (checker)** — six keys unchanged:
- **Multi-key `REVIEW_NEEDED`**: Allowed when multiple verdicts truly apply. **Exception**: if **`good_response`** is present, it must be the **only** key (never mix with another verdict).
- **Buying time**: Brief greeting or one short setup clause before the substantive answer is **acceptable**; anything longer is **not** modest setup.
- **`verbose` tag** also fits when: (1) **REVIEW_NEEDED** value strings use breakdown / excessive evaluation detail **not needed** to support the verdict vs the last Human question; (2) **`ai_review_response`** does not answer the last Human immediately after at most one short setup clause; (3) any **REVIEW_NEEDED** rationale includes internal info (e.g. transaction/request IDs). For (1)-(3), `good_response` alone is wrong.
- **Hard precedence for direct numeric asks** (e.g., "how much", "total", "balance", "spend"): if `ai_review_response` includes a long identity/self-introduction, multi-sentence capability pitch, or pre-answer breakdown before giving the requested number, treat REVIEW_NEEDED `good_response` as incorrect; `verbose` should be present.
- **Scope-purity rule for targeted asks**: when the Human asks for one specific period/metric (e.g., "last month", "total cash"), adding extra periods/projections (this month/next month) or listing components before the requested total counts as unnecessary detail. In those cases, REVIEW_NEEDED with only `good_response` is incorrect; `verbose` should be present.
- **Correction exception (avoid false non_sense)**: if `ai_review_response` explicitly acknowledges an earlier error and clearly corrects it (e.g., apology + corrected value + brief reason), that can still be coherent; do not force `non_sense` in this correction pattern.
- **Cannot fulfill the ask**: If limits are communicated clearly and helpfully and no other red flag applies, **`good_response`** is appropriate—**not** a defect or “missing” answer.

**good_copy** (schema only): JSON parses; every top-level key is one of the six above; every value is a non-empty string (no nested objects/arrays). Multi-key OK when multiple red flags apply; **`good_response` solo** when it is the only applicable flag. Rationale flaws → **`info_correct` only**; `good_copy` stays true if shape is valid. `good_copy` false for malformed JSON, disallowed keys, empty/non-string values, or `good_response` paired with any other key.

**info_correct**: Verdict set and rationales match definitions and axes. Prior-AI contradiction relevant to this reply → **`non_sense` required**; solo **`good_response`** there is wrong.
If REVIEW_NEEDED uses a valid key that correctly captures the primary failure (e.g., `incoherent_flow` for an off-topic dead-end), do not mark false only because another key (like `verbose`) could also apply.

**Rationale hygiene** (each value string): Tight; no unnecessary “First/Second/Third”, bullets, or **multi-sentence essay** for a trivial one-line `ai_review_response` → **false**. No internal IDs in any rationale → **false** if present.

**Terminology**: Rationales must match conversation labels (e.g. groceries vs savings).

**Checklist**: Past issues → `good_copy` → `info_correct` → `eval_text` if needed.
"""

class CheckVerbalizedResponseReviewer:
  """Handles all Gemini API interactions for checking VerbalizedResponseReviewer outputs against rules"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for checking VerbalizedResponseReviewer evaluations"""
    # API Configuration
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)
    
    # Runtime Configuration
    self.json = False
    self.sanitize = False

    # Model Configuration
    self.thinking_budget = 500
    self.model_name = model_name
    
    # Generation Configuration Constants
    self.top_k = 40
    self.temperature = 0.6
    self.top_p = 0.95
    self.max_output_tokens = 6000
    
    # Safety Settings
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
    
    # System Prompt
    self.system_prompt = SYSTEM_PROMPT

  
  def generate_response(self, eval_input: str, past_review_outcomes: list, review_needed: str) -> dict:
    """
    Generate a response using Gemini API for checking VerbalizedResponseReviewer outputs.
    
    Args:
      eval_input: JSON string containing conversation turns and ai_review_response (the AI response being evaluated).
      past_review_outcomes: An array of past review outcomes, each containing `output`, `good_copy`, `info_correct`, and `eval_text`.
      review_needed: The VerbalizedResponseReviewer output that needs to be reviewed (JSON string: verdict key(s) → rationale string(s)).
      
    Returns:
      Dictionary with good_copy, info_correct, and eval_text keys
    """
    # Create request text with the new input structure
    request_text_str = f"""<EVAL_INPUT>

{eval_input}

</EVAL_INPUT>

<PAST_REVIEW_OUTCOMES>

{json.dumps(past_review_outcomes, indent=2)}

</PAST_REVIEW_OUTCOMES>

<REVIEW_NEEDED>

{review_needed}

</REVIEW_NEEDED>

Output:"""
    
    print(request_text_str)
    print(f"\n{'='*80}\n")
    
    request_text = types.Part.from_text(text=request_text_str)
    
    # Create content and configuration
    contents = [types.Content(role="user", parts=[request_text])]
    
    generate_content_config = types.GenerateContentConfig(
      top_k=self.top_k,
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
    if thought_summary.strip():
      print(f"{'=' * 80}")
      print("THOUGHT SUMMARY:")
      print(thought_summary.strip())
      print("=" * 80)
    
    # Check if response is empty
    if not output_text or not output_text.strip():
      raise ValueError(f"Empty response from model. Check API key and model availability.")
    
    # Parse JSON response
    try:
      # Remove markdown code blocks if present
      if "```json" in output_text:
        json_start = output_text.find("```json") + 7
        json_end = output_text.find("```", json_start)
        if json_end != -1:
          output_text = output_text[json_start:json_end].strip()
      elif "```" in output_text:
        # Try to find JSON in code blocks
        json_start = output_text.find("```") + 3
        json_end = output_text.find("```", json_start)
        if json_end != -1:
          output_text = output_text[json_start:json_end].strip()
      
      # Extract JSON object from the response
      json_start = output_text.find('{')
      json_end = output_text.rfind('}') + 1
      
      if json_start != -1 and json_end > json_start:
        json_str = output_text[json_start:json_end]
        return json.loads(json_str)
      else:
        # Try parsing the whole response
        return json.loads(output_text.strip())
    except json.JSONDecodeError as e:
      raise ValueError(f"Failed to parse JSON response: {e}\nResponse length: {len(output_text)}\nResponse preview: {output_text[:500]}")


def run_test_case(test_name: str, eval_input: str, review_needed: str, past_review_outcomes: list = None, checker: 'CheckVerbalizedResponseReviewer' = None):
  """
  Utility function to run a test case with custom inputs, common error handling, and output formatting.
  
  Args:
    test_name: Name of the test case
    eval_input: JSON string containing conversation turns and ai_review_response (the AI response being evaluated).
    review_needed: The VerbalizedResponseReviewer output that needs to be reviewed (JSON string: verdict key(s) → rationale string(s)).
    past_review_outcomes: An array of past review outcomes, each containing `output`, `good_copy`, `info_correct`, and `eval_text`. Defaults to empty list.
    checker: Optional CheckVerbalizedResponseReviewer instance. If None, creates a new one.
    
  Returns:
    Dictionary with good_copy, info_correct, and eval_text keys, or None if error occurred
  """
  if past_review_outcomes is None:
    past_review_outcomes = []
  if checker is None:
    checker = CheckVerbalizedResponseReviewer()

  print(f"\n{'='*80}")
  print(f"Running test: {test_name}")
  print(f"{'='*80}")

  try:
    # Directly call the checker's response with the provided inputs.
    result = checker.generate_response(eval_input, past_review_outcomes, review_needed)
    print(f"Result:")
    print(json.dumps(result, indent=2))
    print(f"{'='*80}")
    return result
  except Exception as e:
    print(f"ERROR: {type(e).__name__}: {str(e)}")
    print(f"{'='*80}")
    return None


def _compare_checker_result(actual_result: Optional[dict], ideal_output: dict) -> Tuple[bool, str]:
  """Compare good_copy and info_correct strictly; eval_text is checked only when needed."""
  if actual_result is None:
    return False, "no model output"

  actual_good_copy = bool(actual_result.get("good_copy"))
  actual_info_correct = bool(actual_result.get("info_correct"))
  ideal_good_copy = bool(ideal_output.get("good_copy"))
  ideal_info_correct = bool(ideal_output.get("info_correct"))

  if actual_good_copy != ideal_good_copy or actual_info_correct != ideal_info_correct:
    return (
      False,
      f"good_copy model={actual_good_copy} ideal={ideal_good_copy}; "
      f"info_correct model={actual_info_correct} ideal={ideal_info_correct}",
    )

  if not ideal_good_copy or not ideal_info_correct:
    actual_eval_text = (actual_result.get("eval_text") or "").strip()
    if not actual_eval_text:
      return False, "expected non-empty eval_text when a boolean is false"
  return True, "booleans match ideal output"


TEST_CASES = [
  {
    "batch": 1,
    "name": "cash_total_direct_answer_good_response",
    "eval_input": {
      "conversation_turns": [
        {"speaker": "Human", "message": "What's my total cash balance right now?"}
      ],
      "ai_review_response": "Your total cash balance is $27,593."
    },
    "review_needed": {
      "good_response": "The response directly answers the Human's question with the requested total cash amount."
    },
    "ideal_output": {"good_copy": True, "info_correct": True, "eval_text": ""}
  },
  {
    "batch": 1,
    "name": "cannot_fulfill_but_helpful_good_response",
    "eval_input": {
      "conversation_turns": [
        {"speaker": "Human", "message": "Can you check Tesla's stock performance today?"}
      ],
      "ai_review_response": "I can't access live stock market prices. For today's Tesla performance, please check your brokerage app or a market data site."
    },
    "review_needed": {
      "good_response": "The response clearly communicates a limitation and provides helpful next steps, which is appropriate."
    },
    "ideal_output": {"good_copy": True, "info_correct": True, "eval_text": ""}
  },
  {
    "batch": 1,
    "name": "self_intro_long_should_be_verbose_not_good_response",
    "eval_input": {
      "conversation_turns": [
        {"speaker": "Human", "message": "How much did I spend on groceries last month?"}
      ],
      "ai_review_response": "Hi, I'm Penny, your personal AI financial copilot built to help you understand spending trends, optimize cash flow, and build healthier money habits with confidence. I analyze transaction behavior, budgeting patterns, recurring bills, and category-level changes so you can make informed decisions over time. Last month, your grocery spending was $420."
    },
    "review_needed": {
      "good_response": "The response gives the grocery total and is helpful."
    },
    "ideal_output": {"good_copy": True, "info_correct": False, "eval_text": "Should be verbose due to unnecessary self-introduction before answering."}
  },
  {
    "batch": 2,
    "name": "food_last_month_with_extra_months_should_be_verbose",
    "eval_input": {
      "conversation_turns": [
        {"speaker": "Human", "message": "What was my food spending last month?"}
      ],
      "ai_review_response": "Your food spending was $510 last month, $460 this month so far, and next month is projected to be around $540."
    },
    "review_needed": {
      "good_response": "The response provides the spending figures and answers the request."
    },
    "ideal_output": {"good_copy": True, "info_correct": False, "eval_text": "Should be verbose because it adds this month and projected next month unnecessarily."}
  },
  {
    "batch": 2,
    "name": "total_cash_breakdown_before_total_should_be_verbose",
    "eval_input": {
      "conversation_turns": [
        {"speaker": "Human", "message": "What's my total cash?"}
      ],
      "ai_review_response": "You have $6,200 in checking, $8,900 in high-yield savings, and $2,450 in emergency savings. Your total cash is $17,550."
    },
    "review_needed": {
      "good_response": "The response gives complete information and includes the total cash amount."
    },
    "ideal_output": {"good_copy": True, "info_correct": False, "eval_text": "Should be verbose because the total should be given immediately before account-by-account breakdown."}
  },
  {
    "batch": 2,
    "name": "brief_buy_time_then_answer_is_good_response",
    "eval_input": {
      "conversation_turns": [
        {"speaker": "Human", "message": "How much did I spend on transport this week?"}
      ],
      "ai_review_response": "Sure, give me a second to check."
    },
    "review_needed": {
      "incomplete": "Does not answer the user's question."
    },
    "ideal_output": {"good_copy": True, "info_correct": False, "eval_text": "Should be good_response since it properly responds to the user's message."}
  },
  {
    "batch": 3,
    "name": "review_needed_with_transaction_id_should_fail",
    "eval_input": {
      "conversation_turns": [
        {"speaker": "Human", "message": "What was my biggest dining charge?"}
      ],
      "ai_review_response": "Your biggest dining charge was $87 at Bistro Cafe (Transaction ID 2134)."
    },
    "review_needed": {
      "good_response": "The answer is clear and correct. Transaction ID 99231 validates the charge."
    },
    "ideal_output": {"good_copy": True, "info_correct": False, "eval_text": "Rationale should not include internal identifiers."}
  },
  {
    "batch": 3,
    "name": "contradiction_with_prior_ai_requires_non_sense",
    "eval_input": {
      "conversation_turns": [
        {"speaker": "Human", "message": "How much did I spend on groceries last month?"},
        {"speaker": "AI", "message": "You spent $180 on groceries last month."},
        {"speaker": "Human", "message": "How does this month compare?"}
      ],
      "ai_review_response": "This month you are at $100, and last month was $230."
    },
    "review_needed": {
      "good_response": "The response compares this month and last month clearly."
    },
    "ideal_output": {"good_copy": True, "info_correct": False, "eval_text": "Should be non_sense because it contradicts prior AI grocery amount."}
  },
  {
    "batch": 3,
    "name": "contradiction_with_prior_ai_requires_non_sense",
    "eval_input": {
      "conversation_turns": [
        {"speaker": "Human", "message": "How much did I spend on groceries last month?"},
        {"speaker": "AI", "message": "You spent $180 on groceries last month."},
        {"speaker": "Human", "message": "How does this month compare?"}
      ],
      "ai_review_response": "This month you are at $100, and, actually, looks like last month was $230. Apologies for the error earlier! I missed one transaction."
    },
    "review_needed": {
      "good_response": "The response compares this month and last month clearly."
    },
    "ideal_output": {"good_copy": True, "info_correct": True, "eval_text": ""}
  },
  {
    "batch": 4,
    "name": "good_response_plus_another_key_is_bad_copy",
    "eval_input": {
      "conversation_turns": [
        {"speaker": "Human", "message": "What's my checking balance?"}
      ],
      "ai_review_response": "Your checking balance is looking good."
    },
    "review_needed": {
      "good_response": "It answers the question.",
    },
    "ideal_output": {"good_copy": True, "info_correct": False, "eval_text": "Should be incomplete since the user's question was not answered."}
  },
  {
    "batch": 4,
    "name": "repetitive_information_correct_label",
    "eval_input": {
      "conversation_turns": [
        {"speaker": "Human", "message": "Is my spending on track?"}
      ],
      "ai_review_response": "Your spending is on track. You are on track this month. Overall, your spending is on track."
    },
    "review_needed": {
      "repetitive_information": "The response repeats the same idea without adding meaning or context."
    },
    "ideal_output": {"good_copy": True, "info_correct": True, "eval_text": ""}
  },
  {
    "batch": 4,
    "name": "incoherent_flow_correct_label",
    "eval_input": {
      "conversation_turns": [
        {"speaker": "Human", "message": "How much did I spend on gas this month?"}
      ],
      "ai_review_response": "You spent $160 on gas this month. Also, remember that snowflakes can have six sides, and anyway let's end here."
    },
    "review_needed": {
      "incoherent_flow": "The reply starts on-topic but then breaks flow with irrelevant content and a dead-end transition."
    },
    "ideal_output": {"good_copy": True, "info_correct": True, "eval_text": ""}
  },
]


def run_test(tc: dict, checker: Optional[CheckVerbalizedResponseReviewer] = None):
  if checker is None:
    checker = CheckVerbalizedResponseReviewer()

  result = run_test_case(
    tc["name"],
    json.dumps(tc["eval_input"], indent=2),
    json.dumps(tc["review_needed"], indent=2),
    [],
    checker,
  )
  print("Ideal output:")
  print(json.dumps(tc["ideal_output"], indent=2))
  ok, detail = _compare_checker_result(result, tc["ideal_output"])
  print("Match:", "PASS" if ok else "FAIL")
  print(detail)
  print(f"{'='*80}")
  return result


BATCHES = {
  1: [tc for tc in TEST_CASES if tc["batch"] == 1],
  2: [tc for tc in TEST_CASES if tc["batch"] == 2],
  3: [tc for tc in TEST_CASES if tc["batch"] == 3],
  4: [tc for tc in TEST_CASES if tc["batch"] == 4],
}


def main():
  """Main function to test the VerbalizedResponseReviewer checker. 4 batches (mixed correct/wrong fixtures). Supports --batch 1|2|3|4 and --runs N."""
  parser = argparse.ArgumentParser(description="Run VerbalizedResponseReviewer checker tests by batch (4 batches).")
  parser.add_argument("--batch", type=int, choices=[1, 2, 3, 4], default=None, help="Run only this batch (1-4). If omitted, run all 4 batches once.")
  parser.add_argument("--runs", type=int, default=3, help="Number of runs per batch when --batch is set (default: 3).")
  args = parser.parse_args()

  checker = CheckVerbalizedResponseReviewer()
  if args.batch is not None:
    runs = [args.batch]
    per_batch = args.runs
  else:
    runs = [1, 2, 3, 4]
    per_batch = 1

  for batch_id in runs:
    for run_num in range(per_batch):
      if per_batch > 1:
        print(f"\n>>> BATCH {batch_id} — RUN {run_num + 1}/{per_batch}")
      for tc in BATCHES[batch_id]:
        run_test(tc, checker)


if __name__ == "__main__":
  main()
