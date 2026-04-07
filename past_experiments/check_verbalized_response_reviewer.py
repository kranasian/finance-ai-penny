from google import genai
from google.genai import types
import os
import json
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Prompt axes: greeting rule = acceptable only when (1) no Human before then Human then
# ai_review_response, or (2) no Human before ai_review_response; unacceptable when
# multiple turns (e.g. AI-Human-AI-ai_review_response or Human-AI-Human-ai_review_response).
# Also: rationale matches actual issues; advice requires recommendation early; no IDs.

SYSTEM_PROMPT = """You are a checker verifying VerbalizedResponseReviewer outputs against rules.

## Input:
- **EVAL_INPUT**: JSON string containing conversation turns and `ai_review_response` (the AI response being evaluated)
- **PAST_REVIEW_OUTCOMES**: Array of past reviews, each with `output`, `good_copy`, `info_correct`, `eval_text`
- **REVIEW_NEEDED**: The VerbalizedResponseReviewer output to review (JSON string with `rating` and `rationale`)

## Output:
JSON: `{"good_copy": boolean, "info_correct": boolean, "eval_text": string}`
- `good_copy`: True if REVIEW_NEEDED is valid JSON with required `rating` and `rationale` keys, and rating is one of the valid keys
- `info_correct`: True if REVIEW_NEEDED follows all rules below (including content-quality axes)
- `eval_text`: Required if either boolean is False; be specific and concise

## Critical Priority: Learn from PAST_REVIEW_OUTCOMES
**MANDATORY**: If PAST_REVIEW_OUTCOMES flags issues that still exist in REVIEW_NEEDED, mark as incorrect.
- Extract all issues from past `eval_text` fields
- Check if REVIEW_NEEDED repeats the same mistakes
- If past reviews flag a missing element and it's still missing → mark `info_correct: False`

## Rules

### Output Format Requirements
1. **JSON Structure**: Must be valid JSON with exactly two keys: `rating` and `rationale`
2. **Rating Key**: Must be one of: `good_response`, `non_sense`, `repetitive_information`, `incoherent_flow`, `verbose`, `incomplete`
3. **Rationale**: Must be present and brief

### Content-Quality Axes for REVIEW_NEEDED (rationale and rating)
Apply these when deciding `info_correct`. REVIEW_NEEDED = the reviewer's JSON output (rating + rationale).
- **Verbose rationale**: Flag `info_correct: False` if the rationale (1) breaks the evaluation into unnecessary sub-points (e.g. "First... Second... Third...", or separated ideas even if short), or (2) is excessively long for the inquiry. The rationale must be brief and direct, not a long or stepwise breakdown, including no bullet points, numbered lists, or similar structural separations.
- **Length**: Flag `info_correct: False` if the rationale is too long to be digestible; it should support the rating concisely.
- **Direct + conversational**: The evaluation should reflect that a good ai_review_response both answers the Human's last turn directly and maintains conversational flow with smooth transitions. Flag only if the rating or rationale clearly contradicts this.
- **No internal system details**: Transaction IDs, request IDs, internal reference IDs, and any system-only identifiers must never appear in the rationale. If present → `info_correct: False`.
- **Consistency with previous AI messages**: If earlier AI turns in EVAL_INPUT contained errors (wrong numbers, misalignment), REVIEW_NEEDED should acknowledge or reflect that where relevant (e.g. rating `non_sense` or rationale noting inconsistency). When there is no prior error, do not require acknowledgement.
- **Greeting in ai_review_response — acceptable vs unacceptable flows**: Greetings (Hi, Hello, Hey, Good morning, etc.) in `ai_review_response` are **acceptable** only in these exact patterns:
  - **(1) [No Human messages at all before this]** Human: ... → ai_review_response: ... (the only turn before ai_review_response is a single Human message, and there is no AI turn before that Human — i.e. the conversation so far is strictly one Human then ai_review_response).
  - **(2) [No Human messages at all before this]** ai_review_response: ... (zero Human messages before ai_review_response; conversation_turns may be empty or only AI turns).
  Greetings are **unacceptable** when the flow is like:
  - **(1)** AI: ... Human: ... AI: ... ai_review_response: ... (there was an AI turn after the first Human before ai_review_response — greeting mid-conversation).
  - **(2)** Human: ... AI: ... Human: ... ai_review_response: ... (two or more Human messages before ai_review_response).
  So: if any AI turn appears **before** the Human turn that immediately precedes ai_review_response (e.g. AI then Human then ai_review_response), a greeting is **unacceptable** → rating `incoherent_flow`. **This rule takes precedence:** if an unacceptable greeting is present, the rating MUST be `incoherent_flow`, even if other factual errors might suggest a `non_sense` rating.
- **Rationale must match actual issues in ai_review_response**: Verify that the rationale in REVIEW_NEEDED describes issues that are actually present in the text of `ai_review_response`. If the rationale cites problems that do not exist in the response, or ignores issues that are present, mark `info_correct: False` and state the mismatch in eval_text.
- **Rationale terminology alignment**: The language used in the rationale must align precisely with the terminology and context established in the `EVAL_INPUT` conversation. For instance, if the conversation is about "grocery spending," the rationale should use "grocery spending" instead of a broader term like "savings." If there's a misalignment, mark `info_correct: False`.
- **Advice requests require a recommendation, stated early**: When the Human asks for advice (e.g. what to do, what they should do), `ai_review_response` should provide a clear recommendation. If the response also gives alternatives, the recommendation must be mentioned early so the message is straight to the point. If the response only lists options without a direct recommendation, or buries the recommendation after long alternatives, rate `incomplete` or `verbose` as appropriate; REVIEW_NEEDED that correctly reflects this is info_correct.

### Evaluation Rules (from VerbalizedResponseReviewer)
- Prioritize relevance to the last Human turn; use latest values from `ai_review_response`; evaluate as a whole with full context.
- Rating must accurately reflect quality of `ai_review_response`. Accept `good_response` when the response directly answers the Human's question and any extra detail is modest and relevant (e.g. one comparable prior period); do not require `verbose` for that. Reserve `verbose` for when the response is excessively long, breaks into unnecessary sub-points, or harms digestibility. When the Human asks for a recommendation (e.g. "what should I do") and the response only lists options without giving a direct recommendation, or gives alternatives without stating the recommendation early, `incomplete` or `verbose` is appropriate. Other ratings: `non_sense` = misaligned or inappropriate; `incoherent_flow` = opener in middle of conversation or dead end.

## Verification Steps
1. **PAST_REVIEW_OUTCOMES**: If REVIEW_NEEDED repeats flagged issues → `info_correct: False`.
2. **good_copy**: Valid JSON, both `rating` and `rationale`, `rating` in valid set.
3. **info_correct**: Apply all rules and content-quality axes (rationale brief, no stepwise breakdown, not too long; no transaction/internal IDs in rationale; greeting rule: acceptable only when (1) single Human then ai_review_response with no AI before that Human, or (2) zero Human messages before ai_review_response — if any AI turn appears before the Human that precedes ai_review_response, greeting is unacceptable; rationale describes actual issues; rationale terminology alignment; advice requires recommendation early; consistency with prior AI errors; rating matches ai_review_response quality).
4. **eval_text**: If either boolean is False, list specific issues in one or two short sentences; reference unfixed PAST_REVIEW_OUTCOMES when relevant.
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
    
    # Model Configuration
    self.thinking_budget = 1024
    self.model_name = model_name
    
    # Generation Configuration Constants
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
      review_needed: The VerbalizedResponseReviewer output that needs to be reviewed (JSON string with rating and rationale).
      
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
      temperature=self.temperature,
      top_p=self.top_p,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=self.system_prompt)],
      thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
    )

    # Generate response
    output_text = ""
    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=generate_content_config,
    ):
      if chunk.text is not None:
        output_text += chunk.text
    
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
    review_needed: The VerbalizedResponseReviewer output that needs to be reviewed (JSON string with rating and rationale).
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


def run_correct_response(checker: CheckVerbalizedResponseReviewer = None):
  """
  Run the test case for correct_response.
  """
  eval_input = json.dumps({
    "conversation_turns": [
      {
        "speaker": "Human",
        "message": "What's my total cash balance?"
      }
    ],
    "ai_review_response": "You have $27,593 in total cash across your checking and savings accounts."
  }, indent=2)
  
  review_needed = json.dumps({
    "rating": "good_response",
    "rationale": "The response directly answers the Human's last inquiry about total cash balance with a specific dollar amount, providing clear and relevant information."
  }, indent=2)
  
  return run_test_case("correct_response", eval_input, review_needed, [], checker)


def run_greeting_response(checker: CheckVerbalizedResponseReviewer = None):
  """
  Run the test case for a response that starts with a greeting in an ongoing conversation.
  """
  eval_input = json.dumps({
    "conversation_turns": [
      {
        "speaker": "Human",
        "message": "I want to save $50,000 for a down payment."
      },
      {
        "speaker": "AI",
        "message": "That's a great goal! Let me help you create a savings plan. Based on your current income and expenses, you can save about $1,200 per month."
      },
      {
        "speaker": "Human",
        "message": "How long will it take to reach my goal?"
      },
      {
        "speaker": "AI",
        "message": "With your current savings rate, it will take approximately 42 months, or about 3.5 years, to reach $50,000."
      },
      {
        "speaker": "Human",
        "message": "What's my current savings balance?"
      }
    ],
    "ai_review_response": "Hello! Your current savings balance is $8,500 across your savings accounts."
  }, indent=2)
  
  review_needed = json.dumps({
    "rating": "incoherent_flow",
    "rationale": "The response contains a conversation opener ('Hello!') in the middle of an ongoing conversation, which breaks the conversation flow and is inappropriate for an established dialogue."
  }, indent=2)
  
  return run_test_case("greeting_response", eval_input, review_needed, [], checker)


def run_unnecessary_details(checker: CheckVerbalizedResponseReviewer = None):
  """
  Exchange: Human asks dining out last month; ai_review_response gives last month
  and previous month. REVIEW_NEEDED good_response (direct answer, acceptable).
  """
  eval_input = json.dumps({
    "conversation_turns": [
      {"speaker": "Human", "message": "What did I spend dining out last month?"}
    ],
    "ai_review_response": "You spent $500 last month, and $400 the previous month."
  }, indent=2)
  review_needed = json.dumps({
    "rating": "good_response",
    "rationale": "The response directly answers the Human's question about dining spend last month with a clear amount ($500)."
  }, indent=2)
  return run_test_case("verbose_breakdown_response", eval_input, review_needed, [], checker)


def run_transaction_id_in_rationale(checker: CheckVerbalizedResponseReviewer = None):
  """
  Run the test case for REVIEW_NEEDED that contains internal system details
  (e.g. transaction IDs) in the rationale — should be flagged info_correct False.
  Exchange: Human asks biggest expense; ai_review_response includes ID 2352.
  """
  eval_input = json.dumps({
    "conversation_turns": [
      {"speaker": "Human", "message": "What was my biggest expense last month?"}
    ],
    "ai_review_response": "Your biggest expense last month was on McDonald's (ID 2352) for $780."
  }, indent=2)
  review_needed = json.dumps({
    "rating": "good_response",
    "rationale": "The response answers the Human's question. ID 2352 confirms the expense at McDonald's."
  }, indent=2)
  return run_test_case("transaction_id_in_rationale", eval_input, review_needed, [], checker)


def run_first_message_greeting_ok(checker: CheckVerbalizedResponseReviewer = None):
  """
  Run the test case where ai_review_response is the first AI reply to the Human's
  first message; a greeting here is appropriate — should NOT be flagged incoherent_flow.
  """
  eval_input = json.dumps({
    "conversation_turns": [
      {"speaker": "Human", "message": "I'd like to know my spending this month."}
    ],
    "ai_review_response": "Hello! Your spending this month is $1,240 across all categories."
  }, indent=2)
  review_needed = json.dumps({
    "rating": "good_response",
    "rationale": "This is the first exchange; a greeting is appropriate. The response then directly answers the Human's inquiry with a clear dollar amount."
  }, indent=2)
  return run_test_case("first_message_greeting_ok", eval_input, review_needed, [], checker)


def run_does_not_directly_answer(checker: CheckVerbalizedResponseReviewer = None):
  """
  ai_review_response does not directly answer the Human's message/question.
  Human asks what they should do; response lists options without a direct recommendation.
  """
  eval_input = json.dumps({
    "conversation_turns": [
      {"speaker": "Human", "message": "What should I do with my Amazon stock?"}
    ],
    "ai_review_response": "You have three main options to consider with your Amazon stock. First, you could buy more shares if you believe the stock will appreciate and want to increase your position. Second, you could sell some or all of your holdings if you need liquidity, want to lock in gains, or have concerns about the company's outlook. Third, you could keep the amount you have right now—maintaining your current position without adding or reducing—which may be appropriate if you're uncertain or waiting for more information. Each choice depends on your goals, time horizon, and risk tolerance. I can help you explore any of these in more detail."
  }, indent=2)
  review_needed = json.dumps({
    "rating": "incomplete",
    "rationale": "The Human asked what they should do; the response only lists options (buy, sell, hold) without giving a direct recommendation or actionable answer."
  }, indent=2)
  return run_test_case("does_not_directly_answer", eval_input, review_needed, [], checker)


def run_misaligned_with_previous_ai(checker: CheckVerbalizedResponseReviewer = None):
  """
  Information in ai_review_response is misaligned with information previously
  shared by the AI in the conversation. Expected: rating non_sense.
  """
  eval_input = json.dumps({
    "conversation_turns": [
      {"speaker": "Human", "message": "How much was my grocery spending last month??"},
      {"speaker": "AI", "message": "Last month, you spent $180 on groceries."},
      {"speaker": "Human", "message": "How does that compare with the current month?"}
    ],
    "ai_review_response": "Since your grocery spending this month is at $100 and last month was $200, you are at half of last month's spending so far."
  }, indent=2)
  review_needed = json.dumps({
    "rating": "non_sense",
    "rationale": "The response states savings is $200 but the AI previously said savings is $180; the information is misaligned with the conversation."
  }, indent=2)
  return run_test_case("misaligned_with_previous_ai", eval_input, review_needed, [], checker)


def run_hello_after_exchanges(checker: CheckVerbalizedResponseReviewer = None):
  """
  ai_review_response starts with 'Hello', and there have already been exchanges
  between Human and AI before ai_review_response. Expected: incoherent_flow.
  """
  eval_input = json.dumps({
    "conversation_turns": [
      {"speaker": "Human", "message": "What did I spend on dining last week?"},
      {"speaker": "AI", "message": "Last week you spent $180 on dining."},
      {"speaker": "Human", "message": "Break it down by day."}
    ],
    "ai_review_response": "Hello! Monday $45, Tuesday $32, Wednesday $0, Thursday $58, Friday $25, Saturday $20."
  }, indent=2)
  review_needed = json.dumps({
    "rating": "incoherent_flow",
    "rationale": "The response opens with 'Hello' in the middle of an ongoing conversation after multiple exchanges, which breaks the flow."
  }, indent=2)
  return run_test_case("hello_after_exchanges", eval_input, review_needed, [], checker)


def run_hi_first_message_from_human(checker: CheckVerbalizedResponseReviewer = None):
  """
  ai_review_response starts with 'Hi', and the only message before this was from
  the Human (first exchange). Greeting is appropriate → good_response.
  """
  eval_input = json.dumps({
    "conversation_turns": [
      {"speaker": "Human", "message": "What's my account balance?"}
    ],
    "ai_review_response": "Hi! Your account balance is $4,750."
  }, indent=2)
  review_needed = json.dumps({
    "rating": "good_response",
    "rationale": "A greeting is appropriate as this is the first exchange. The response directly answers with the balance."
  }, indent=2)
  return run_test_case("hi_first_message_from_human", eval_input, review_needed, [], checker)


def run_hey_after_ai_message_only(checker: CheckVerbalizedResponseReviewer = None):
  """
  Flow: AI says shopping spending is high; Human asks for breakdown.
  ai_review_response starts with 'Hi!' mid-conversation → incoherent_flow.
  """
  eval_input = json.dumps({
    "conversation_turns": [
      {"speaker": "AI", "message": "Your shopping spending is high this month at $500."},
      {"speaker": "Human", "message": "Can I have a breakdown?"}
    ],
    "ai_review_response": "Hi! You spent on Nordstrom, H&M, Old Navy, and Zara."
  }, indent=2)
  review_needed = json.dumps({
    "rating": "incoherent_flow",
    "rationale": "The response opens with 'Hi!' in the middle of an ongoing conversation after the Human asked for a breakdown; conversation openers are not appropriate here."
  }, indent=2)
  return run_test_case("hey_after_ai_message_only", eval_input, review_needed, [], checker)


def run_contradicts_previous_ai_categories(checker: CheckVerbalizedResponseReviewer = None):
  """
  AI said pet spending is higher than expected; Human asks which categories are higher.
  ai_review_response says no categories are higher — contradicts previous AI → non_sense.
  """
  eval_input = json.dumps({
    "conversation_turns": [
      {"speaker": "AI", "message": "Your pet spending is higher than expected this week at $500!"},
      {"speaker": "Human", "message": "Thanks for letting me know. What are all of the categories that are higher than expected this week?"}
    ],
    "ai_review_response": "You're doing great this week! There are no categories at all this week that you are spending more than expected on."
  }, indent=2)
  review_needed = json.dumps({
    "rating": "incoherent_flow",
    "rationale": "The ai_review_response contains an unacceptable greeting (\"You're doing great this week!\") given the conversation flow, which mandates an 'incoherent_flow' rating. This takes precedence over other potential issues like factual contradictions."
  }, indent=2)
  return run_test_case("contradicts_previous_ai_categories", eval_input, review_needed, [], checker)


def run_good_morning_first_reply(checker: CheckVerbalizedResponseReviewer = None):
  """
  ai_review_response starts with 'Good morning' and is the AI's reply to the
  first Human message (acceptable flow). Expected: good_response.
  """
  eval_input = json.dumps({
    "conversation_turns": [
      {"speaker": "Human", "message": "Can you show my budget summary for this month?"}
    ],
    "ai_review_response": "Good morning! Your budget summary for this month: you've used 72% of your dining budget and 45% of your groceries budget. You're on track overall."
  }, indent=2)
  review_needed = json.dumps({
    "rating": "good_response",
    "rationale": "A greeting is appropriate as this is the first exchange. The response directly answers with a clear budget summary."
  }, indent=2)
  return run_test_case("good_morning_first_reply", eval_input, review_needed, [], checker)


def run_rationale_stepwise_breakdown(checker: CheckVerbalizedResponseReviewer = None):
  """
  REVIEW_NEEDED has a rationale that breaks down into unnecessary sub-points
  (First... Second... Third...). Checker should flag info_correct False.
  """
  eval_input = json.dumps({
    "conversation_turns": [
      {"speaker": "Human", "message": "What's my balance?"}
    ],
    "ai_review_response": "Your balance is $3,200."
  }, indent=2)
  review_needed = json.dumps({
    "rating": "good_response",
    "rationale": "First, the response answers the Human's question. Second, it provides a specific dollar amount. Third, it is concise. Therefore it is a good response."
  }, indent=2)
  return run_test_case("rationale_stepwise_breakdown", eval_input, review_needed, [], checker)


def run_rationale_too_long(checker: CheckVerbalizedResponseReviewer = None):
  """
  REVIEW_NEEDED has an excessively long rationale for a simple inquiry.
  Checker should flag info_correct False.
  """
  eval_input = json.dumps({
    "conversation_turns": [
      {"speaker": "Human", "message": "How much did I spend on coffee?"}
    ],
    "ai_review_response": "You spent $42 on coffee this month."
  }, indent=2)
  long_rationale = (
    "The response is appropriate because it directly addresses the user's question about coffee spending. "
    "It provides a specific numerical value which is what the user was seeking. The tone is professional and "
    "concise. Additionally, the response does not include unnecessary details or digressions. It matches "
    "the context of a financial assistant providing spending information. The brevity is also a positive "
    "factor in terms of user experience and clarity of communication."
  )
  review_needed = json.dumps({
    "rating": "good_response",
    "rationale": long_rationale
  }, indent=2)
  return run_test_case("rationale_too_long", eval_input, review_needed, [], checker)


def run_repetitive_information(checker: CheckVerbalizedResponseReviewer = None):
  """
  ai_review_response repeats the same point without value. REVIEW_NEEDED
  correctly rates repetitive_information with a brief rationale.
  """
  eval_input = json.dumps({
    "conversation_turns": [
      {"speaker": "Human", "message": "Is my spending on track?"},
      {"speaker": "AI", "message": "Checking your budget now."}
    ],
    "ai_review_response": "Your spending is on track. You're doing well—your spending is on track this month. Overall, spending is on track."
  }, indent=2)
  review_needed = json.dumps({
    "rating": "repetitive_information",
    "rationale": "The response repeats the same idea (spending is on track) multiple times without adding new information or context."
  }, indent=2)
  return run_test_case("repetitive_information", eval_input, review_needed, [], checker)


def run_unfulfilled_request_explained(checker: 'CheckVerbalizedResponseReviewer' = None):
  """
  AI cannot fulfill a request but clearly communicates the reason why.
  Expected: unfulfilled_request, info_correct: true.
  """
  eval_input = json.dumps({
    "conversation_turns": [
      {"speaker": "Human", "message": "Can you tell me how my Tesla stock performed last year?"}
    ],
    "ai_review_response": "I cannot provide real-time stock performance data as my capabilities are limited to analyzing your personal financial transactions. To get Tesla stock performance, please consult a financial news website or your brokerage account."
  }, indent=2)
  review_needed = json.dumps({
    "rating": "unfulfilled_request",
    "rationale": "The AI clearly states its inability to fulfill the request and provides a helpful reason and alternative, which is appropriate for an unfulfilled request."
  }, indent=2)
  return run_test_case("unfulfilled_request_explained", eval_input, review_needed, [], checker)


BATCHES = {
  1: [run_correct_response, run_unnecessary_details, run_good_morning_first_reply, run_unfulfilled_request_explained],
  2: [run_greeting_response, run_hello_after_exchanges, run_hey_after_ai_message_only],
  3: [run_transaction_id_in_rationale, run_rationale_stepwise_breakdown, run_rationale_too_long],
  4: [run_first_message_greeting_ok, run_hi_first_message_from_human, run_repetitive_information],
  5: [run_does_not_directly_answer, run_misaligned_with_previous_ai, run_contradicts_previous_ai_categories],
}


def main():
  """Main function to test the VerbalizedResponseReviewer checker. 5 batches of 3 tests each (15 total). Supports --batch 1|2|3|4|5 and --runs N."""
  parser = argparse.ArgumentParser(description="Run VerbalizedResponseReviewer checker tests by batch (5 batches of 3).")
  parser.add_argument("--batch", type=int, choices=[1, 2, 3, 4, 5], default=None, help="Run only this batch (1-5). If omitted, run all 5 batches once.")
  parser.add_argument("--runs", type=int, default=3, help="Number of runs per batch when --batch is set (default: 3).")
  args = parser.parse_args()

  checker = CheckVerbalizedResponseReviewer()
  if args.batch is not None:
    runs = [args.batch]
    per_batch = args.runs
  else:
    runs = [1, 2, 3, 4, 5]
    per_batch = 1

  for batch_id in runs:
    for run_num in range(per_batch):
      if per_batch > 1:
        print(f"\n>>> BATCH {batch_id} — RUN {run_num + 1}/{per_batch}")
      for fn in BATCHES[batch_id]:
        fn(checker)


if __name__ == "__main__":
  main()
