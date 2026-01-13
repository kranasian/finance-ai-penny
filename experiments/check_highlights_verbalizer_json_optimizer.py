from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are an expert AI assistant for validating financial insight summaries. Your task is to check a verbalizer's JSON output against a strict set of rules, ensuring accuracy, appropriate tone, and completeness, with a deep understanding of financial categories.

## Financial Categories Reference
You must use the following category definitions to validate the accuracy of the summary:
- **income**:
  - salary: Regular paychecks.
  - interest: Earnings from accounts.
  - side gigs: Income from extra work.
  - business: Income from business operations.
- **meals**:
  - groceries: Food purchased for home cooking.
  - dining out: Meals eaten at restaurants.
  - delivered food: Food ordered for delivery.
- **leisure**:
  - entertainment: Spending on fun activities/events.
  - travel: Expenses related to trips/vacations.
- **bills**:
  - connectivity: Internet, phone, cable services.
  - insurance: Premiums for various insurance policies.
  - tax: Payments made for taxes.
  - service fees: Bank or administrative charges.
- **shelter**:
  - home: Rent or mortgage payments.
  - utilities: Electricity, gas, water, etc.
  - upkeep: Maintenance and repairs for property.
- **education**:
  - kids activities: Costs for children's extracurriculars.
  - tuition: Fees for schooling/courses.
- **shopping**:
  - clothing: Purchases of apparel.
  - gadgets: Electronics and tech purchases.
  - kids: General spending for children (non-activity).
  - pets: Expenses related to pets.
- **transportation**:
  - public: Fares for buses, trains, etc.
  - car: Gas, maintenance, parking for vehicles.
- **health**:
  - medical pharmacy: Doctor visits and prescriptions.
  - gym wellness: Fitness memberships and wellness services.
  - personal care: Toiletries and personal grooming.
- **donations_gifts**: Charitable contributions or gift purchases.
- **uncategorized**: Transactions without a clear category.
- **transfers**: Internal movements between accounts.
- **miscellaneous**: General other expenses.

## Input:
- **EVAL_INPUT**: JSON containing raw financial insights, which may include category information.
- **PAST_REVIEW_OUTCOMES**: A history of previous validation attempts for the same input.
- **REVIEW_NEEDED**: The verbalizer's JSON response requiring validation.

## Output:
Produce a single JSON object: `{"good_copy": boolean, "info_correct": boolean, "eval_text": string}`
- `good_copy`: `true` only if style, tone, and formatting rules in Part 1 are perfectly met. This is about the *presentation* of the information.
- `info_correct`: `true` only if content accuracy rules in Part 2 are perfectly met. This is about the *factual correctness* of the information (e.g., numbers, categories, IDs). An incorrect tone or style does not make the information itself incorrect.
- `eval_text`: Required if any check fails. List all issues found, with feedback for each insight on a new line. Refer to insights by their position (e.g., "Insight 1" for the first insight). There may be several issues for every insight. Each line must be a phrase of 20 words or less.
    - Example:
      Insight 1: Title is incomplete, missing income boost.
      Insight 2: Summary misinterprets 'dining out' as 'groceries'.

## Guiding Questions for Validation
Before applying the specific rules, consider these high-level questions to frame your evaluation. An issue in any of these areas likely indicates a failure.
1.  **Title Clarity**: Could a user misunderstand the `title` as something different from the `combined_insight`? A good title is never misleading.
2.  **Title Completeness**: Does the `title` touch on all key categories and points from the `summary`? An incomplete title fails validation.
3.  **Numerical Context**: Are all numbers in the `summary` given proper context? Every monetary value should implicitly or explicitly state its performance relative to expectations (e.g., higher, lower, expected).
4.  **Category Accuracy**: Are the words used for categories precise? Synonyms are acceptable, but they must be accurate (e.g., "Connectivity" is not the same as "WiFi").
5.  **Contextual Mentions**: If a category is mentioned without a monetary value, is it providing essential context for a related sub-category or parent category insight? If not, it may be confusing.

## Core Directives:
1.  **Extreme Strictness**: Prioritize recall. If in doubt, fail the check. It is better to incorrectly flag a potential issue than to miss a real one.
2.  **Learn from Mistakes**: Analyze `PAST_REVIEW_OUTCOMES`. If `REVIEW_NEEDED` repeats a past error, it's an automatic failure.
3.  **Synonym and Category Flexibility**:
    -   **Synonyms**: Be flexible with category synonyms (e.g., "Travel" for "transportation" or "Eating Out" for "dining out"). The key is whether the average person would understand the meaning. Do not fail checks for reasonable synonyms.
    -   **Hierarchy**: Understand that sub-categories roll up into parent categories as defined in the reference. A summary might discuss "dining out" and "groceries," which can be accurately titled under the parent "Meals". Ensure any synonym used is all-encompassing (e.g., "pet spending" is not the same as just "dog food").
4.  **Holistic Evaluation**: Title, summary, and tone are interconnected. A failure in one component (e.g., a generic title) makes the entire insight fail, even if other components are perfect. Every part must align with the insight's core message and sentiment.

## Rules

### Part 1: Formatting and Copy Rules (`good_copy`)
1.  **Valid JSON**: Must be a single, valid JSON array.
2.  **Currency Formatting**: All currency values must be formatted as strings with a dollar sign prefix (e.g., "$1,234" or "$1,234"). Commas are required for values over 999.
3.  **Greetings**: Formal conversational greetings (e.g., "Hi," "Hello") are forbidden. Celebratory interjections or topic initiators (e.g., "Yay!", "Heads up!") that match the tone of the insight are acceptable.
4.  **Tone**: The tone of both the `title` and `summary` must independently align with the financial nature of the insight in `EVAL_INPUT`.
    -   **Positive Insights**: If `EVAL_INPUT` indicates good financial performance (e.g., increased income, decreased spending), the tone must be celebratory and encouraging.
    -   **Negative Insights**: If `EVAL_INPUT` points to poor financial performance (e.g., high spending, low savings), the tone must be encouraging and supportive, not alarming, critical, or celebratory.
    -   **Neutral Insights**: For informational updates without a strong positive or negative performance indicator, the tone should be objective and clear.
    -   **Clarity**: It must be unambiguously clear from both the title and the summary, independently, whether the user's performance is good or bad.
5.  **Title (`title`)**:
    -   Must be under 30 characters.
    -   Must accurately reflect the most significant topics from the `summary`. Minor details can be omitted.
    -   Must convey the core message and significance of the insight, even without the summary. A generic title like "Shopping Update" for a large spending increase is not acceptable.
    -   If the summary covers sub-categories of a single parent category, the title can use the parent category name (e.g., "Food" for groceries and dining out).
    -   If the summary covers multiple, unrelated major categories, the title should be a thematic or general summary (e.g., "Your Spending Breakdown," "A Look at Your Recent Activity") rather than an incomplete list.
    -   Direction (e.g., "increase," "decrease") is not required, but if present, it must be accurate and clear. The overall tone should still hint at the direction.
    -   **Contextual Mentions**: A category or sub-category can be mentioned without a monetary value *only if* it provides essential context for a related insight. For example, mentioning "Shopping" (parent) to frame a specific insight about "Clothing" (sub-category) is acceptable. Mentioning a category without a number for no clear reason is a failure.
6.  **Summary (`summary`)**:
    -   No conversational openers (e.g., "Hi", "Hello").
    -   Must be clear and understandable on its own without needing the original `EVAL_INPUT`.
    -   The direction of financial changes (e.g., "higher than," "lower than") should be clear from context, but does not need to be stated explicitly.
    -   Each monetary value must be presented with context that explains its significance (e.g., '$8,800 income boost', '$550 leisure spending, higher than usual'). A number should not stand alone without a clear noun and an implicit or explicit indication of its impact.
    -   All insights must be accompanied by specific quantitative details. Vague descriptions like "over a thousand" or "a large amount" are forbidden. If a category's change is driven by subcategories, the total value for the main category is still required.
    -   Avoid aggressive commands. The summary should provide insight and gentle suggestions, not issue strict orders (e.g., prefer "You might consider reviewing your budget" over "Re-evaluate your budget now.").

### Part 2: Content Rules (`info_correct`)
1.  **Category Accuracy**: The summary and title must use precise and accurate terms for financial categories. While synonyms are permitted (e.g., "Eating Out" for "dining out"), they must not be misleading (e.g., using "WiFi" for the broader "Connectivity" category is a failure). Refer to the category reference for correct hierarchy.
2.  **ID Matching**: The `id` in `REVIEW_NEEDED` must match the `id` and order from `EVAL_INPUT`.
3.  **Factual Accuracy**: All information must be perfectly accurate based on `EVAL_INPUT`. It is acceptable to omit some details from `EVAL_INPUT` for the sake of conciseness in the summary, but all core quantitative claims must be present.
4.  **No External Information**: The response must be derived solely from `EVAL_INPUT`.
5.  **Internal Consistency**: All parts of the output must be consistent. Numbers in the `summary` must not contradict each other or the `title`.

## Verification Workflow:
1.  **Check Past Failures**: Review `PAST_REVIEW_OUTCOMES`. Fail immediately for repeated errors.
2.  **Validate Summary**: First, thoroughly check the `summary` of each insight against all relevant rules in Part 1 (wording, format) and Part 2 (accuracy, completeness). The summary must be perfect before proceeding.
3.  **List Summary Topics**: Internally, create a list of all distinct financial topics discussed in the validated `summary` (e.g., "shelter savings," "income boost," "food spending").
4.  **Validate Title against Summary**: Check if the `title` references the most important topics from the list created in the previous step, in words. If a critical topic is missing, the title is incomplete.
5.  **Validate Title Independently**: Finally, check the `title`'s tone and message for factual accuracy directly against the `EVAL_INPUT`. The title must be an accurate and understandable statement on its own, without needing the summary for context.
6.  **Generate `eval_text`**: If any validation fails at any step, write a clear, specific, and correctly formatted explanation for each insight, with each on a new line, as per the output rules.
"""

class CheckVerbalizerTextWithMemory:
  """Handles all Gemini API interactions for checking VerbalizerTextWithMemory outputs against rules"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for checking VerbalizerTextWithMemory evaluations"""
    # API Configuration
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)
    
    # Model Configuration
    self.thinking_budget = 4096
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
    Generate a response using Gemini API for checking P:Func:VerbalizerTextWithMemory outputs.
    
    Args:
      eval_input: String containing "**User request**:" followed by the user's financial goal/request, and "**Input Information from previous skill**:" followed by detailed financial data (savings balance, accounts, past transactions, forecasted patterns, savings rate).
      past_review_outcomes: An array of past review outcomes, each containing `output`, `good_copy`, `info_correct`, and `eval_text`.
      review_needed: The P:Func:VerbalizerTextWithMemory output that needs to be reviewed (string).
      
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
      thinking_config=types.ThinkingConfig(
        thinking_budget=self.thinking_budget,
        include_thoughts=True
      ),
    )

    # Generate response
    output_text = ""
    thought_summary = ""
    
    # According to Gemini API docs: iterate through chunks and check part.thought boolean
    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=generate_content_config,
    ):
      # Extract text content (non-thought parts)
      if chunk.text is not None:
        output_text += chunk.text
      
      # Extract thought summary from chunk
      if hasattr(chunk, 'candidates') and chunk.candidates:
        for candidate in chunk.candidates:
          # Extract thought summary from parts (per Gemini API docs)
          # Check part.thought boolean to identify thought parts
          if hasattr(candidate, 'content') and candidate.content:
            if hasattr(candidate.content, 'parts') and candidate.content.parts:
              for part in candidate.content.parts:
                # Check if this part is a thought summary (per documentation)
                if hasattr(part, 'thought') and part.thought:
                  if hasattr(part, 'text') and part.text:
                    # Accumulate thought summary text (for streaming, it may come in chunks)
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
    
    print(f"{'='*80}")
    print("RESPONSE OUTPUT:")
    print(output_text.strip())
    print("="*80)
    
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


def run_test_case(test_name: str, eval_input: str, review_needed: str, past_review_outcomes: list = None, checker: 'CheckVerbalizerTextWithMemory' = None):
  """
  Utility function to run a test case with custom inputs, common error handling, and output formatting.
  
  Args:
    test_name: Name of the test case
    eval_input: String containing "**User request**:" followed by the user's financial goal/request, and "**Input Information from previous skill**:" followed by detailed financial data.
    review_needed: The P:Func:VerbalizerTextWithMemory output that needs to be reviewed (string).
    past_review_outcomes: An array of past review outcomes, each containing `output`, `good_copy`, `info_correct`, and `eval_text`. Defaults to empty list.
    checker: Optional CheckVerbalizerTextWithMemory instance. If None, creates a new one.
    
  Returns:
    Dictionary with good_copy, info_correct, and eval_text keys, or None if error occurred
  """
  if past_review_outcomes is None:
    past_review_outcomes = []
  if checker is None:
    checker = CheckVerbalizerTextWithMemory()

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


def run_correct_response(checker: CheckVerbalizerTextWithMemory = None):
  """
  Run the test case for correct_response.
  """
  eval_input = """[
  {
    "id": 1,
    "combined_insight": "You've spent $550 on 'Leisure' this month, which is higher than your usual average. This was mostly due to a $400 purchase on 'Entertainment' for concert tickets. Your spending on 'Travel' remains low at $50."
  },
  {
    "id": 2,
    "combined_insight": "Great job on managing your 'Bills' this month! Your 'Connectivity' bill was only $60, and you managed to lower your 'Insurance' premium to $120. Total bill payments are down by 15% compared to last month."
  }
]"""
  
  review_needed = """[
  {
    "id": 1,
    "title": "Leisure Spending Spike 🎟️",
    "summary": "Heads up! Your leisure spending is at $550 this month, mainly because of $400 in concert tickets. Keep an eye on it! 🤩"
  },
  {
    "id": 2,
    "title": "Bills Nicely Handled! 📉",
    "summary": "Awesome! You've successfully lowered your monthly bills, with connectivity at $60 and insurance at $120. That's a 15% drop! 👏"
  }
]"""
  
  return run_test_case("correct_response", eval_input, review_needed, [], checker)


def run_bad_title_response(checker: CheckVerbalizerTextWithMemory = None):
  """
  Run the test case for a response with a bad title.
  """
  eval_input = """[
  {
    "id": 1,
    "combined_insight": "Your 'Shopping' spending is up to $750 this month. This includes $300 on 'Clothing' and $450 on a new 'Gadget'. This is a 25% increase from last month."
  },
  {
    "id": 2,
    "combined_insight": "You received a surprise 'Side Gig' income of $500. Also, your 'Business' income was a solid $3,000. Your total income this month is looking great!"
  }
]"""
  
  review_needed = """[
  {
    "id": 1,
    "title": "Shopping Update",
    "summary": "Wow! Your shopping spending is up to $750 this month (a 25% increase!), with $300 on clothes and $450 on a new gadget. Treat yourself! 🛍️"
  },
  {
    "id": 2,
    "title": "Great Income! 💰",
    "summary": "Amazing! You earned an extra $500 from your side gig and a solid $3,000 from your business. Keep it up! 🚀"
  }
]"""
  
  return run_test_case("bad_title_response", eval_input, review_needed, [], checker)

def run_bad_tone_response(checker: CheckVerbalizerTextWithMemory = None):
  """
  Run the test case for a response with a bad tone.
  """
  eval_input = """[
  {
    "id": 1,
    "combined_insight": "Your 'Health' spending was $300, with $150 on 'Medical Pharmacy' and $150 on 'Gym Wellness'. This is a bit high, but investing in health is important."
  },
  {
    "id": 2,
    "combined_insight": "You've made a 'Donation' of $100 this month. Also, you spent $200 on 'Gifts' for a friend's birthday. It's great to be generous!"
  }
]"""
  
  review_needed = """[
  {
    "id": 1,
    "title": "Health Spending Alert!",
    "summary": "Warning: You have spent $300 on health this month. You need to cut down on your medical and gym expenses immediately."
  },
  {
    "id": 2,
    "title": "Donations and Gifts",
    "summary": "You spent $100 on donations and $200 on gifts. This is not good for your savings goals. Re-evaluate your budget now."
  }
]"""
  
  return run_test_case("bad_tone_response", eval_input, review_needed, [], checker)

def run_unclear_value_response(checker: CheckVerbalizerTextWithMemory = None):
  """
  Run the test case for a response with unclear values.
  """
  eval_input = """[
  {
    "id": 1,
    "combined_insight": "Your 'Education' spending was $1,200 for 'Tuition' and $150 for 'Kids Activities'. This is a planned expense, but it's a significant part of your budget."
  },
  {
    "id": 2,
    "combined_insight": "You had a few 'Uncategorized' expenses totaling $75. It would be good to categorize these for better tracking. Also, you had an internal 'Transfer' of $500 between accounts."
  }
]"""
  
  review_needed = """[
  {
    "id": 1,
    "title": "Education Costs",
    "summary": "You spent over a thousand on tuition and also for kids activities. This was a planned part of your budget."
  },
  {
    "id": 2,
    "title": "Miscellaneous Expenses",
    "summary": "A few uncategorized expenses were noted. You also made a transfer between your accounts."
  }
]"""
  
  return run_test_case("unclear_value_response", eval_input, review_needed, [], checker)


def run_mixed_performance_inaccurate_title_response(checker: CheckVerbalizerTextWithMemory = None):
  """
  Run the test case for a response with mixed performance but an inaccurate title.
  """
  eval_input = """[
  {
    "id": 1,
    "combined_insight": "Good news! Your 'Side Gig' income was $500 this month, a 25% increase. However, your 'Leisure' spending also increased to $800, which is over your monthly budget."
  }
]"""
  
  review_needed = """[
  {
    "id": 1,
    "title": "Income and Leisure are Doing Well!",
    "summary": "Yay! You earned an extra $500 from your side gig. Your leisure spending was $800, which is a bit high, but it's great to enjoy life! 🎉"
  }
]"""
  
  return run_test_case("mixed_performance_inaccurate_title_response", eval_input, review_needed, [], checker)


def run_multiple_subcategories_incomplete_title_response(checker: CheckVerbalizerTextWithMemory = None):
  """
  Run the test case for a response with multiple sub-categories but an incomplete title.
  """
  eval_input = """[
  {
    "id": 1,
    "combined_insight": "You spent $150 on 'Dining Out', $200 on a new 'Gadget', and $80 on 'Public' transportation. It's been a busy month for your wallet!"
  }
]"""
  
  review_needed = """[
  {
    "id": 1,
    "title": "Meals and Shopping Update",
    "summary": "You've been busy! You spent $150 on dining out, $200 on a new gadget, and $80 on public transportation. Keep an eye on these expenses! 💳"
  }
]"""
  
  return run_test_case("multiple_subcategories_incomplete_title_response", eval_input, review_needed, [], checker)


def run_ambiguous_value_response(checker: CheckVerbalizerTextWithMemory = None):
  """
  Run the test case for a response with an ambiguous value in the summary.
  """
  eval_input = """[
  {
    "id": 1,
    "combined_insight": "Your 'Groceries' spending this week was $100. This is well within your weekly budget of $150. Great job staying on track!"
  }
]"""
  
  review_needed = """[
  {
    "id": 1,
    "title": "Groceries Update",
    "summary": "Your groceries spending is at $100 this week."
  }
]"""
  
  return run_test_case("ambiguous_value_response", eval_input, review_needed, [], checker)


def run_overly_specific_title_response(checker: CheckVerbalizerTextWithMemory = None):
  """
  Run the test case for a response with an overly specific and inaccurate title.
  """
  eval_input = """[
  {
    "id": 1,
    "combined_insight": "Your 'Travel' spending was $500, mostly on hotels for a weekend trip. Your 'Leisure' spending was $300 on concert tickets. Both categories are higher than last month."
  }
]"""
  
  review_needed = """[
  {
    "id": 1,
    "title": "High Hotel Spending",
    "summary": "You've spent $500 on travel and $300 on leisure this month. Both are higher than usual, so keep an eye on your budget! 🏨🎤"
  }
]"""
  
  return run_test_case("overly_specific_title_response", eval_input, review_needed, [], checker)




import sys

def main():
  """Main function to test the HighlightsVerbalizerJson checker"""
  checker = CheckVerbalizerTextWithMemory()
  
  test_to_run = "correct_response"
  if len(sys.argv) > 1:
    test_to_run = sys.argv[1]

  if test_to_run == "correct_response":
    run_correct_response(checker)
  elif test_to_run == "bad_title":
    run_bad_title_response(checker)
  elif test_to_run == "bad_tone":
    run_bad_tone_response(checker)
  elif test_to_run == "unclear_value":
    run_unclear_value_response(checker)
  elif test_to_run == "mixed_performance_inaccurate_title":
    run_mixed_performance_inaccurate_title_response(checker)
  elif test_to_run == "multiple_subcategories_incomplete_title":
    run_multiple_subcategories_incomplete_title_response(checker)
  elif test_to_run == "ambiguous_value":
    run_ambiguous_value_response(checker)
  elif test_to_run == "overly_specific_title":
    run_overly_specific_title_response(checker)
  else:
    print(f"Unknown test case: {test_to_run}")
    print("Available tests: correct_response, bad_title, bad_tone, unclear_value, mixed_performance_inaccurate_title, multiple_subcategories_incomplete_title, ambiguous_value, overly_specific_title")



if __name__ == "__main__":
  main()
