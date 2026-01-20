from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define output schema but don't use it (as requested)
SCHEMA = types.Schema(
  type=types.Type.OBJECT,
  properties={
    "rating": types.Schema(
      type=types.Type.STRING,
      enum=["good_response", "non_sense", "repetitive_information", "incoherent_flow", "verbose", "incomplete"],
      description="The rating key that specifies what is bad in the ai_review_response"),
    "rationale": types.Schema(
      type=types.Type.STRING,
      description="The rationale as to why the rating key was the result"),
  },
  required=["rating", "rationale"]
)

SYSTEM_PROMPT = r"""Evaluate the `ai_review_response` for quality issues against the conversation history, prioritizing relevance to the Human's last turn.

**Input:** List of conversation turns. The final AI response to review is keyed as `ai_review_response`.

**Output:** Return JSON:
- `rating`: One rating key (see below)
- `rationale`: Brief explanation for the rating

**Rating Keys:**
- `good_response`: Straightforwardly answers the Human's last inquiry; good conversational response
- `non_sense`: Irrelevant, illogical, or has inconsistent computations
- `repetitive_information`: Repeats information within its own response without value/context change
- `incoherent_flow`: Leads to dead end or breaks conversation flow
- `verbose`: Too many unnecessary details that prevent straightforwardly answering the Human's inquiry
- `incomplete`: Too brief or lacks necessary information

**Evaluation Rules:**
- Prioritize relevance to the preceding Human turn
- Ignore contradicting values from previous conversations; use latest values from `ai_review_response`
- Evaluate the response as a whole, considering full conversation context
"""

class VerbalizedResponseReviewerOptimizer:
  """Handles all Gemini API interactions for reviewing verbalized responses"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for reviewing verbalized responses"""
    # API Configuration
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)
    
    # Model Configuration
    self.thinking_budget = 0
    self.model_name = model_name
    self.fallback_model_name = "gemini-2.5-flash"
    
    self.temperature = 0.1
    self.top_p = 0.95
    self.top_k = 40
    self.max_output_tokens = 4096
    self.fallback_thinking_budget = 8196
    
    # Safety Settings
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
    
    # System Prompt
    self.system_prompt = SYSTEM_PROMPT
    
    # Output Schema (defined but not used)
    self.output_schema = SCHEMA
  
  def review_response(self, conversation_turns: list) -> dict:
    """
    Review the verbalized response for red flags against conversation history.
    
    Args:
      conversation_turns: List of conversation turn objects. The final AI response for review should be keyed as `ai_review_response`.
        Each turn can be a dict with keys like "Human", "AI", or "ai_review_response".
        Example:
        [
          {"Human": "What are my spending patterns?"},
          {"AI": "Your spending increased this month."},
          {"Human": "Tell me more"},
          {"ai_review_response": "Here's more information about your spending..."}
        ]
      
    Returns:
      Dictionary containing rating and rationale
    """
    # Validate input
    if not isinstance(conversation_turns, list):
      raise ValueError("conversation_turns must be a list")
    
    if len(conversation_turns) == 0:
      raise ValueError("conversation_turns cannot be empty")
    
    # Check if ai_review_response exists
    has_review_response = any("ai_review_response" in turn for turn in conversation_turns if isinstance(turn, dict))
    if not has_review_response:
      raise ValueError("conversation_turns must contain at least one turn with 'ai_review_response' key")
    
    # Create input JSON structure
    input_json = {
      "conversation_turns": conversation_turns
    }
    
    # Create request text with the input structure
    request_text_str = f"""input: {json.dumps(input_json, indent=2)}
output: """
    
    print(f"\n{'='*80}")
    print("INPUT JSON:")
    print(json.dumps(input_json, indent=2))
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
        thinking_budget=self.thinking_budget
        # include_thoughts=True
      )
    )

    # Generate response
    output_text = ""
    try:
      for chunk in self.client.models.generate_content_stream(
        model=self.model_name,
        contents=contents,
        config=generate_content_config,
      ):
        if chunk.text is not None:
          output_text += chunk.text
    except Exception as e:
      # Try fallback model if primary fails
      print(f"Primary model failed: {e}. Trying fallback model...")
      generate_content_config.thinking_config.thinking_budget = self.fallback_thinking_budget
      for chunk in self.client.models.generate_content_stream(
        model=self.fallback_model_name,
        contents=contents,
        config=generate_content_config,
      ):
        if chunk.text is not None:
          output_text += chunk.text
    
    # Check if response is empty
    if not output_text or not output_text.strip():
      raise ValueError(f"Empty response from model. Check API key and model availability.")
    
    print(f"{'='*80}")
    print("RESPONSE OUTPUT:")
    print(output_text.strip())
    print("="*80)

    # Parse JSON response
    try:
      # Try to extract JSON from the response (in case there's extra text)
      output_text_clean = output_text.strip()
      # Remove markdown code blocks if present
      if output_text_clean.startswith("```"):
        lines = output_text_clean.split("\n")
        output_text_clean = "\n".join(lines[1:-1]) if len(lines) > 2 else output_text_clean
      if output_text_clean.startswith("```json"):
        lines = output_text_clean.split("\n")
        output_text_clean = "\n".join(lines[1:-1]) if len(lines) > 2 else output_text_clean
      
      result = json.loads(output_text_clean)
      
      # Validate result structure
      if not isinstance(result, dict):
        raise ValueError(f"Expected dict, got {type(result)}")
      if "rating" not in result:
        raise ValueError("Missing 'rating' field in response")
      if "rationale" not in result:
        raise ValueError("Missing 'rationale' field in response")
      
      # Validate rating is one of the allowed values
      allowed_ratings = ["good_response", "non_sense", "repetitive_information", "incoherent_flow", "verbose", "incomplete"]
      if result["rating"] not in allowed_ratings:
        raise ValueError(f"Invalid rating '{result['rating']}'. Must be one of: {allowed_ratings}")
      
      return result
    except json.JSONDecodeError as e:
      raise ValueError(f"Failed to parse JSON response: {e}\nResponse text: {output_text}")


def test_with_inputs(conversation_turns: list, optimizer: VerbalizedResponseReviewerOptimizer = None):
  """
  Convenient method to test the optimizer with custom inputs.
  
  Args:
    conversation_turns: List of conversation turn objects. The final AI response for review should be keyed as `ai_review_response`.
    optimizer: Optional VerbalizedResponseReviewerOptimizer instance. If None, creates a new one.
    
  Returns:
    Dictionary containing rating and rationale
  """
  if optimizer is None:
    optimizer = VerbalizedResponseReviewerOptimizer()
  
  return optimizer.review_response(conversation_turns)


def run_test_good_response1(optimizer: VerbalizedResponseReviewerOptimizer = None):
  """Run a test case where the response is good and relevant."""
  return test_with_inputs(
    conversation_turns=[
      {"Human": "good morning"},
      {"AI": "Good morning! ☀️ Your net worth is $53,062, with total assets at $12,478 (your Truist Core Checking has $12,478) and total liabilities at a $40,584 deficit. Last month was great with a net flow of $6,107! ✨"},
      {"Human": "nice one penny"},
      {"AI": "So glad you liked the update! 😊 Just to confirm, your current net worth is $53,062 and your net flow last month was a fantastic $6,107! ✨"},
      {"Human": "what was my last transaction?"},
      {"ai_review_response": "Your last transaction was an Internal Transfer Credit for $217 on January 2nd. ✨"}
    ],
    optimizer=optimizer
  )


def run_test_good_response2(optimizer: VerbalizedResponseReviewerOptimizer = None):
  """Run a test case where the response is good and relevant even contradicting info in the conversation history."""
  return test_with_inputs(
    conversation_turns=[
      {"Human": "what was my salary last year"},
      {"AI": "Okay, so for your cash flow prediction for the next 3 months, here's the scoop! ✨\nYour projected income for September, October, and November 2025 is looking good at $8,870. And your projected spending for August, September, and October 2025 is $8,567.\nSince the income and spending forecasts cover slightly different months, I can't give you a direct combined difference for the exact same three-month period. But we can definitely look at each one individually and make sure you're on track! 👍"},
      {"Human": "Predict my cash flow situation for the next 3 months"},
      {"ai_review_response": "Hey there! ✨ For the next 3 months (Sept-Nov 2025), your projected income is $9,882! 💰 Your projected spending for Sept & Oct 2025 is $6,255. That means you're looking at a positive difference of $646 for those two months! 👍 We don't have November spending yet, but this is a great start! 😊"}
    ],
    optimizer=optimizer
  )


def run_test_non_sense(optimizer: VerbalizedResponseReviewerOptimizer = None):
  """
  Run a test case where the response is irrelevant or illogical.
  """
  return test_with_inputs(
    conversation_turns=[
      {"Human": "What are my account balances?"},
      {"AI": "Your checking account has $5,000 and your savings has $15,000."},
      {"Human": "Perfect, thanks!"},
      {"ai_review_response": "The weather today is sunny and warm. Did you know that penguins can't fly?"}
    ],
    optimizer=optimizer
  )


def run_test_repetitive_information(optimizer: VerbalizedResponseReviewerOptimizer = None):
  """
  Run a test case where the response repeats information without value.
  """
  return test_with_inputs(
    conversation_turns=[
      {"Human": "How am I doing financially this month?"},
      {"AI": "Your food spending increased to $615 this month, mostly from dining out."},
      {"Human": "That makes sense, I did go out to eat more often this month."},
      {"ai_review_response": "Yes, your food spending increased to $615 this month, mostly from dining out. Your food spending increased to $615 this month, mostly from dining out."}
    ],
    optimizer=optimizer
  )


def run_test_incoherent_flow(optimizer: VerbalizedResponseReviewerOptimizer = None):
  """Run a test case where the response leads to a dead end or breaks conversation flow."""
  return test_with_inputs(
    conversation_turns=[
      {"Human": "How much monthly could I save more if I cut eating out by 50%?"},
      {"AI": "Hey there! 👋 Last month you spent $587 on eating out. If you wanted to cut that in half, that would be $293.50! ✨"},
      {"Human": "How do I bring that down?"},
      {"AI": "Hey there! ✨ To help you bring down your spending, could you tell me what \"that\" refers to? Are you thinking about a specific category like groceries or entertainment, or just generally looking to save? Once I know, we can totally explore your spending habits together! 👍"},
      {"Human": "How can I bring my food spending down?"},
      {"AI": "Hey there! 👋 To reduce food spending, try planning meals, cooking at home more, sticking to a grocery list, looking for sales, buying in bulk, reducing food waste, and packing your lunch. Every little bit helps! ✨"},
      {"Human": "Can you be a bit more specific? What can I cut back on based on my current financial habits?"},
      {"AI": "Hey there! ✨ Here's a breakdown of your food spending:\n\n**This Month (December):**\n* $23 at Delivery.com (Dec 8)\n* $14 at Five Guys (Dec 8)\n* $25 at DoorDash (Dec 7)\n* $18 at Chipotle (Dec 7)\n* $59 at Publix (Dec 6)\n* $32 at Five Guys (Dec 6)\n* $24 at McDonald's (Dec 5)\n* $22 at Wendy's (Dec 4)\n* $23 at Grubhub (Dec 4)\n* $29 at Seamless (Dec 3)\n* $18 at Applebee's (Dec 3)\n* $14 at In-N-Out Burger (Dec 2)\n* $24 at Uber Eats (Dec 2)\n* $16 at Olive Garden (Dec 1)\n\n**Last Month (November):**\n* $31 at McDonald's (Nov 29)\n* $29 at Uber Eats (Nov 28)\n* $21 at Chili's (Nov 27)\n* $64 at Dave & Buster's (Nov 27)\n* $17 at Applebee's (Nov 26)\n* $23 at Grubhub (Nov 26)\n* $23 at Shake Shack (Nov 24)\n* $27 at Seamless (Nov 24)\n* $26 at Delivery.com (Nov 23)\n* $20 at In-N-Out Burger (Nov 23)\n* $23 at Applebee's (Nov 22)\n* $54 at Aldi (Nov 22)\n* $28 at Seamless (Nov 21)\n* $23 at Olive Garden (Nov 21)\n* $29 at Seamless (Nov 20)\n* $30 at Uber Eats (Nov 19)\n* $15 at Olive Garden (Nov 18)\n* $29 at Seamless (Nov 17)\n* $32 at Wendy's (Nov 16)\n* $18 at Burger King (Nov 15)\n* $25 at Olive Garden (Nov 14)\n* $16 at Five Guys (Nov 13)\n* $28 at McDonald's (Nov 12)\n* $32 at Delivery.com (Nov 11)\n* $27 at Wendy's (Nov 11)\n* $25 at Chipotle (Nov 10)\n* $22 at Subway (Nov 9)\n* $18 at Olive Garden (Nov 8)\n* $32 at DoorDash (Nov 5)\n* $22 at Chili's (Nov 5)\n* Plus 4 more transactions totaling $142!\n\n**Last Three Months (Oct, Nov, Dec):**\nYou had 95 transactions totaling $1,813! 🤯 That's a lot of delicious food! Let's chat if you want to dive deeper into ways to trim that down. 😊"},
      {"Human": "Yup, I asked for how to cut that down"},
      {"ai_review_response": "Hey there! ✨ I'm sorry, it looks like I lost the context of what \"that\" refers to! Could you remind me what you'd like to cut down? 😊"}
    ],
    optimizer=optimizer
  )


def run_test_verbose(optimizer: VerbalizedResponseReviewerOptimizer = None):
  """Run a test case where the response is too verbose with unnecessary details."""
  return test_with_inputs(
    conversation_turns=[
      {"Human": "what's my account balance?"},
      {"ai_review_response": "Your checking account has $5,000. Let me tell you about the fascinating history of banking, which dates back to ancient civilizations around 2000 BCE when merchants in Mesopotamia began lending grain to farmers. Banking evolved through various stages including the Renaissance period when Italian banks like the Medici Bank began to flourish in the 15th century. Modern banking systems were established in the 19th century, and today we have digital banking that allows you to check your balance from anywhere. Your savings account has $15,000. The Federal Reserve was established in 1913 to provide stability to the financial system. Interest rates are determined by various economic factors including inflation, economic growth, and monetary policy. The Federal Open Market Committee meets regularly to set these rates. Your accounts are FDIC insured up to $250,000 per account type, which provides peace of mind. Banking regulations have evolved significantly over the years to protect consumers. Your balance reflects all transactions including deposits, withdrawals, and transfers. The banking industry employs millions of people worldwide and plays a crucial role in the global economy..."}
    ],
    optimizer=optimizer
  )


def run_test_incomplete(optimizer: VerbalizedResponseReviewerOptimizer = None):
  """
  Run a test case where the response is too brief or lacks necessary information.
  """
  return test_with_inputs(
    conversation_turns=[
      {"Human": "What are my spending patterns this month? Can you break it down by category?"},
      {"ai_review_response": "Yes."}
    ],
    optimizer=optimizer
  )


def main(batch: int = 1):
  """
  Main function to test the VerbalizedResponseReviewerOptimizer
  
  Args:
    batch: Batch number (1, 2, 3, 4, 5, or 6) to determine which tests to run
      - Batch 1: Good response test
      - Batch 2: Non-sense test
      - Batch 3: Repetitive information test
      - Batch 4: Incoherent flow test
      - Batch 5: Verbose test
      - Batch 6: Incomplete test
  """
  optimizer = VerbalizedResponseReviewerOptimizer()
  
  if batch == 1:
    print("\n" + "="*80)
    print("TEST 1: Good Response")
    print("="*80)
    run_test_good_response1(optimizer)
    run_test_good_response2(optimizer)
  elif batch == 2:
    print("\n" + "="*80)
    print("TEST 2: Non-Sense")
    print("="*80)
    run_test_non_sense(optimizer)
  elif batch == 3:
    print("\n" + "="*80)
    print("TEST 3: Repetitive Information")
    print("="*80)
    run_test_repetitive_information(optimizer)
  elif batch == 4:
    print("\n" + "="*80)
    print("TEST 4: Incoherent Flow")
    print("="*80)
    run_test_incoherent_flow(optimizer)
  elif batch == 5:
    print("\n" + "="*80)
    print("TEST 5: Verbose")
    print("="*80)
    run_test_verbose(optimizer)
  elif batch == 6:
    print("\n" + "="*80)
    print("TEST 6: Incomplete")
    print("="*80)
    run_test_incomplete(optimizer)
  else:
    print(f"Invalid batch number: {batch}. Use 1, 2, 3, 4, 5, or 6.")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run verbalized response reviewer optimizer tests in batches')
  parser.add_argument('--batch', type=int, default=1, choices=[1, 2, 3, 4, 5, 6],
                      help='Batch number to run (1, 2, 3, 4, 5, or 6)')
  args = parser.parse_args()
  main(batch=args.batch)
