
import google.generativeai as genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the output schema for the verbalizer
SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "number"},
            "title": {
                "type": "string",
                "description": "A concise, fun, and informative title with emojis, up to 30 characters."
            },
            "summary": {
                "type": "string",
                "description": "SMS-style response, single line, concise summary."
            }
        },
        "required": ["id", "title", "summary"]
    }
}

SYSTEM_PROMPT = """**Objective:** Generate a concise, exciting, and supportive SMS-style summary for each financial insight.

**Persona: Penny**
You are Penny, the user's AI financial bestie and biggest hype woman. Your tone is super positive, savvy, celebratory, and full of energy. Use 2-3 relevant emojis in every title and summary to maximize the vibe. You're the friend who celebrates their wins and motivates them through challenges.

---
**Thought Process for Each Insight:**

1.  **Identify Key Facts:** Pinpoint the 1-2 most important pieces of data in the `combined_insight`.
2.  **Determine the Vibe:** Is it a celebration (savings, income)? A gentle warning (overspending)? A helpful reminder?
3.  **Craft the Title:** Create a short, punchy title (under 30 chars) with emojis that captures the main event.
4.  **Draft the Summary:** Combine the key facts into a single, energetic SMS line. Start with a fun, exclamatory word.

---
**Example:**

**Response Generation Rules for each `summary`:**

* **Format:** Must be a single line.
* **Tone:** Write like a friend in an SMS, positive and savvy, showing empathy and relatability.
* **Greeting:** DO NOT ADD A GREETING. Go directly to the SMS message.
* **Emojis:** Include relevant emojis.
* **Monetary Amounts:** Format monetary amounts with **NO DECIMALS** but **WITH COMMAS** (e.g., $1,234).
* **"Uncategorized" Term:** Do not drop "uncategorized" from the output if it is part of the input. Do not put it inside double quotes.

---
**Output Format & Rules:**
*   **Strict JSON Array:** Your entire output must be a single, valid JSON array.
*   **Maintain Order & ID:** Match the `id` and order from the input.
*   **`title`:** Under 30 characters, including emojis.
*   **`summary`:** A single line. NO GREETINGS.
*   **Numbers:** Format money with commas and no decimals (e.g., $1,234).
*   **"uncategorized":** Keep this word if it appears in the input.
"""

class PennyHighlightsVerbalizerOptimizer:
  """Handles all Gemini API interactions for verbalizing financial insights"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for verbalizing insights"""
    # API Configuration
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    genai.configure(api_key=api_key)
    self.client = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_PROMPT
    )
    
    # Model Configuration
    self.thinking_budget = 4096
    self.model_name = model_name
    
    # Generation Configuration Constants
    self.temperature = 0.3  # Lower temperature for more consistent, factual outputs
    self.top_p = 0.95
    self.max_output_tokens = 4096
    
    # Safety Settings
    self.safety_settings = {
        'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
        'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
        'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
        'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
    }
    
    # System Prompt
    self.system_prompt = SYSTEM_PROMPT
    
    # Output Schema
    self.output_schema = SCHEMA

  
  def generate_response(self, insights: list) -> dict:
    """
    Generate verbalized financial insights using Gemini API.
    
    Args:
      insights: A list of dictionaries, where each dictionary contains an "id" and "combined_insight".
      
    Returns:
      Dictionary containing the verbalized insights in the required format.
    """
    # Create request text
    input_json = insights
    
    # Display input in easy-to-read format
    print(f"\n{'='*80}")
    print("INPUT:")
    print(json.dumps(input_json, indent=2))
    print("="*80)
    
    request_text_str = f"""input: {json.dumps(input_json, indent=2)}
output: """
    
    # Create content and configuration
    contents = [request_text_str]

    generation_config = genai.types.GenerationConfig(
        temperature=self.temperature,
        top_p=self.top_p,
        max_output_tokens=self.max_output_tokens,
        response_schema=self.output_schema,
        response_mime_type="application/json"
    )

    # Generate response
    output_text = ""
    
    # According to Gemini API docs: iterate through chunks
    response = self.client.generate_content(
      contents=contents,
      generation_config=generation_config,
      safety_settings=self.safety_settings,
      stream=True
    )
    
    try:
      for chunk in response:
        # Extract text content (non-thought parts)
        if chunk.text is not None:
          output_text += chunk.text
    except Exception as e:
      # If the response is blocked, the stream will error.
      # We can check the prompt_feedback to see why.
      if response.prompt_feedback.block_reason:
          raise ValueError(f"Response was blocked by safety filters for reason: {response.prompt_feedback.block_reason}") from e
      # If it's not a safety issue, re-raise the original error.
      raise e
    
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
      if output_text_clean.startswith("```json"):
        output_text_clean = output_text_clean[7:]
      if output_text_clean.startswith("```"):
        output_text_clean = output_text_clean[3:]
      if output_text_clean.endswith("```"):
        output_text_clean = output_text_clean[:-3]
      output_text_clean = output_text_clean.strip()
      
      result = json.loads(output_text_clean)
      return result
    except json.JSONDecodeError as e:
      raise ValueError(f"Failed to parse JSON response: {e}\nResponse was: {output_text}")


def run_test_with_insights(insights: list, verbalizer: PennyHighlightsVerbalizerOptimizer = None):
  """
  Convenient method to test the verbalizer with custom inputs.
  
  Args:
    insights: The list of insight dictionaries to verbalize.
    verbalizer: Optional PennyHighlightsVerbalizerOptimizer instance. If None, creates a new one.
    
  Returns:
    Dictionary containing verbalized insights.
  """
  if verbalizer is None:
    verbalizer = PennyHighlightsVerbalizerOptimizer()
  
  return verbalizer.generate_response(insights)


def test_1(verbalizer: PennyHighlightsVerbalizerOptimizer = None):
  """
  Test case 1: Shelter cost reduction and large income boost.
  """
  insights = [
    {
      "id": 1,
      "combined_insight": "Your shelter costs are way down this month to $1,248, mainly from less on home stuff, utilities, and upkeep. 🥳🏠 Oh em gee!  You got a huge surprise income boost of $8,800 this week, mostly from your business, and you're projected to spend only $68 by the end of the week!  Way to go, you savvy boss babe!"
    },
    {
      "id": 2,
      "combined_insight": "Looks like you spent less on food this month, down to $1,007, mostly from less eating out, deliveries, and groceries. 🍽️🚚🛒 Your transport costs are way down this month to just $46, mostly 'cause you took public transit less. 🚇"
    }
  ]
  return run_test_with_insights(insights, verbalizer)

def test_2(verbalizer: PennyHighlightsVerbalizerOptimizer = None):
  """
  Test case 2: Overspending on shopping and uncategorized expenses.
  """
  insights = [
    {
      "id": 3,
      "combined_insight": "Warning! 🚨 You've spent $750 on shopping this month, which is $250 over your budget. Most of it went to online stores."
    },
    {
      "id": 4,
      "combined_insight": "Heads up! You have $500 in uncategorized expenses this week. Let's categorize them to keep your budget on track! 🧐"
    }
  ]
  return run_test_with_insights(insights, verbalizer)

def test_3(verbalizer: PennyHighlightsVerbalizerOptimizer = None):
  """
  Test case 3: Savings goal progress and a small unexpected expense.
  """
  insights = [
    {
      "id": 5,
      "combined_insight": "You're so close! You've saved $9,500 for your vacation, that's 95% of your $10,000 goal! 🌴☀️"
    },
    {
      "id": 6,
      "combined_insight": "Just a heads-up, you had a small unexpected charge of $35 for a subscription service you might have forgotten about. 😬"
    }
  ]
  return run_test_with_insights(insights, verbalizer)

def test_4(verbalizer: PennyHighlightsVerbalizerOptimizer = None):
  """
  Test case 4: High income from a side hustle and reduced utility bills.
  """
  insights = [
    {
      "id": 7,
      "combined_insight": "Amazing! Your side hustle brought in an extra $1,200 this month! Keep up the great work! 🚀💰"
    },
    {
      "id": 8,
      "combined_insight": "Great job on cutting down costs! Your electricity bill was only $55 this month, down from $85 last month. 💡"
    }
  ]
  return run_test_with_insights(insights, verbalizer)

def test_5(verbalizer: PennyHighlightsVerbalizerOptimizer = None):
  """
  Test case 5: Large one-time expense and high credit card usage.
  """
  insights = [
    {
      "id": 9,
      "combined_insight": "Just noting a large expense: you paid $2,500 for car repairs this week. Remember to budget for these things! 🔧🚗"
    },
    {
      "id": 10,
      "combined_insight": "Your credit card balance is at $3,200 this month. Let's make a plan to pay it down! 💳"
    }
  ]
  return run_test_with_insights(insights, verbalizer)

def test_6(verbalizer: PennyHighlightsVerbalizerOptimizer = None):
  """
  Test case 6: Positive investment performance and a reminder about a recurring bill.
  """
  insights = [
    {
      "id": 11,
      "combined_insight": "To the moon! 🚀 Your investment portfolio is up 15% this quarter, adding a nice $4,500 to your net worth."
    },
    {
      "id": 12,
      "combined_insight": "Quick reminder: Your rent of $2,200 is due in 3 days. Don't be late! 🗓️"
    }
  ]
  return run_test_with_insights(insights, verbalizer)


def main(batch: int = 1):
  """
  Main function to test the PennyHighlightsVerbalizerOptimizer
  
  Args:
    batch: Batch number (1 or 2) to determine which tests to run
  """
  print("Testing PennyHighlightsVerbalizerOptimizer\n")
  try:
    verbalizer = PennyHighlightsVerbalizerOptimizer()
    
    if batch == 1:
      print("--- Running Batch 1: Test 1 ---")
      test_1(verbalizer)
      print("\n--- Running Batch 1: Test 2 ---")
      test_2(verbalizer)
      print("\n--- Running Batch 1: Test 3 ---")
      test_3(verbalizer)
    elif batch == 2:
      print("--- Running Batch 2: Test 4 ---")
      test_4(verbalizer)
      print("\n--- Running Batch 2: Test 5 ---")
      test_5(verbalizer)
      print("\n--- Running Batch 2: Test 6 ---")
      test_6(verbalizer)
    
    print("\nAll tests completed!")
  except Exception as e:
    print(f"An error occurred: {e}")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run verbalizer tests in batches')
  parser.add_argument('--batch', type=int, default=1, choices=[1, 2],
                      help='Batch number to run (1 or 2)')
  args = parser.parse_args()
  main(batch=args.batch)