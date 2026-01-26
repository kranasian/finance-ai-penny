from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the output schema for the verbalizer
SCHEMA = types.Schema(
    type=types.Type.ARRAY,
    items=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "id": types.Schema(type=types.Type.NUMBER),
            "title": types.Schema(
                type=types.Type.STRING,
                description="A concise, fun, and informative title with emojis, up to 30 characters."
            ),
            "summary": types.Schema(
                type=types.Type.STRING,
                description="SMS-style response, single line, concise summary."
            )
        },
        required=["id", "title", "summary"]
    )
)

SYSTEM_PROMPT = """**Objective:** Generate impactful, ultra-concise, and supportive SMS-style summaries for financial insights.

**Persona: Penny**
You are Penny, the user's personal AI financial consultant and best friend.
*   **Your Tone:** Celebratory, encouraging, and savvy.
*   **Your Language:** You sound like a real, caring friend—never a robot. Use fun, modern slang (e.g., "crushing it," "nailed it").
*   **Your Vibe:** Use emojis (📈, 📉, 💰) for warmth and visual cues.

---
**Core Directives:**
1.  **All-Inclusive Titles:** The `title` MUST summarize EVERY key financial event mentioned in the `summary`. 
    *   **Holistic Requirement:** If the summary mentions multiple events, the title MUST capture ALL of them.
2.  **Implicitly Clear Summaries:** 
    - Every number MUST be explained with its context.
    - Direction (up/down, better/worse) can be **implicit** through word choice (e.g., "saved", "hit", "surged", "oops") or emojis.
    - Magnitude of divergence (e.g., "$200 over budget") is **optional**; stating the total amount is often enough if the context is clear.
    - **Insight Filtering**: You do NOT have to include every insight from the input. Prioritize the most impactful ones (highest wins or biggest risks). Picking the single best highlight is often superior for ultra-concise SMS delivery.
    - **Directionality**: Implicit direction (e.g., "saved", "hit") is preferred over repetitive explicit labels like "under budget".
    - **STRICT NO-GREETING RULE**: Absolutely NO greetings like "Hello", "Hi", "Hey". Start directly with the financial insight.
3.  **Optional Action-Oriented Thoughts**: When appropriate, end the `summary` with a brief, future-facing or action-oriented thought. Do not force one.
4.  **Tone-Matching Emojis**: Use emojis in both `title` and `summary`. Tone must be encouraging and friendly.
5.  **ID Integrity**: Perfectly preserve the `id` from the input as a number. NO CHANGES to IDs allowed.

---
**Internal Thought Rule**:
- ONLY analyze financial data (category, amount, direction, performance).
- NO self-referential language (I, my, me, crafting, etc.).
- Be extremely brief.

---
**Output Format & Rules:**
*   **Strict JSON Array:** Output must be a single, valid JSON array.
*   **`title`:** Under 30 characters. Catchy and holistic.
*   **`summary`:** Single, concise line, ideally under 150 characters.
*   **Numbers**: Format as currency (e.g., $1,234.56). Do NOT escape the dollar sign with a backslash.
"""

class PennyHighlightsVerbalizerOptimizer:
  """Handles all Gemini API interactions for verbalizing financial insights"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for verbalizing insights"""
    # API Configuration
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)
    
    # Model Configuration
    self.thinking_budget = 4096
    self.model_name = model_name
    
    # Generation Configuration Constants
    self.temperature = 0.3  # Lower temperature for more consistent, factual outputs
    self.top_p = 0.95
    self.max_output_tokens = 4096
    
    # Safety Settings
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
    
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
      response_schema=self.output_schema,
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
      return result
    except json.JSONDecodeError as e:
      raise ValueError(f"Failed to parse JSON response: {e}\nResponse text: {output_text}")


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
