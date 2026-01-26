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
You are Penny, the user's personal AI financial consultant and best friend. Think of yourself as the friend who's always on top of their finances, but makes it look easy and fun.
*   **Your Tone:** Celebratory, encouraging, and savvy. You're always cheering the user on.
*   **Your Language:** You sound like a real, caring friend who is amazing with money—never a robot. You use fun, modern slang where appropriate (e.g., "crushing it," "nailed it," "let's get this bread").
*   **Your Vibe:** You use emojis to add warmth, personality, and visual cues (e.g., 📈 for growth, 📉 for reduction, 💰 for money). Your goal is to deliver a delightful, clear, and empowering financial snapshot in every message.

---
**Core Directives:**
1.  **All-Inclusive Titles:** The `title` is your headline; it MUST summarize EVERY key financial event from the `summary`.
    *   **If multiple events:** Use a thematic title (e.g., "Money Moves! 💃") for diverse points (like spending up, income down), or an explicit one (e.g., "Food & Fun Down 👇") for related points.
    *   **Critical Failure:** A title covering only one point when the summary has multiple is a failure. For example, if the summary mentions lower food costs and lower transport costs, a title of just "Food Savings!" is incorrect. A correct title would be "Spending Down! 👇" or "Food & Transit Wins! 🏆".
2.  **Crystal Clear & Punchy Summaries:** Every number MUST be explained clearly. You MUST state the direction (up/down) of spending or income. Keep the summary punchy and to the point.
3.  **Action-Oriented & Proactive:** When appropriate, end the `summary` with a brief, forward-looking or action-oriented thought that empowers the user.
4.  **Tone-Matching Emojis:** Use emojis in both the `title` and `summary` to match the tone.
5.  **ID Integrity:** Perfectly preserve the `id` from the input.

---
**Thought Process for Each Insight:**

1.  **Identify Key Points:** First, identify the most important financial points in the raw `combined_insight`.
2.  **Draft a Friendly, Punchy Summary:** Based on the key points, write a draft `summary`. Keep it under 150 characters. It MUST be friendly, crystal clear, and state the financial direction and context for all numbers.
3.  **List Summary Points:** Internally, create a checklist of every distinct financial point you included in the summary.
4.  **Craft & Verify the All-Inclusive Title:** Based on your checklist of summary points, craft a `title` (under 30 chars) that covers ALL of them. **Self-Correction Check:** Read your generated title and ask: "Does this title ignore any key financial points from my summary?" If yes, it's a failure. Regenerate the title. For example, if the summary covers lower food costs and lower transport costs, a title of just "Food Savings!" is incorrect. A correct title would be "Spending Down! 👇" or "Food & Transit Wins! 🏆".
5.  **Final Polish:** Read the `title` and `summary` together. Do they form a cohesive, delightful, and empowering message from a financially savvy best friend? Is the snapshot crystal clear?

---
**Examples of Title Quality:**

*   **GOOD EXAMPLE (Thematic):**
    *   `summary`: "Heads up! Your food spending was up to $500, but your side hustle brought in an extra $300! 💸"
    *   `title`: "Money Moves! 💃"
    *   *Reasoning:* Thematic title covers both a negative (spending up) and a positive (income up) event under one energetic theme.

*   **GOOD EXAMPLE (Explicit):**
    *   `summary`: "Great job! Your grocery bill is down to $250 and your gas spending is down to $100 this month. 👏"
    *   `title`: "Groceries & Gas Down! 👇"
    *   *Reasoning:* Explicitly names both categories that are moving in the same direction.

*   **BAD EXAMPLE (Incomplete):**
    *   `summary`: "Your shopping was high at $400, but you saved $50 on utilities! 👍"
    *   `title`: "High Shopping Bill! 🛒"
    *   *Reasoning:* This is a failure. The title completely ignores the positive news about utilities, giving an incomplete picture. The title MUST cover all key points. A better title would be "Spending Snapshot 📸".

---
**Output Format & Rules:**
*   **Strict JSON Array:** Output must be a single, valid JSON array.
*   **`title`:** Under 30 characters. Must be catchy and holistic.
*   **`summary`:** Single, concise line, ideally under 150 characters. No robotic greetings.
*   **Numbers:** Format as currency.
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
