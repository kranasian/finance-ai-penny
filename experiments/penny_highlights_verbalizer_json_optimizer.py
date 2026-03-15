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
    *   **Length Constraint:** Keep it under 30 characters, but prioritize coverage over brevity.
    *   **Holistic Requirement:** If a summary mentions a specific amount and a specific category, the title should ideally reflect both or the overall impact.
    *   **Critical Failure:** A title covering only one point when the summary has multiple is a failure. For example, if the summary mentions lower food costs and lower transport costs, a title of just "Food Savings!" is incorrect. A correct title would be "Spending Down! 👇" or "Food & Transit Wins! 🏆".
2.  **Crystal Clear & Punchy Summaries:** Every number MUST be explained clearly. You MUST state the direction (up/down) of spending or income. Keep the summary punchy and to the point.
3.  **Monetary Precision:** Round all monetary values to the nearest dollar. NEVER use abbreviations like "$11.3k". Use "$11,300" instead. (e.g., $11,300.06 becomes $11,300; $11,300.50 becomes $11,301).
    *   **Calculation Rule:** If the input says "spent $750.60 which is $250.60 over budget", the summary should say "spent $751, which is $251 over budget".
    *   **Consistency:** Use the same rounded value in both the title (if applicable) and the summary.
4.  **Category Integrity:** "Miscellaneous" and "Uncategorized" are distinct categories. Do NOT use them interchangeably.
    *   **Uncategorized:** Transactions that haven't been assigned a category yet.
    *   **Miscellaneous:** A specific category for various small or unusual expenses.
    *   **Strictness:** If the input says "Miscellaneous", use "Miscellaneous" or "Misc". If it says "Uncategorized", use "Uncategorized". Never swap them.
5.  **Action-Oriented & Proactive:** When appropriate, end the `summary` with a brief, forward-looking or action-oriented thought that empowers the user.
6.  **Tone-Matching Emojis:** Use emojis in both the `title` and `summary` to match the tone.
7.  **ID Integrity:** Perfectly preserve the `id` from the input.
8.  **No Robotic Fillers:** Avoid starting summaries with "Heads up!" or "Warning!" unless the input explicitly uses them or it's a critical alert. Start with the insight directly.
9.  **Summary Content:** Ensure the summary includes the most relevant details from the `combined_insight`. If the input mentions a specific reason for a change (e.g., "gym membership"), include it.
10. **Title Fun & Variety:** Use a variety of catchy titles. Avoid repeating the same title structure for different insights. Make them feel personal and energetic.
11. **Rounding Consistency:** If an amount is $299.00, it stays $299. If it's $299.50, it becomes $300. Always round to the nearest whole dollar.
12. **SMS Style:** Keep the language informal, like a text from a friend. Use "you" and "your". Use contractions (e.g., "you've", "don't").

---
**Thought Process for Each Insight:**

1.  **Identify Key Points:** First, identify the most important financial points in the raw `combined_insight`.
2.  **Draft a Friendly, Punchy Summary:** Based on the key points, write a draft `summary`. Keep it under 150 characters. It MUST be friendly, crystal clear, and state the financial direction and context for all numbers. Ensure all amounts are rounded to the nearest dollar.
3.  **List Summary Points:** Internally, create a checklist of every distinct financial point you included in the summary.
4.  **Craft & Verify the All-Inclusive Title:** Based on your checklist of summary points, craft a `title` (under 30 chars) that covers ALL of them. **Self-Correction Check:** Read your generated title and ask: "Does this title ignore any key financial points from my summary?" If yes, it's a failure. Regenerate the title.
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
*   **`title`:** Under 30 characters. Must be catchy and holistic. NO newlines or extra whitespace.
*   **`summary`:** Single, concise line, ideally under 150 characters. No robotic greetings. NO newlines or extra whitespace.
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
    self.thinking_budget = 0
    self.model_name = "gemini-flash-lite-latest"
    
    # Generation Configuration Constants
    self.temperature = 0.5
    self.top_p = 0.95
    self.top_k = 40
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
      top_k=self.top_k,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=self.system_prompt)],
      response_mime_type="application/json",
      response_schema=self.output_schema,
    )

    # Generate response
    response = self.client.models.generate_content(
      model=self.model_name,
      contents=contents,
      config=generate_content_config,
    )
    
    output_text = response.text
    
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
      "combined_insight": "Your shelter costs are way down this month to $1,248.12, mainly from less on home stuff, utilities, and upkeep. 🥳🏠 Oh em gee! You got a huge surprise income boost of $8,800.50 this week, mostly from your business, and you're projected to spend only $68.25 by the end of the week! Way to go, you savvy boss babe!"
    },
    {
      "id": 2,
      "combined_insight": "Looks like you spent less on food this month, down to $1,007.89, mostly from less eating out, deliveries, and groceries. 🍽️🚚 Your transport costs are way down this month to just $46.40, mostly 'cause you took public transit less. 🚇"
    }
  ]
  return run_test_with_insights(insights, verbalizer)

def test_2(verbalizer: PennyHighlightsVerbalizerOptimizer = None):
  """
  Test case 2: Overspending on shopping and miscellaneous expenses.
  """
  insights = [
    {
      "id": 3,
      "combined_insight": "Warning! 🚨 You've spent $750.60 on shopping this month, which is $250.60 over your budget. Most of it went to online stores."
    },
    {
      "id": 4,
      "combined_insight": "Heads up! You have $500.25 in miscellaneous expenses this week. Let's keep an eye on those! 🧐"
    }
  ]
  return run_test_with_insights(insights, verbalizer)

def test_3(verbalizer: PennyHighlightsVerbalizerOptimizer = None):
  """
  Test case 3: Uncategorized transactions and health spending.
  """
  insights = [
    {
      "id": 5,
      "combined_insight": "You have $320.75 in uncategorized transactions. Let's get those sorted! 📂"
    },
    {
      "id": 6,
      "combined_insight": "Your health & wellness spending is up to $215.40 this month, mainly from that new gym membership and some vitamins. 🧘‍♀️"
    }
  ]
  return run_test_with_insights(insights, verbalizer)

def test_4(verbalizer: PennyHighlightsVerbalizerOptimizer = None):
  """
  Test case 4: Side hustle income and utility savings.
  """
  insights = [
    {
      "id": 7,
      "combined_insight": "Amazing! Your side hustle brought in an extra $1,200.99 this month! Keep up the great work! 🚀💰"
    },
    {
      "id": 8,
      "combined_insight": "Great job on cutting down costs! Your electricity bill was only $55.30 this month, down from $85.10 last month. 💡"
    }
  ]
  return run_test_with_insights(insights, verbalizer)

def test_5(verbalizer: PennyHighlightsVerbalizerOptimizer = None):
  """
  Test case 5: Travel expenses and dining out.
  """
  insights = [
    {
      "id": 9,
      "combined_insight": "Pack your bags! ✈️ You spent $1,450.80 on travel this month. Hope it was a blast!"
    },
    {
      "id": 10,
      "combined_insight": "Yum! 🍕 Your dining out spending reached $425.65 this month. That's a lot of tasty treats!"
    }
  ]
  return run_test_with_insights(insights, verbalizer)

def test_6(verbalizer: PennyHighlightsVerbalizerOptimizer = None):
  """
  Test case 6: Subscription audit and entertainment.
  """
  insights = [
    {
      "id": 11,
      "combined_insight": "Time for a subscription audit? 🕵️‍♂️ You spent $89.95 on various streaming services this month."
    },
    {
      "id": 12,
      "combined_insight": "Let's have some fun! 🍿 You spent $120.40 on entertainment, including those movie tickets and the concert."
    }
  ]
  return run_test_with_insights(insights, verbalizer)

def test_7(verbalizer: PennyHighlightsVerbalizerOptimizer = None):
  """
  Test case 7: Education and personal growth.
  """
  insights = [
    {
      "id": 13,
      "combined_insight": "Investing in yourself! 📚 You spent $299.00 on that online course this month. Knowledge is power!"
    },
    {
      "id": 14,
      "combined_insight": "Bookworm alert! 📖 Your book purchases totaled $45.20 this month. Enjoy the reads!"
    }
  ]
  return run_test_with_insights(insights, verbalizer)

def test_8(verbalizer: PennyHighlightsVerbalizerOptimizer = None):
  """
  Test case 8: Pet care and home improvement.
  """
  insights = [
    {
      "id": 15,
      "combined_insight": "Paws-itive vibes! 🐾 You spent $150.75 on pet supplies and treats this month. Your furry friend says thanks!"
    },
    {
      "id": 16,
      "combined_insight": "Home sweet home! 🏡 You spent $340.50 on some new decor and small repairs. Looking good!"
    }
  ]
  return run_test_with_insights(insights, verbalizer)

def test_9(verbalizer: PennyHighlightsVerbalizerOptimizer = None):
  """
  Test case 9: Gifts and donations.
  """
  insights = [
    {
      "id": 17,
      "combined_insight": "So generous! 🎁 You spent $210.30 on gifts for friends and family this month."
    },
    {
      "id": 18,
      "combined_insight": "Making a difference! ❤️ You donated $100.00 to your favorite charity this month. Way to go!"
    }
  ]
  return run_test_with_insights(insights, verbalizer)

def test_10(verbalizer: PennyHighlightsVerbalizerOptimizer = None):
  """
  Test case 10: Insurance and taxes.
  """
  insights = [
    {
      "id": 19,
      "combined_insight": "Stay protected! 🛡️ Your insurance premiums totaled $185.20 this month."
    },
    {
      "id": 20,
      "combined_insight": "Tax time! 📝 You paid $1,200.45 in estimated taxes this quarter. Stay ahead of the game!"
    }
  ]
  return run_test_with_insights(insights, verbalizer)

def test_11(verbalizer: PennyHighlightsVerbalizerOptimizer = None):
  """
  Test case 11: Hobbies and electronics.
  """
  insights = [
    {
      "id": 21,
      "combined_insight": "Level up! 🎮 You spent $350.90 on that new gaming console and some games."
    },
    {
      "id": 22,
      "combined_insight": "Tech savvy! 💻 You spent $125.60 on some new computer accessories this month."
    }
  ]
  return run_test_with_insights(insights, verbalizer)

def test_12(verbalizer: PennyHighlightsVerbalizerOptimizer = None):
  """
  Test case 12: Beauty and personal care.
  """
  insights = [
    {
      "id": 23,
      "combined_insight": "Treat yourself! 💅 You spent $145.30 on beauty treatments and skincare this month."
    },
    {
      "id": 24,
      "combined_insight": "Looking sharp! 💇‍♂️ Your haircut and grooming products cost $65.00 this month."
    }
  ]
  return run_test_with_insights(insights, verbalizer)

def main(batch: int = 1):
  """
  Main function to test the PennyHighlightsVerbalizerOptimizer
  
  Args:
    batch: Batch number (1, 2, 3, or 4) to determine which tests to run
  """
  print(f"Testing PennyHighlightsVerbalizerOptimizer - Batch {batch}\n")
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
    elif batch == 3:
      print("--- Running Batch 3: Test 7 ---")
      test_7(verbalizer)
      print("\n--- Running Batch 3: Test 8 ---")
      test_8(verbalizer)
      print("\n--- Running Batch 3: Test 9 ---")
      test_9(verbalizer)
    elif batch == 4:
      print("--- Running Batch 4: Test 10 ---")
      test_10(verbalizer)
      print("\n--- Running Batch 4: Test 11 ---")
      test_11(verbalizer)
      print("\n--- Running Batch 4: Test 12 ---")
      test_12(verbalizer)
    
    print("\nAll tests completed!")
  except Exception as e:
    print(f"An error occurred: {e}")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run verbalizer tests in batches')
  parser.add_argument('--batch', type=int, default=1, choices=[1, 2, 3, 4],
                      help='Batch number to run (1, 2, 3, or 4)')
  args = parser.parse_args()
  main(batch=args.batch)
