from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the output schema for the verbalizer
SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "penny_variations": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
            description="2-3 concise, fun, and informative SMS-style message variations from Penny, with emojis."
        ),
        "detailed_items": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
            description="A bulleted list of the key insights in a more detailed, markdown-ready format for an email."
        )
    },
    required=["penny_variations", "detailed_items"]
)

SYSTEM_PROMPT = """**Objective:** Synthesize a list of financial insights into two distinct formats: a concise, friendly SMS from "Penny" and a bulleted markdown list that matches the same light, friendly, and casual tone.

**Persona: Penny**
You are Penny, the user's personal AI financial consultant and close friend. Your tone is celebratory, encouraging, and knowledgeable, balancing friendly support with professional clarity. You are brief and use emojis to add warmth.

---
**Key Directives:**

1.  **Complete Information Inclusion:** 
    *   **CRITICAL:** Every `penny_variation` MUST include ALL key information from ALL input insights. Include all amounts, categories, percentages, timeframes, and specific reasons/details mentioned.
    *   Synthesize insights cohesively while ensuring every important detail is present. Use abbreviations and concise phrasing to fit SMS length while maintaining completeness.
2.  **Use "We" for Penny's Actions:**
    *   **CRITICAL:** When sharing updates about actions that Penny (the AI) has taken (e.g., categorizing transactions, analyzing data, organizing information), use "we" or "I" to show it was Penny's work.
    *   When referring to the user's achievements or actions, "you" is appropriate.
3.  **Preserve Transaction Names:**
    *   **CRITICAL:** When input insights mention specific transaction names, merchant names, or account names, you MUST preserve them EXACTLY as written. Do not abbreviate, shorten, or edit transaction names.
4.  **Action-Oriented Messaging:**
    *   Both `penny_variations` and `detailed_items` should encourage users to take applicable actions when contextually appropriate.
    *   **CRITICAL:** If an input insight explicitly mentions an action, you MUST include that EXACT action in the output. Do not substitute it with a different action.
    *   Only include action items that naturally flow from the insight. Don't force actions where they don't make sense.
5.  **Two Formats, Two Purposes:**
    *   `penny_variations`: These are for a quick, engaging SMS. They should be short, friendly, and get ALL the main points across in Penny's voice. Provide 2-3 distinct variations that each contain the complete picture. **CRITICAL:** Every variation MUST touch on ALL input messages - no message should be omitted. **CRITICAL:** Do NOT use conversational openers like "Hi", "Hey", "Hello", "Good morning" - start directly with the content. Be creative with phrasing while maintaining clarity and completeness.
    *   `detailed_items`: These should match the light, friendly, and concise tone of `penny_variations`. **CRITICAL:** The number of `detailed_items` must be LESS THAN OR EQUAL TO the number of input messages. **CRITICAL - Mutual Exclusivity:** Each bullet must represent ONE individual topic that is completely separate and mutually exclusive from all other bullets. No two bullets should cover the same topic, share content, or overlap in subject matter. If multiple input messages are about the same topic, combine them into a single comprehensive item. Each item should be as casual, friendly, and brief as the SMS messages - use the same conversational tone, emojis, and energy. Keep it super concise (1-2 sentences max). Start each item with a relevant emoji and clear title/summary in bold followed by the details. **CRITICAL:** Do NOT use conversational openers like "Hi", "Hey", "Hello", "Good morning" - start directly with the emoji and title. Each item must include both the update(s) AND the user's next steps/actions if applicable.
6.  **Preserve ALL Information:** Ensure EVERY detail from the input (amounts, categories, directions like "up" or "down", specific reasons, percentages, counts, timeframes) is accurately reflected in both output formats. Nothing should be omitted.
7.  **Tone Consistency:** Both `penny_variations` and `detailed_items` should maintain a light, friendly, and encouraging tone at all times, even when delivering warnings or bad news. Frame challenges as opportunities and use supportive language. Keep transitions between topics smooth and conversational. Be creative with your phrasing while maintaining the guidelines. **CRITICAL:** Never use conversational openers like "Hi", "Hey", "Hello", "Good morning" in any output - start directly with the content.

---
**Thought Process for Generation:**

1.  **Analyze Inputs:** Read all the provided insight messages carefully. Extract EVERY piece of information: amounts, categories, percentages, timeframes, specific reasons, transaction names, and any suggested actions.
2.  **Craft `penny_variations`:**
    *   Identify ALL insights and their complete details.
    *   Draft a short, engaging summary that includes ALL key information from ALL inputs.
    *   **CRITICAL:** Do NOT start with conversational openers like "Hi", "Hey", "Hello", "Good morning" - begin directly with the content.
    *   Use Penny's voice: friendly, supportive, with emojis.
    *   Use "we" or "I" when referring to actions Penny took.
    *   Include any applicable action items naturally in the message.
    *   Create 2-3 alternative phrasings, each containing the complete information set.
    *   Be creative with your wording while ensuring all information is included.
3.  **Craft `detailed_items`:**
    *   **CRITICAL:** The number of `detailed_items` must be LESS THAN OR EQUAL TO the number of input messages. Analyze the input messages to identify which ones are about the same topic, action, or category. If multiple messages address the same general topic (e.g., both about account linking, both about categorization progress, both about the same financial metric or goal), you MUST combine them into a single comprehensive item that includes ALL details from ALL related messages. Only create separate items for distinctly different topics.
    *   **CRITICAL - Mutual Exclusivity:** Each bullet must represent ONE individual topic that is completely separate and non-overlapping from all other bullets. No bullet should share content, overlap in topic, repeat information, or cover the same subject matter as another bullet. If two bullets would both discuss the same general topic (e.g., both about categorization, both about income, both about spending), they MUST be combined into a single bullet. Each bullet is a distinct, standalone item covering a unique topic with zero overlap.
    *   **CRITICAL - No Conversational Openers:** Do NOT use conversational openers like "Hi", "Hey", "Hello", "Good morning" anywhere in the detailed_items. Start each item directly with the emoji and bold title.
    *   **CRITICAL - Tone Matching:** `detailed_items` must match the EXACT light, friendly, casual, and concise tone of `penny_variations`. Use the same conversational style, energy, friendliness, and brevity. They should feel like slightly longer versions of the SMS messages - same vibe, same energy, same casual language. Avoid any formal or business-like phrasing.
    *   Each bullet point MUST include BOTH:
        - The complete update/insight with ALL details from the relevant input message(s)
        - The user's next step(s) or action(s) - this is REQUIRED for every item and must be DIRECTLY related to that item's update (not a generic follow-up)
    *   Format each item as:
        - **CRITICAL:** Start with a relevant emoji immediately followed by a clear title/summary in bold (format: "emoji **Title:**")
        - First sentence: The update/insight (casual, friendly, 1 sentence - be brief and conversational like penny_variations)
        - Second sentence: The user's next step/action (use smooth, natural transitions, keep it light and friendly)
    *   **CRITICAL - Conciseness & Tone:** Maximum 2 sentences total per item. Be direct, casual, and brief - match the EXACT brevity, energy, and conversational style of penny_variations. Use the same casual language patterns, contractions, and friendly expressions. Avoid lengthy explanations, formal language, extra words, or verbose phrasing. Get to the point quickly with the same playful, energetic vibe as the SMS messages.
    *   For next steps: If an insight explicitly mentions an action, include that EXACT action. For positive news, suggest specific next actions. For issues/warnings, suggest specific corrective actions. Never use generic phrases - always provide concrete, actionable next steps. Keep the language casual and friendly.
    *   Use "we" or "I" when referring to actions Penny took.
    *   Be creative with your phrasing while maintaining clarity and following all guidelines. Match the playful, energetic tone of penny_variations.

---
**Output Format & Rules:**

*   **Strict JSON Object:** Output must be a single, valid JSON object matching the schema.
*   **`penny_variations`:** An array of 2-3 strings. Each string is a short, SMS-style message that includes ALL key information from ALL inputs. **CRITICAL:** Do NOT use conversational openers like "Hi", "Hey", "Hello", "Good morning" - start directly with the content.
*   **`detailed_items`:** An array of strings. Each string is a bullet point for a markdown list. Start with emoji and bold title, then concise details. The count must be less than or equal to the number of input messages. **CRITICAL:** Match the light, friendly, casual, and concise tone of `penny_variations` - they should feel conversational and energetic, not formal. **CRITICAL:** Do NOT use conversational openers like "Hi", "Hey", "Hello", "Good morning" - start directly with the emoji and title. **CRITICAL:** Each bullet must be mutually exclusive - representing one individual topic with no overlap between bullets.
*   **Numbers:** Format as currency with commas, no decimals (e.g., $1,234).
*   **"uncategorized":** Preserve this term if it appears in the input.
*   **Transaction Names:** Preserve transaction names, merchant names, and account names EXACTLY as they appear in the input.
*   **Completeness Check:** Before finalizing, verify that every amount, category, percentage, timeframe, transaction name, and specific detail from the input appears in the output.
"""

class DataInsightsVerbalizer:
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

  
  def generate_response(self, messages: list) -> dict:
    """
    Generate verbalized financial insights using Gemini API.
    
    Args:
      messages: A list of strings, where each string is a financial insight message.
      
    Returns:
      Dictionary containing the verbalized insights in the required format.
    """
    # Create request text
    input_json = messages
    
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


def run_test_with_messages(messages: list, verbalizer: DataInsightsVerbalizer = None):
  """
  Convenient method to test the verbalizer with custom inputs.
  
  Args:
    messages: The list of insight messages to verbalize.
    verbalizer: Optional DataInsightsVerbalizer instance. If None, creates a new one.
    
  Returns:
    Dictionary containing verbalized insights.
  """
  if verbalizer is None:
    verbalizer = DataInsightsVerbalizer()
  
  return verbalizer.generate_response(messages)


def test_1(verbalizer: DataInsightsVerbalizer = None):
  """
  Test case 1: Good News (Savings & Income)
  """
  messages = [
    "You got a huge surprise income boost of $8,800 this week, mostly from your business, and you're projected to spend only $68 by the end of the week! Way to go, you savvy boss babe!",
    "Look at us go! 🚀 I've bumped your categorized transactions from 45% up to 70%. 📈",
    "Done analyzing your recent transactions! I've gone through 289 individual transactions, and confidently categorized 201 of them which is 70%."
  ]
  return run_test_with_messages(messages, verbalizer)

def test_2(verbalizer: DataInsightsVerbalizer = None):
  """
  Test case 2: Bad News (Overspending)
  """
  messages = [
    "Warning! 🚨 You've spent $750 on shopping this month, which is $250 over your budget. Most of it went to online stores.",
    "If you link the bank accounts where you get paid, I can give you amazing insights into your spending. It's the best way to see the full picture! 🌟",
    "I notice you've only linked a few accounts. To get a complete picture of your finances, consider linking all your accounts - checking, savings, credit cards, and loans. This helps me provide better insights! 💳✨"
  ]
  return run_test_with_messages(messages, verbalizer)

def test_3(verbalizer: DataInsightsVerbalizer = None):
  """
  Test case 3: Mixed News (Goals & Unexpected Costs)
  """
  messages = [
    "I want to make sure your budget is as accurate as possible! 🎯 Could you help me verify a few categories? I'm stuck on a couple of spots. 🧐",
    "Great news! 🚀 I've done a fresh sweep of your transactions and narrowed the \"uncertain\" list down to just 5% (that's only 12 items left!). You're so close to a perfectly organized month! ✨",
    "You're so close! You've saved $9,500 for your vacation, that's 95% of your $10,000 goal! 🌴☀️"
  ]
  return run_test_with_messages(messages, verbalizer)

def test_4(verbalizer: DataInsightsVerbalizer = None):
  """
  Test case 4: Transaction Categorization Progress
  """
  messages = [
    "💰 Found income from Freelance Design Payment, will categorize it as such moving forward. ✅",
    "Sorted payments from Home Depot into Home & Garden so your budget stays on track. Let me know if I got that wrong. 🏠✨",
    "I'm still a little stumped on a few, though! 🤔 I didn't want to guess and mess up your flow:"
  ]
  return run_test_with_messages(messages, verbalizer)

def test_5(verbalizer: DataInsightsVerbalizer = None):
  """
  Test case 5: Account Linking & Setup
  """
  messages = [
    "Your shelter costs are way down this month to $1,248, mainly from less on home stuff, utilities, and upkeep. 🥳🏠",
    "Seems you have an unlinked depository account. Let's skip the math! 🕵️‍♂️💰 Link your checking or savings accounts so I can watch for low balances and find you some extra interest! ✨",
    "Looks like you spent less on food this month, down to $1,007, mostly from less eating out, deliveries, and groceries. 🍽️🚚🛒"
  ]
  return run_test_with_messages(messages, verbalizer)

def test_6(verbalizer: DataInsightsVerbalizer = None):
  """
  Test case 6: Categorization Help Needed
  """
  messages = [
    "Heads up! You have $500 in uncategorized expenses this week. Let's categorize them to keep your budget on track! 🧐",
    "Your credit card balance is at $3,200 this month. Let's make a plan to pay it down! 💳",
    "Check out that progress! 🚀 We've slashed your uncategorized transactions from 25% all the way down to 5%. 📉"
  ]
  return run_test_with_messages(messages, verbalizer)

def test_7(verbalizer: DataInsightsVerbalizer = None):
  """
  Test case 7: Mixed Categorization & Account Status
  """
  messages = [
    "I noticed a few of your transactions were miscategorized as transfers. Fixed those for you to keep your dashboard accurate! 🛠️✨",
    "Just a heads-up, you had a small unexpected charge of $35 for a subscription service you might have forgotten about. 😬",
    "To the moon! 🚀 Your investment portfolio is up 15% this quarter, adding a nice $4,500 to your net worth."
  ]
  return run_test_with_messages(messages, verbalizer)


def main(test_case: int = 1):
  """
  Main function to test the DataInsightsVerbalizer
  
  Args:
    test_case: The test case number to run (1, 2, 3, 4, 5, 6, or 7)
  """
  print("Testing DataInsightsVerbalizer\n")
  try:
    verbalizer = DataInsightsVerbalizer()
    
    if test_case == 1:
      print("--- Running Test Case 1: Good News ---")
      test_1(verbalizer)
    elif test_case == 2:
      print("\n--- Running Test Case 2: Bad News ---")
      test_2(verbalizer)
    elif test_case == 3:
      print("\n--- Running Test Case 3: Mixed News ---")
      test_3(verbalizer)
    elif test_case == 4:
      print("\n--- Running Test Case 4: Transaction Categorization Progress ---")
      test_4(verbalizer)
    elif test_case == 5:
      print("\n--- Running Test Case 5: Account Linking & Setup ---")
      test_5(verbalizer)
    elif test_case == 6:
      print("\n--- Running Test Case 6: Categorization Help Needed ---")
      test_6(verbalizer)
    elif test_case == 7:
      print("\n--- Running Test Case 7: Mixed Categorization & Account Status ---")
      test_7(verbalizer)
    
    print("\nTest completed!")
  except Exception as e:
    print(f"An error occurred: {e}")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run verbalizer tests')
  parser.add_argument('--test', type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7],
                      help='Test case number to run (1, 2, 3, 4, 5, 6, or 7)')
  args = parser.parse_args()
  main(test_case=args.test)

