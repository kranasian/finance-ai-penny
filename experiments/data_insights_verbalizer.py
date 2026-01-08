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

SYSTEM_PROMPT = """**Objective:** Synthesize a list of financial insights into two distinct formats: a concise, friendly SMS from "Penny" and a detailed, bulleted markdown list for an email.

**Persona: Penny**
You are Penny, the user's personal AI financial consultant and close friend. Your tone is celebratory, encouraging, and knowledgeable, balancing friendly support with professional clarity. You are brief and use emojis to add warmth.

---
**Key Directives:**

1.  **Synthesize, Don't Just List:** The `penny_variations` should be a holistic summary of all input insights, not just a concatenation. Find the common theme or the most important news to highlight.
2.  **Two Formats, Two Purposes:**
    *   `penny_variations`: These are for a quick, engaging SMS. They should be short, friendly, and get the main point across in Penny's voice. Provide 2-3 distinct variations.
    *   `detailed_items`: This is for a more formal email summary. Each item should be a clear, informative bullet point that expands on one of the original insights. This should be markdown-ready.
3.  **Preserve Key Information:** Ensure all important details from the input (e.g., amounts, categories, directions like "up" or "down") are accurately reflected in both output formats.
4.  **Tone Adaptation:** The tone for `penny_variations` is more casual and emoji-heavy (Penny's persona). The tone for `detailed_items` is slightly more formal but still encouraging and clear.

---
**Thought Process for Generation:**

1.  **Analyze Inputs:** Read all the provided insight messages. What is the overall financial story? Is it about saving, earning, overspending, or a mix?
2.  **Craft `penny_variations`:**
    *   Identify the most impactful news.
    *   Draft a short, engaging summary that captures this news.
    *   Use Penny's voice: friendly, supportive, with emojis.
    *   Create 1-2 alternative phrasings of the same summary.
3.  **Craft `detailed_items`:**
    *   Go through each input insight one by one.
    *   Rewrite each insight as a clear, concise bullet point.
    *   Ensure it's suitable for a markdown list in an email.
    *   Start each item with a relevant emoji.

---
**Output Format & Rules:**

*   **Strict JSON Object:** Output must be a single, valid JSON object matching the schema.
*   **`penny_variations`:** An array of 2-3 strings. Each string is a short, SMS-style message.
*   **`detailed_items`:** An array of strings. Each string is a bullet point for a markdown list.
*   **Numbers:** Format as currency with commas, no decimals (e.g., $1,234).
*   **"uncategorized":** Preserve this term if it appears in the input.
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
    "Your shelter costs are way down this month to $1,248, mainly from less on home stuff, utilities, and upkeep. 🥳🏠",
    "You got a huge surprise income boost of $8,800 this week, mostly from your business, and you're projected to spend only $68 by the end of the week! Way to go, you savvy boss babe!",
    "Looks like you spent less on food this month, down to $1,007, mostly from less eating out, deliveries, and groceries. 🍽️🚚🛒"
  ]
  return run_test_with_messages(messages, verbalizer)

def test_2(verbalizer: DataInsightsVerbalizer = None):
  """
  Test case 2: Bad News (Overspending)
  """
  messages = [
    "Warning! 🚨 You've spent $750 on shopping this month, which is $250 over your budget. Most of it went to online stores.",
    "Heads up! You have $500 in uncategorized expenses this week. Let's categorize them to keep your budget on track! 🧐",
    "Your credit card balance is at $3,200 this month. Let's make a plan to pay it down! 💳"
  ]
  return run_test_with_messages(messages, verbalizer)

def test_3(verbalizer: DataInsightsVerbalizer = None):
  """
  Test case 3: Mixed News (Goals & Unexpected Costs)
  """
  messages = [
    "You're so close! You've saved $9,500 for your vacation, that's 95% of your $10,000 goal! 🌴☀️",
    "Just a heads-up, you had a small unexpected charge of $35 for a subscription service you might have forgotten about. 😬",
    "To the moon! 🚀 Your investment portfolio is up 15% this quarter, adding a nice $4,500 to your net worth."
  ]
  return run_test_with_messages(messages, verbalizer)


def main(test_case: int = 1):
  """
  Main function to test the DataInsightsVerbalizer
  
  Args:
    test_case: The test case number to run (1, 2, or 3)
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
    
    print("\nTest completed!")
  except Exception as e:
    print(f"An error occurred: {e}")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run verbalizer tests')
  parser.add_argument('--test', type=int, default=1, choices=[1, 2, 3],
                      help='Test case number to run (1, 2, or 3)')
  args = parser.parse_args()
  main(test_case=args.test)

