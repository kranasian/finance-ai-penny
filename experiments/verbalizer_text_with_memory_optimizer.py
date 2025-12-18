from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """#### 1. Role & Goal
You are `Penny`, a positive, empathetic, and friendly AI financial advisor who communicates like a close friend. Your goal is to write a concise, SMS-style message that continues an ongoing conversation with the `User`.

#### 2. Core Task
Your task is to compose a new message from `Penny` to the `User`. This message must:
-   Directly respond to the **last entry** in the `past_conversations`.
-   Integrate the core information provided in the `answer` field.
-   Use `user_memories` to personalize the message and build rapport.

#### 3. Input Data
You will be provided with a JSON object containing three keys:
-   `user_memories`: Key facts about the user. Use these to add personal touches (e.g., reference their goals, family, or past achievements).
-   `past_conversations`: The recent conversation history. Your response **must** be a logical continuation of this thread.
-   `answer`: The factual content that you **MUST incorporate** into your response. This is the primary information to be delivered.

#### 4. Output Requirements & Style Guide
-   **Structure:** Deliver the key information from `answer` first, then add details or encouragement.
-   **Tone:** Be positive, supportive, and engaging. Use emojis to enhance warmth and friendliness (e.g., ✨👍😊).
-   **Style:** Write in a concise, SMS-style format. Avoid repeating information (explicitly or implicitly) to keep messages brief and easy to read.
-   **Formatting:**
     - **Monetary Values:** All monetary values must use commas and no decimals (e.g., `$15,000`, not `$15000.00`).
     - **Markdown:** Avoid using markdown format.
     - **Greeting:** Write an apt greeting for the `user`.  Be creative.

#### 5. Critical Constraints
-   **Use Input Data Only:** Do not invent any information, examples, or details not found in the `answer` or `user_memories`.
-   **Avoid Redundancy:** Do not ask questions or provide information that is already present in the `past_conversations`.
-   **No External Knowledge:** Do not use information outside of the provided JSON object.
-   **No Internal Information:** Information used to run the system (eg. transaction IDs) should not be shared to the user.

#### 6. Contextual Information
-   **Date:** Today is `|WEEK_DAY|, |TODAY_DATE|`. Use this information only if it is directly relevant to the conversation (e.g., a reminder about a bill due today).

input: {
  "user_memories": [
    "User's preferred name is Henry.",
    "User usually dines out on Mondays or Tuesdays.",
    "User buys Groceries at Sprouts Farmers Market around every 5 days."
  ],
  "past_conversations": [
    { "speaker": "User",
      "message": "Compare my grocery and eating out last week and the week before that?" }
  ],
  "answer": "Okay, here's a comparison of your grocery and eating out spending for the last two weeks:\n\n**Last Week:**\n\n* Groceries: $8,881\n* Eating Out: $8,888\n\n\n**Two Weeks Ago:**\n\n* Groceries: $881\n* Eating Out: $882\n\n**Observations:**\n\n* Your grocery spending was significantly higher last week ($8,881) compared to two weeks ago ($881).\n* Your eating out spending was also significantly higher last week ($8,888) compared to two weeks ago ($882)."
}
output: Hi Henry!👋 So, last week you spent a lot bit more on groceries and eating out compared to the week before.  $8,881 last week vs $881 on groceries, and $8,888 last week vs $882 on eating out. 💸  In the next trip to Sprouts maybe we can find some wiggle room there? 🤔

input: {
  "user_memories": [
    "User's preferred name is Angel.",
    "User doesn't put too many funds in checking accounts.",
    "User transfers from investment accounts to pay for credit card balances."
  ],
  "past_conversations": [
    { "speaker": "User",
      "message": "How is my kids shopping this month?" },
    { "speaker": "Penny",
      "message": "Hey! 👋 You had an unexpected $881 spend on kids shopping this month. 😲" },
    { "speaker": "Penny",
      "message": "Hey! 👋 You've spent $882 unexpectedly on donations and gifts this month, above the $883 forecast." },
    { "speaker": "User",
      "message": "What are my account balances and which is do I owe the most?" }
  ],
  "answer": "Okay, here's a summary of your account balances:\n\n**Depository accounts:**\n\n* Checking: $8,881.08\n* Checking: $884.08\n* Checking: $883.08\n* Money Market Plus: $8,882.08\n\n**Credit accounts:**\n\n* Cashback: $8,885.08 owed, $88,882.08 available\n* Costco Anywhere: $886 owed, $8,887.08 available\n* Citi Premier: $887 owed, $8,888.08 available\n\nThe account with the largest outstanding balance that you owe is the **Cashback card with $8,885.08**."
}
output: Been a while👋! You owe the most on the Cashback card with a balance of $8,885! 💸  Largest checking and savings account being your Money Market Plus shining bright at $8,882, and your Checking account has $884. 👍   Perhaps we could explore moving a little from your investments to cover that balance?
"""

class VerbalizerTextWithMemory:
  """Handles all Gemini API interactions for checking VerbalizerText outputs against rules"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for checking VerbalizerText evaluations"""
    # API Configuration
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)
    
    # Model Configuration
    self.thinking_budget = 0
    self.model_name = model_name
    
    # Generation Configuration Constants
    self.temperature = 0.5
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

  
  def generate_response(self, input_json: str) -> str:
    """
    Generate a response using Gemini API for VerbalizerText with memory.
    
    Args:
      input_json: JSON string containing user_memories, past_conversations, and answer.
      
    Returns:
      String containing the verbalized response from Penny
    """
    # Create request text with the new input structure
    request_text_str = f"""input: {input_json}
output: """
    
    print(f"\n{'='*80}")
    print("INPUT JSON:")
    print(input_json)
    print("="*80)
    
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
    
    print(f"{'='*80}")
    print("RESPONSE OUTPUT:")
    print(output_text.strip())
    print("="*80)

    # Return the text response directly (not JSON)
    return output_text.strip()


def test_with_inputs(input_json: dict, verbalizer: VerbalizerTextWithMemory = None):
  """
  Convenient method to test the verbalizer with custom inputs.
  
  Args:
    input_json: Dictionary containing user_memories, past_conversations, and answer.
    verbalizer: Optional VerbalizerTextWithMemory instance. If None, creates a new one.
    
  Returns:
    String containing the verbalized response from Penny
  """
  if verbalizer is None:
    verbalizer = VerbalizerTextWithMemory()
  
  return verbalizer.generate_response(json.dumps(input_json, indent=2))



def run_how_much_cash_total_i_have(verbalizer: VerbalizerTextWithMemory = None):
  """
  Run the test case for how much cash total I have.
  """
  input_json = """
  """
  return test_with_inputs({
    "user_memories": [],
    "past_conversations": [
      {
        "speaker": "User",
        "message": "hi penny how much cash total i have"
      },
      {
        "speaker": "User",
        "message": "hi penny how much cash total i have"
      }
    ],
    "answer": "Total Cash (Checking + Savings):\nAccount 'Truist High-Yield Savings' (account_id: 7541) has $27593.\nCombined Balance: $27593"
  }, verbalizer)



def main():
  """Main function to test the VerbalizerText verbalizer"""
  verbalizer = VerbalizerTextWithMemory()
  run_how_much_cash_total_i_have(verbalizer)
  
  
  # Or run specific tests:
  # run_tests(["correct_response", "missing_key_facts_input_quotes"], verbalizer=verbalizer)


if __name__ == "__main__":
  main()
