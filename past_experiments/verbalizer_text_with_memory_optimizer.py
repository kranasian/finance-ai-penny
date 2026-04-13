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
-   **Tone:**
    -   Be positive, supportive, and engaging.
    -   **Apologetic & Actionable:** If a request cannot be fulfilled or there are mistakes, be apologetic and offer actionable alternatives or thank them for feedback.
-   **Style:** Write in a concise, SMS-style format. Avoid repeating information to keep messages brief. Aim for 2-3 short sentences maximum.
-   **Formatting:**
    -   **No Greetings:** Do NOT use greetings like "Hi", "Hey", "Good morning", etc. Start directly with the content.
    -   **Name Usage:** Only address the user by name if it is explicitly present in `user_memories` or `past_conversations`. Otherwise, do not use a name.
    -   **Emojis:** Use emojis to enhance warmth and readability (e.g., 💰, 📉, ✨), but **never** use unicode format (e.g., U+1F4B0).
    -   **Monetary Values:** Use commas and no decimals (e.g., `$15,000`).
    -   **Categories:** Remove underscores from category names (e.g., "Food_and_Drink" -> "Food and Drink").
    -   **No Markdown:** Output must be plain text. Do NOT use markdown formatting (bold, italics, lists, etc.).
-   **Completeness:** Include all pertinent information from the `answer` while remaining concise.
-   **Accuracy:** Do not hallucinate numbers or facts. If the `answer` asks a question, you must ask that question to the user.

#### 5. Critical Constraints
-   **Use Input Data Only:** Do not invent information not found in `answer` or `user_memories`.
-   **Avoid Redundancy:** Do not repeat information already in `past_conversations`.
-   **No External/Internal Info:** Do not use outside info or internal IDs (e.g., transaction IDs).

#### 6. Contextual Information
-   **Date:** Today is `Tuesday, September 30, 2025`.

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
output: Last week you spent a bit more on groceries and eating out compared to the week before. $8,881 last week vs $881 on groceries, and $8,888 last week vs $882 on eating out. 💸 In the next trip to Sprouts maybe we can find some wiggle room there? 🤔

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
      "message": "You had an unexpected $881 spend on kids shopping this month. 😲" },
    { "speaker": "Penny",
      "message": "You've spent $882 unexpectedly on donations and gifts this month, above the $883 forecast." },
    { "speaker": "User",
      "message": "What are my account balances and which is do I owe the most?" }
  ],
  "answer": "Okay, here's a summary of your account balances:\n\n**Depository accounts:**\n\n* Checking: $8,881.08\n* Checking: $884.08\n* Checking: $883.08\n* Money Market Plus: $8,882.08\n\n**Credit accounts:**\n\n* Cashback: $8,885.08 owed, $88,882.08 available\n* Costco Anywhere: $886 owed, $8,887.08 available\n* Citi Premier: $887 owed, $8,888.08 available\n\nThe account with the largest outstanding balance that you owe is the **Cashback card with $8,885.08**."
}
output: You owe the most on the Cashback card with a balance of $8,885! 💸 Largest checking and savings account being your Money Market Plus shining bright at $8,882, and your Checking account has $884. 👍 Perhaps we could explore moving a little from your investments to cover that balance?
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
    self.thinking_budget = 700
    self.model_name = model_name
    
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
      top_k=self.top_k,
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

    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=generate_content_config,
    ):
      if chunk.text is not None:
        output_text += chunk.text
      
      # Extract thought summary from chunk
      if hasattr(chunk, 'candidates') and chunk.candidates:
        for candidate in chunk.candidates:
          if hasattr(candidate, 'content') and candidate.content:
            if hasattr(candidate.content, 'parts') and candidate.content.parts:
              for part in candidate.content.parts:
                if hasattr(part, 'thought') and part.thought:
                  if hasattr(part, 'text') and part.text:
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


# --- BATCH 1: Spending Analysis & Insights ---
def run_batch_1(verbalizer: VerbalizerTextWithMemory):
    print("\n--- Running Batch 1: Spending Analysis & Insights ---")
    
    # Test 1.1: High Spending Alert (No Name)
    print("\nTest 1.1: High Spending Alert (No Name)")
    test_with_inputs({
        "user_memories": ["User loves concerts."],
        "past_conversations": [{"speaker": "User", "message": "How much did I spend on entertainment?"}],
        "answer": "You spent $800 on Entertainment last month. This is $300 higher than your average."
    }, verbalizer)

    # Test 1.2: Subscription Check (With Name)
    print("\nTest 1.2: Subscription Check (With Name)")
    test_with_inputs({
        "user_memories": ["User's preferred name is Sarah.", "User hates unused subscriptions."],
        "past_conversations": [{"speaker": "User", "message": "Check for recurring charges."}],
        "answer": "Found a recurring charge of $12.99 for 'Digital_Magazine_Sub' on the 1st."
    }, verbalizer)

    # Test 1.3: Income Update (No Name)
    print("\nTest 1.3: Income Update (No Name)")
    test_with_inputs({
        "user_memories": ["User is a consultant."],
        "past_conversations": [{"speaker": "User", "message": "Did Client Y pay?"}],
        "answer": "Yes, a deposit of $4,000 from 'Client Y Corp' was received today."
    }, verbalizer)


# --- BATCH 2: Budgeting & Goals ---
def run_batch_2(verbalizer: VerbalizerTextWithMemory):
    print("\n--- Running Batch 2: Budgeting & Goals ---")
    
    # Test 2.1: Budget Overflow (With Name)
    print("\nTest 2.1: Budget Overflow (With Name)")
    test_with_inputs({
        "user_memories": ["User's preferred name is Mike.", "User is saving for a car."],
        "past_conversations": [{"speaker": "User", "message": "How is my dining budget?"}],
        "answer": "You have spent $600 on Food_and_Dining this month. Your budget is $450. You are over by $150."
    }, verbalizer)

    # Test 2.2: Goal Progress (No Name)
    print("\nTest 2.2: Goal Progress (No Name)")
    test_with_inputs({
        "user_memories": ["User calls their savings 'Freedom Fund'."],
        "past_conversations": [{"speaker": "User", "message": "Status of my freedom fund?"}],
        "answer": "Your 'Freedom Fund' is at $15,000. You are 75% of the way to your $20,000 goal."
    }, verbalizer)

    # Test 2.3: Creating a new budget (No Name)
    print("\nTest 2.3: Creating a New Budget (No Name)")
    test_with_inputs({
        "user_memories": ["User loves gadgets."],
        "past_conversations": [{"speaker": "User", "message": "I need a budget for electronics."}],
        "answer": "I can help set a budget for 'Electronics'. What is your limit?"
    }, verbalizer)


# --- BATCH 3: Merchant & Transaction Details ---
def run_batch_3(verbalizer: VerbalizerTextWithMemory):
    print("\n--- Running Batch 3: Merchant & Transaction Details ---")
    
    # Test 3.1: Specific Merchant Spend (With Name)
    print("\nTest 3.1: Specific Merchant Spend (With Name)")
    test_with_inputs({
        "user_memories": ["User's preferred name is Leo.", "User shops at Amazon."],
        "past_conversations": [{"speaker": "User", "message": "Amazon spend this year?"}],
        "answer": "You have spent $2,500.50 at Amazon year-to-date across 20 transactions."
    }, verbalizer)

    # Test 3.2: Transaction Search (No Name)
    print("\nTest 3.2: Transaction Search (No Name)")
    test_with_inputs({
        "user_memories": ["User went to Paris."],
        "past_conversations": [{"speaker": "User", "message": "Find the hotel charge from Paris."}],
        "answer": "Found a transaction for $1,200 at 'Hotel_Paris' on June 10th."
    }, verbalizer)

    # Test 3.3: Category Breakdown (No Name)
    print("\nTest 3.3: Category Breakdown (No Name)")
    test_with_inputs({
        "user_memories": ["User is into fitness."],
        "past_conversations": [{"speaker": "User", "message": "Health spend lately?"}],
        "answer": "In the last 30 days, you spent $400 on Health_and_Fitness. Top merchants: Gym_Shark ($200) and Whole_Foods ($150)."
    }, verbalizer)


# --- BATCH 4: Complex Reasoning & Apologies ---
def run_batch_4(verbalizer: VerbalizerTextWithMemory):
    print("\n--- Running Batch 4: Complex Reasoning & Apologies ---")
    
    # Test 4.1: Unable to Fulfill Request (With Name)
    print("\nTest 4.1: Unable to Fulfill Request (With Name)")
    test_with_inputs({
        "user_memories": ["User's preferred name is Jen."],
        "past_conversations": [{"speaker": "User", "message": "Predict my stocks for next year."}],
        "answer": "I cannot predict stock market performance. I can only analyze past performance."
    }, verbalizer)

    # Test 4.2: Data Discrepancy (No Name)
    print("\nTest 4.2: Data Discrepancy (No Name)")
    test_with_inputs({
        "user_memories": ["User is detail-oriented."],
        "past_conversations": [
            {"speaker": "User", "message": "You said $200, app says $250."},
            {"speaker": "Penny", "message": "Let me double check that for you."}
        ],
        "answer": "My previous calculation missed a $50 transaction at 'Shell_Gas'. Correct total is $250."
    }, verbalizer)

    # Test 4.3: Affordability (No Name)
    print("\nTest 4.3: Affordability (No Name)")
    test_with_inputs({
        "user_memories": ["User pays rent on the 1st."],
        "past_conversations": [{"speaker": "User", "message": "Can I buy a $500 watch?"}],
        "answer": "Checking balance: $1,000. Rent due: $800. Remaining: $200. You cannot afford the watch."
    }, verbalizer)


def main():
  """Main function to test the VerbalizerText verbalizer"""
  import argparse
  parser = argparse.ArgumentParser(description='Run verbalizer tests in batches')
  parser.add_argument('--batch', type=int, default=1, choices=[1, 2, 3, 4],
                      help='Batch number to run (1-4)')
  args = parser.parse_args()
  
  verbalizer = VerbalizerTextWithMemory()
  
  if args.batch == 1:
      run_batch_1(verbalizer)
  elif args.batch == 2:
      run_batch_2(verbalizer)
  elif args.batch == 3:
      run_batch_3(verbalizer)
  elif args.batch == 4:
      run_batch_4(verbalizer)

if __name__ == "__main__":
  main()
