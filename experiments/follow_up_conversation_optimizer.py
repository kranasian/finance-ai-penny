from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SCHEMA = types.Schema(
  type=types.Type.OBJECT,
  properties={
    "reply": types.Schema(
      type=types.Type.STRING,
      description="The appropriate response message text based on the user's last message, insights, and conversation context. This could be a conversation continuation or a conversation ender."),
    "reasoning": types.Schema(
      type=types.Type.STRING,
      description="Brief explanation of why this reply was chosen"),
  },
  required=["reply", "reasoning"]
)

SYSTEM_PROMPT = """#### 1. Role & Goal
You are a financial advisor assistant generating follow-up responses. This template is ONLY for Acknowledgments, Closing, or General Conversational Turns—NOT for new data requests or action requests.

#### 2. Core Task
Generate the most appropriate reply matching the conversation type:
- **Acknowledgments:** User expressed thanks, confirmed understanding, or acknowledged information
- **Closing:** User's question answered, expressed satisfaction, or conversation reached natural conclusion
- **General Conversational Turns:** User made a general comment requiring a conversational response

#### 3. Input Data
- `insights`: Array of NEW financial insights. Each string contains spending patterns, forecasts, categories, or other relevant financial data.
- `last_conversation`: List of 5 messages: `["User: message", "Assistant: message", ...]` showing 2 back-and-forth exchanges plus user's final message (chronological order, oldest first).

#### 4. Output Requirements
Return JSON with:
- `reply`: Response message text (BRIEF, concise, friendly, financial advisor tone—keep it short)
- `reasoning`: Brief explanation of reply choice

**Insight Integration:** You are ENCOURAGED to mention the most appropriate insight if it:
- Relates naturally to the conversation topic
- Adds value without feeling forced
- Enhances the conversation flow
- Has NOT been mentioned in the conversation yet

**Reply Guidelines (Keep Brief):**
- **Acknowledgments:** Warmly acknowledge briefly, optionally share relevant insight
- **Closing:** Gracefully conclude briefly, optionally mention insight before closing
- **General Conversational:** Maintain rapport briefly, optionally relate relevant insight

#### 5. Critical Constraints
- Use only provided insights and conversation history
- Consider full conversation context, not just last message
- Prioritize natural flow—insights should enhance, not disrupt
- Maintain friendly, professional financial advisor tone
- **Keep replies BRIEF**—be concise and to the point
"""

class FollowUpConversationOptimizer:
  """Handles all Gemini API interactions for determining appropriate follow-up conversation responses"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for generating follow-up conversation responses"""
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
    
    # Output Schema
    self.output_schema = SCHEMA
  
  def determine_follow_up_response(self, insights: list, last_conversation) -> dict:
    """
    Determine the most appropriate follow-up response based on insights and conversation history.
    
    Args:
      insights: List of strings, where each string is a NEW financial insight that has NOT been mentioned in the conversation yet.
      last_conversation: List of tuples or strings. If tuples, format: [("User", "..."), ("Assistant", "..."), ...]
        If strings, format: ["User: ...", "Assistant: ...", ...]
        Must contain exactly 2 back-and-forth exchanges plus the user's final message (5 messages total). Messages are in chronological order (oldest first, newest last). The last message is the user's most recent message.
      
    Returns:
      Dictionary containing reply and reasoning
    """
    # Convert to list of strings format: ["User: message", "Assistant: message", ...]
    # Accept list of tuples: [("User", "..."), ("Assistant", "..."), ...]
    if isinstance(last_conversation, list):
      conversation_list = []
      for item in last_conversation:
        if isinstance(item, (list, tuple)) and len(item) == 2:
          speaker, message = item[0], item[1]
          conversation_list.append(f"{speaker}: {message}")
        elif isinstance(item, dict):
          if "speaker" in item and "message" in item:
            speaker, message = item["speaker"], item["message"]
            conversation_list.append(f"{speaker}: {message}")
        elif isinstance(item, str):
          # Already in the correct format
          conversation_list.append(item)
      
      # Create input JSON structure
      input_json = {
        "insights": insights,
        "last_conversation": conversation_list
      }
    else:
      raise ValueError("last_conversation must be a list")
    
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


def test_with_inputs(insights: list, last_conversation, optimizer: FollowUpConversationOptimizer = None):
  """
  Convenient method to test the optimizer with custom inputs.
  
  Args:
    insights: List of strings, where each string is a NEW financial insight that has NOT been mentioned in the conversation yet.
    last_conversation: List of tuples or strings. If tuples, format: [("User", "..."), ("Assistant", "..."), ...]
      If strings, format: ["User: ...", "Assistant: ...", ...]
      Must contain exactly 2 back-and-forth exchanges plus the user's final message (5 messages total). Messages are in chronological order (oldest first, newest last).
    optimizer: Optional FollowUpConversationOptimizer instance. If None, creates a new one.
    
  Returns:
    Dictionary containing reply and reasoning
  """
  if optimizer is None:
    optimizer = FollowUpConversationOptimizer()
  
  return optimizer.determine_follow_up_response(insights, last_conversation)


def run_test_acknowledgment(optimizer: FollowUpConversationOptimizer = None):
  """
  Run a test case where user acknowledges information (Acknowledgments scenario).
  """
  return test_with_inputs(
    insights=[
      "How should I categorize your $2,850 outflow to Property Group LLC? Was it for Clothing?",
      "Your dining out expenses have decreased by 15% compared to last month."
    ],
    last_conversation=[
      ("User", "What are my spending patterns this month?"),
      ("Assistant", "Your shopping spending increased to $1,725 this month, which is higher than usual."),
      ("User", "I see, I did buy some new clothes."),
      ("Assistant", "That makes sense. Would you like me to help you set a budget for shopping next month?"),
      ("User", "Got it, thanks for letting me know!")
    ],
    optimizer=optimizer
  )


def run_test_satisfaction_ender(optimizer: FollowUpConversationOptimizer = None):
  """
  Run a test case where user expresses satisfaction (should end conversation).
  """
  return test_with_inputs(
    insights=[
      "You have 3 uncategorized transactions totaling $450.",
      "Your Tuition and Kids Activities spending is lower than expected at $581 this month! 🥳"
    ],
    last_conversation=[
      ("User", "What are my account balances?"),
      ("Assistant", "Your checking account has $5,000 and your savings has $15,000."),
      ("User", "Perfect, thanks!"),
      ("Assistant", "You're welcome! Is there anything else you'd like to know?"),
      ("User", "Thanks, that's exactly what I needed!")
    ],
    optimizer=optimizer
  )


def run_test_general_conversational_turn(optimizer: FollowUpConversationOptimizer = None):
  """
  Run a test case where user makes a general conversational comment (General Conversational Turn scenario).
  """
  return test_with_inputs(
    insights=[
      "Your subscription costs have increased by $25 this month.",
      "Heads up! Your Entertainment spending increased to $173 this week. 📈"
    ],
    last_conversation=[
      ("User", "How am I doing financially this month?"),
      ("Assistant", "Your food spending increased to $615 this month, mostly from dining out. Overall, you're staying within your budget goals."),
      ("User", "That makes sense, I did go out to eat more often this month."),
      ("Assistant", "Yes, and you're still on track with your overall financial goals."),
      ("User", "Good to know, thanks for the update!")
    ],
    optimizer=optimizer
  )


def run_test_thanks_for_update(optimizer: FollowUpConversationOptimizer = None):
  """
  Run a test case based on real conversation about shelter and transport spending.
  """
  return test_with_inputs(
    insights=[
      "Your Food spending is lower than expected at $161 this month! 🥳",
      "Your Medical and Pharmacy spending is higher than expected at $60 this month. 😥"
    ],
    last_conversation=[
      ("Assistant", "Your shelter costs were $2,850, a touch more on Home expenses. Totally fine, just info for you! ✨"),
      ("Assistant", "Good job! Your transport spending is down a little this month at $324, mostly on car stuff. 🚗💨 Every bit helps!"),
      ("User", "Thanks for the update!")
    ],
    optimizer=optimizer
  )

def run_test_thanks(optimizer: FollowUpConversationOptimizer = None):
  """
  Run a test case based on real conversation about shelter and transport spending.
  """
  return test_with_inputs(
    insights=[
      "Your Food spending is lower than expected at $161 this month! 🥳",
      "Your Medical and Pharmacy spending is higher than expected at $60 this month. 😥"
    ],
    last_conversation=[
      ("Assistant", "Your shelter costs were $2,850, a touch more on Home expenses. Totally fine, just info for you! ✨"),
      ("Assistant", "Good job! Your transport spending is down a little this month at $324, mostly on car stuff. 🚗💨 Every bit helps!"),
      ("User", "Thanks!")
    ],
    optimizer=optimizer
  )


def run_test_bye(optimizer: FollowUpConversationOptimizer = None):
  """
  Run a test case based on real conversation about shelter and transport spending.
  """
  return test_with_inputs(
    insights=[
      "Your Food spending is lower than expected at $161 this month! 🥳",
      "Your Medical and Pharmacy spending is higher than expected at $60 this month. 😥"
    ],
    last_conversation=[
      ("Assistant", "Your shelter costs were $2,850, a touch more on Home expenses. Totally fine, just info for you! ✨"),
      ("Assistant", "Good job! Your transport spending is down a little this month at $324, mostly on car stuff. 🚗💨 Every bit helps!"),
      ("User", "Thanks! Bye for now!")
    ],
    optimizer=optimizer
  )


def main(batch: int = 1):
  """
  Main function to test the FollowUpConversationOptimizer
  
  Args:
    batch: Batch number (1, 2, 3, or 4) to determine which tests to run
      - Batch 1: Acknowledgment test
      - Batch 2: Closing test
      - Batch 3: General Conversational Turn test
      - Batch 4: Shelter and Transport Spending test
  """
  optimizer = FollowUpConversationOptimizer()
  
  if batch == 1:
    print("\n" + "="*80)
    print("TEST 1: Acknowledgment")
    print("="*80)
    run_test_acknowledgment(optimizer)
  elif batch == 2:
    print("\n" + "="*80)
    print("TEST 2: Closing")
    print("="*80)
    run_test_satisfaction_ender(optimizer)
  elif batch == 3:
    print("\n" + "="*80)
    print("TEST 3: General Conversational Turn")
    print("="*80)
    run_test_general_conversational_turn(optimizer)
  elif batch == 4:
    print("\n" + "="*80)
    print("TEST 4a: Thanks for update")
    print("="*80)
    run_test_thanks_for_update(optimizer)
    print("\n" + "="*80)
    print("TEST 4b: Thanks")
    print("="*80)
    run_test_thanks(optimizer)
  elif batch == 5:
    print("\n" + "="*80)
    print("TEST 5: Bye")
    print("="*80)
    run_test_bye(optimizer)
  else:
    print(f"Invalid batch number: {batch}. Use 1, 2, 3, 4, or 5.")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run follow-up conversation optimizer tests in batches')
  parser.add_argument('--batch', type=int, default=1, choices=[1, 2, 3, 4, 5],
                      help='Batch number to run (1, 2, 3, 4, or 5)')
  args = parser.parse_args()
  main(batch=args.batch)
