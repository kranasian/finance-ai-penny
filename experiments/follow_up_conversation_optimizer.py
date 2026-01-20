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
    "key": types.Schema(
      type=types.Type.STRING,
      description="The key of the insight that was integrated into the reply, if any. Only include if an insight was actually mentioned in the reply."),
  },
  required=["reply", "reasoning"]
)

SYSTEM_PROMPT = r"""#### 1. Role & Goal
Financial advisor assistant generating brief follow-up responses.

#### 2. Core Task
Generate BRIEF reply matching conversation type:
- **Acknowledgments:** User thanked, confirmed understanding, or acknowledged info
- **Closing:** Question answered, satisfaction expressed, or natural conclusion
- **General Conversational:** User made general comment needing response

#### 3. Input Data
- `insights`: Array of NEW financial insight objects, each containing:
  - `key`: Unique identifier for the insight
  - `value`: The insight text (spending patterns, forecasts, categories, etc.)
- `last_conversation`: Array of conversation message objects, each containing:
  - `role`: Either "user" or "ai" (representing Penny)
  - `message`: The message text as a string
  Messages are in chronological order (oldest first, newest last). The last message is the user's most recent message.

#### 4. Output Requirements
Return JSON:
- `reply`: BRIEF response (1-2 sentences max, friendly financial advisor tone)
- `reasoning`: Brief explanation
- `key`: (Optional) The key of the insight that was integrated into the reply. Only include if an insight was actually mentioned in the reply.

**Insight Usage:** ENCOURAGE mentioning most appropriate insight (using its `value`) if it:
- Relates naturally to conversation topic
- Adds value without feeling forced
- Has NOT been mentioned in conversation yet
- Is DIFFERENT from topics already discussed (avoid repeating)
- **Preserve all amounts/details**—when mentioning insights, include all numerical values, amounts, and specific details exactly as provided

**If an insight is integrated into the reply:** Include the insight's `key` in the output `key` field. If no insight is mentioned, omit the `key` field.

**Reply Types (All Brief):**
- Acknowledgments: Warm brief acknowledgment + optional relevant insight
- Closing: Brief graceful conclusion + optional insight before closing
- General: Brief rapport maintenance + optional related insight

#### 5. Critical Constraints
- Use only provided data
- Consider full conversation context
- Prioritize natural flow—insights enhance, don't disrupt
- **DO NOT repeat any topics, categories, amounts, or information already mentioned in `last_conversation`**—keep the reply fresh and avoid redundancy
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
  
  def determine_follow_up_response(self, insights: list, last_conversation: list) -> dict:
    """
    Determine the most appropriate follow-up response based on insights and conversation history.
    
    Args:
      insights: List of insight objects, where each object contains:
        - `key`: Unique identifier for the insight
        - `value`: The insight text (a NEW financial insight that has NOT been mentioned in the conversation yet)
      last_conversation: List of conversation message objects, each containing:
        - `role`: Either "user" or "ai" (representing Penny)
        - `message`: The message text as a string
        Must contain back-and-forth exchanges plus the user's final message. Messages are in chronological order (oldest first, newest last). The last message is the user's most recent message.
      
    Returns:
      Dictionary containing reply, reasoning, and optionally key (if an insight was integrated)
    """
    # Validate that last_conversation is a list
    if not isinstance(last_conversation, list):
      raise ValueError("last_conversation must be a list of message objects")
    
    # Validate that each item in last_conversation has the required structure
    for i, msg in enumerate(last_conversation):
      if not isinstance(msg, dict):
        raise ValueError(f"last_conversation[{i}] must be a dictionary")
      if "role" not in msg or "message" not in msg:
        raise ValueError(f"last_conversation[{i}] must have 'role' and 'message' fields")
      if msg["role"] not in ["user", "ai"]:
        raise ValueError(f"last_conversation[{i}]['role'] must be either 'user' or 'ai'")
      if not isinstance(msg["message"], str):
        raise ValueError(f"last_conversation[{i}]['message'] must be a string")
    
    # Create input JSON structure
    input_json = {
      "insights": insights,
      "last_conversation": last_conversation
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


def test_with_inputs(insights: list, last_conversation: list, optimizer: FollowUpConversationOptimizer = None):
  """
  Convenient method to test the optimizer with custom inputs.
  
  Args:
    insights: List of insight objects, where each object contains:
      - `key`: Unique identifier for the insight
      - `value`: The insight text (a NEW financial insight that has NOT been mentioned in the conversation yet)
    last_conversation: List of conversation message objects, each containing:
      - `role`: Either "user" or "ai" (representing Penny)
      - `message`: The message text as a string
      Must contain back-and-forth exchanges plus the user's final message. Messages are in chronological order (oldest first, newest last).
    optimizer: Optional FollowUpConversationOptimizer instance. If None, creates a new one.
    
  Returns:
    Dictionary containing reply, reasoning, and optionally key (if an insight was integrated)
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
      {"key": "uncategorized_transaction_1", "value": "How should I categorize your $2,850 outflow to Property Group LLC? Was it for Clothing?"},
      {"key": "dining_out_decrease", "value": "Your dining out expenses have decreased by 15% compared to last month."}
    ],
    last_conversation=[
      {"role": "user", "message": "What are my spending patterns this month?"},
      {"role": "ai", "message": "Your shopping spending increased to $1,725 this month, which is higher than usual."},
      {"role": "user", "message": "I see, I did buy some new clothes."},
      {"role": "ai", "message": "That makes sense. Would you like me to help you set a budget for shopping next month?"},
      {"role": "user", "message": "Got it, thanks for letting me know!"}
    ],
    optimizer=optimizer
  )


def run_test_satisfaction_ender(optimizer: FollowUpConversationOptimizer = None):
  """
  Run a test case where user expresses satisfaction (should end conversation).
  """
  return test_with_inputs(
    insights=[
      {"key": "uncategorized_transactions", "value": "You have 3 uncategorized transactions totaling $450."},
      {"key": "tuition_lower", "value": "Your Tuition and Kids Activities spending is lower than expected at $581 this month! 🥳"}
    ],
    last_conversation=[
      {"role": "user", "message": "What are my account balances?"},
      {"role": "ai", "message": "Your checking account has $5,000 and your savings has $15,000."},
      {"role": "user", "message": "Perfect, thanks!"},
      {"role": "ai", "message": "You're welcome! Is there anything else you'd like to know?"},
      {"role": "user", "message": "Thanks, that's exactly what I needed!"}
    ],
    optimizer=optimizer
  )


def run_test_general_conversational_turn(optimizer: FollowUpConversationOptimizer = None):
  """
  Run a test case where user makes a general conversational comment (General Conversational Turn scenario).
  """
  return test_with_inputs(
    insights=[
      {"key": "subscription_increase", "value": "Your subscription costs have increased by $25 this month."},
      {"key": "entertainment_increase", "value": "Heads up! Your Entertainment spending increased to $173 this week. 📈"}
    ],
    last_conversation=[
      {"role": "user", "message": "How am I doing financially this month?"},
      {"role": "ai", "message": "Your food spending increased to $615 this month, mostly from dining out. Overall, you're staying within your budget goals."},
      {"role": "user", "message": "That makes sense, I did go out to eat more often this month."},
      {"role": "ai", "message": "Yes, and you're still on track with your overall financial goals."},
      {"role": "user", "message": "Good to know, thanks for the update!"}
    ],
    optimizer=optimizer
  )


def run_test_thanks_for_update(optimizer: FollowUpConversationOptimizer = None):
  """
  Run a test case based on real conversation about shelter and transport spending.
  """
  return test_with_inputs(
    insights=[
      {"key": "food_lower", "value": "Your Food spending is lower than expected at $161 this month! 🥳"},
      {"key": "medical_higher", "value": "Your Medical and Pharmacy spending is higher than expected at $60 this month. 😥"}
    ],
    last_conversation=[
      {"role": "ai", "message": "Your shelter costs were $2,850, a touch more on Home expenses. Totally fine, just info for you! ✨"},
      {"role": "ai", "message": "Good job! Your transport spending is down a little this month at $324, mostly on car stuff. 🚗💨 Every bit helps!"},
      {"role": "user", "message": "Thanks for the update!"}
    ],
    optimizer=optimizer
  )

def run_test_thanks(optimizer: FollowUpConversationOptimizer = None):
  """
  Run a test case based on real conversation about shelter and transport spending.
  """
  return test_with_inputs(
    insights=[
      {"key": "food_lower", "value": "Your Food spending is lower than expected at $161 this month! 🥳"},
      {"key": "medical_higher", "value": "Your Medical and Pharmacy spending is higher than expected at $60 this month. 😥"}
    ],
    last_conversation=[
      {"role": "ai", "message": "Your shelter costs were $2,850, a touch more on Home expenses. Totally fine, just info for you! ✨"},
      {"role": "ai", "message": "Good job! Your transport spending is down a little this month at $324, mostly on car stuff. 🚗💨 Every bit helps!"},
      {"role": "user", "message": "Thanks!"}
    ],
    optimizer=optimizer
  )


def run_test_bye(optimizer: FollowUpConversationOptimizer = None):
  """
  Run a test case based on real conversation about shelter and transport spending.
  """
  return test_with_inputs(
    insights=[
      {"key": "food_lower", "value": "Your Food spending is lower than expected at $161 this month! 🥳"},
      {"key": "medical_higher", "value": "Your Medical and Pharmacy spending is higher than expected at $60 this month. 😥"}
    ],
    last_conversation=[
      {"role": "ai", "message": "Your shelter costs were $2,850, a touch more on Home expenses. Totally fine, just info for you! ✨"},
      {"role": "ai", "message": "Good job! Your transport spending is down a little this month at $324, mostly on car stuff. 🚗💨 Every bit helps!"},
      {"role": "user", "message": "Thanks! Bye for now!"}
    ],
    optimizer=optimizer
  )


def run_test_health_spending_thanks(optimizer: FollowUpConversationOptimizer = None):
  """
  Run a test case based on real conversation about health spending with thanks acknowledgment.
  """
  return test_with_inputs(
    insights=[
      {"key": "tuition_lower_health", "value": "Great job! Your Tuition is drastically lower this month at $3,773! 🎓"},
      {"key": "salary_lower", "value": "Oh no! Your Salary is drastically lower this month at $2,841. 😢"}
    ],
    last_conversation=[
      {"role": "ai", "message": "Hey! 👋 Your health spending is at $881 this week, way above the $882 forecast! 😬 Check it out here: https://p3n.me/wWB"},
      {"role": "ai", "message": "Hey there! 📱 Looks like you're on track with medicine & pharmacy this month, only spending $304 so far. That's way less than the $8,881 we expected! 🎉 Keep it up! 💪"},
      {"role": "user", "message": "thanks!"}
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
  elif batch == 6:
    print("\n" + "="*80)
    print("TEST 6: Health Spending Thanks")
    print("="*80)
    run_test_health_spending_thanks(optimizer)
  else:
    print(f"Invalid batch number: {batch}. Use 1, 2, 3, 4, 5, or 6.")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run follow-up conversation optimizer tests in batches')
  parser.add_argument('--batch', type=int, default=1, choices=[1, 2, 3, 4, 5, 6],
                      help='Batch number to run (1, 2, 3, 4, 5, or 6)')
  args = parser.parse_args()
  main(batch=args.batch)
