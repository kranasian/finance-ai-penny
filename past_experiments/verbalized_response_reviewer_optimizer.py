from google import genai
from google.genai import types
import os
import json
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """**Objective:** Evaluate the `ai_review_response` for red flags against the conversation history, prioritizing its *appropriateness*.

**Input:**

Input is a list of conversation turns. The `conversation_history` is derived from all elements in the input list *except* the final element, which is always the `ai_review_response`.

**Output:**
- The output is a JSON object where each key is a red flag key that specifies an issue in the `ai_review_response`, and the value is the rationale for choosing that flag. Multiple keys can be present if there are several issues. **However, if `good_response` is among the applicable keys, it *must* be the ONLY key present in the output.** The rationale should be concise but comprehensively explain why the red flag was chosen, making it easily understandable even without having to reference the list of guidelines. The output *must* be a JSON object.

**Possible red flag keys (with definitions):**
- `good_response`: The response is appropriate for the conversation. This includes straightforwardly answering the `Human`'s last inquiry, or clearly stating limitations (e.g., technical error, out of scope) in a well-worded and helpful manner, even if the user's request cannot be fulfilled. If `ai_review_response` contains a greeting (e.g., "Hey," "Hi," "Hello," "Good morning") AND ((it is the very first message in the entire exchange) OR (it is an immediate response to the Human's first message in the entire exchange)) AND no other red flags apply, then this *must* be `good_response`. If no greeting is present and no other red flag keys apply, then this *must* be `good_response`. This must be the only flag used if no other red flags apply.
- `non_sense`: Irrelevant or illogical response, or response with inconsistent computations, or if the `ai_review_response` contradicts information previously provided by the AI in the conversation. If `ai_review_response` contradicts information previously provided by the AI in the conversation, then this *must* be `non_sense`, and `good_response` CANNOT be applied.
- `repetitive_information`: Repeats information from within its own response without value/context change.
- `incoherent_flow`: Leads to a dead end or breaks the conversation flow. This flag *must* be used if `ai_review_response` contains a greeting (e.g., "Hey," "Hi," "Hello," "Good morning") AND NOT ((it is the very first message in the entire exchange) OR (it is an immediate response to the Human's first message in the entire exchange)). If this `incoherent_flow` condition for a greeting is met, then `good_response` CANNOT be applied, even if other conditions for `good_response` might seem to apply.
- `verbose`: Does not straightforwardly answer the `Human` last inquiry because of too many details or information unnecessary to the question.
- `incomplete`: Too brief or lacks necessary information.
"""

class VerbalizedResponseReviewer:
  """Handles all Gemini API interactions for checking AI responses for red flags"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for checking AI response red flags"""
    # API Configuration
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)
    
    # Model Configuration
    self.thinking_budget = 1024
    self.model_name = model_name
    
    # Generation Configuration Constants
    self.top_k = 40
    self.top_p = 0.95
    self.temperature = 0.1
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

  
  def evaluate_response(self, input_list_json_str: str) -> dict:
    """
    Evaluate an AI response for red flags using Gemini API.
    
    Args:
      input_list_json_str: JSON string of a list of conversation turns, where the final element is the ai_review_response.
      
    Returns:
      Dictionary containing the red flag key and rationale
    """
    # Parse the input JSON string into a Python list
    input_list = json.loads(input_list_json_str)

    # Extract ai_review_response (last element of the list)
    ai_review_response_dict = input_list[-1]
    ai_review_response = ai_review_response_dict["ai_review_response"]
    
    # Extract conversation history (all elements except the last)
    conversation_history_list = input_list[:-1]
    
    # Format conversation history for the prompt
    formatted_conversation_history = []
    for turn in conversation_history_list:
      if "Human" in turn:
        formatted_conversation_history.append({"speaker": "Human", "message": turn["Human"]})
      elif "AI" in turn:
        formatted_conversation_history.append({"speaker": "AI", "message": turn["AI"]})

    # Create request text with the new input structure
    request_payload = {
        "conversation_history": formatted_conversation_history,
        "ai_review_response": ai_review_response
    }
    request_text_str = f"""input: {json.dumps(request_payload, indent=2)}
output: """
    
    print(f"\n{'='*80}")
    print("INPUT JSON:")
    print(input_list_json_str)
    print("="*80)
    
    request_text = types.Part.from_text(text=request_text_str)
    
    # Create content and configuration
    contents = [types.Content(role="user", parts=[request_text])]
    
    generate_content_config = types.GenerateContentConfig(
      temperature=self.temperature,
      top_p=self.top_p,
      max_output_tokens=4096,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=self.system_prompt)],
      thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
    )

    # Generate response
    output_text = ""
    try:
      for chunk in self.client.models.generate_content_stream(
        model=self.model_name,
        contents=contents,
        config=generate_content_config,
      ):
        if chunk.text is not None:
          output_text += chunk.text
    except Exception as e:
      error_msg = f"API Error: {str(e)}";
      print(f"\n{'='*80}");
      print("ERROR:");
      print(error_msg);
      print("="*80);
      raise Exception(f"Failed to generate response from Gemini API: {error_msg}")
    
    # Check if response is empty
    if not output_text or not output_text.strip():
      raise ValueError(f"Empty response from model. Check API key and model availability.")
    
    print(f"{'='*80}")
    print("RESPONSE OUTPUT:")
    print(output_text.strip())
    print("="*80)

    # Try to parse as JSON, if it fails return as text
    try:
      cleaned_output = output_text.strip() # Initialize cleaned_output

      # Remove markdown code blocks if present
      if cleaned_output.startswith("```json"):
        json_start_idx = cleaned_output.find("```json") + len("```json")
        json_end_idx = cleaned_output.rfind("```")
        if json_start_idx != -1 and json_end_idx != -1 and json_end_idx > json_start_idx:
          cleaned_output = cleaned_output[json_start_idx:json_end_idx].strip()
      elif cleaned_output.startswith("```"):
        # Generic markdown block, just remove start and end lines
        lines = cleaned_output.split('\n')
        if len(lines) > 1 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
          cleaned_output = '\n'.join(lines[1:-1]).strip()
        
      # Find the first opening brace and the last closing brace to extract the JSON string
      json_start = cleaned_output.find('{')
      json_end = cleaned_output.rfind('}') + 1

      if json_start != -1 and json_end > json_start:
        json_str = cleaned_output[json_start:json_end]
        return json.loads(json_str)
      else:
        # Fallback if no valid JSON object found
        raise json.JSONDecodeError("No JSON object found or invalid JSON structure", cleaned_output, 0)

    except json.JSONDecodeError as e:
      # If not valid JSON, return as a dict with the raw text and error info
      print(f"\nWarning: Failed to parse JSON response: {e}")
      print(f"Raw output preview: {output_text[:200]}...")
      return {"raw_output": output_text.strip(), "parse_error": str(e)}


BATCHES = {
  "initial_greeting_good": [
    {
      "Human": "Hello, how are you doing today? 👋"
    },
    {
      "ai_review_response": "Hello there! I'm doing great, thanks for asking! How can I help you out today? ✨"
    }
  ],
  "initial_greeting_human_first_good": [
    {
      "Human": "How much did I earn last year?"
    },
    {
      "ai_review_response": "Hey there! 😊 Last year, you earned $58,000💰"
    }
  ],
  "initial_greeting_human_first_good_2": [
    {
      "Human": "How much did I spend last year?"
    },
    {
      "ai_review_response": "Hey there! 😊 Last year, you spent $40,000💰"
    }
  ],
  "no_greeting_good": [
    {
      "Human": "How much cash do I have in my accounts? 💸"
    },
    {
      "AI": "Let me quickly check your accounts for you! Just a moment. ⏳"
    },
    {
      "Human": "What's my total cash balance right now?"
    },
    {
      "ai_review_response": "You've got a grand total of $27,593 across your checking and savings accounts! 💰"
    }
  ],
  "unnecessary_greeting_bad": [
    {
      "Human": "Can you tell me about my recent transactions? 🧾"
    },
    {
      "AI": "Absolutely! I can certainly help you with that. 👍"
    },
    {
      "Human": "What were the latest 5 transactions I made?"
    },
    {
      "ai_review_response": "Hello! ✨ Here are your recent transactions: ... (This is an example, actual transactions would be listed here) "
    }
  ],
  "ai_contradiction_non_sense": [
   {
      "Human": "What's my current account balance, please? 🤔"
    },
    {
      "AI": "Your current balance is exactly $1,000. 🏦"
    },
    {
      "Human": "And what was it after my very last transaction?"
   },
    {
      "ai_review_response": "Oops, my apologies! Your current balance is actually $500. 😅 The previous balance I mentioned was incorrect. My mistake! 🙏"
    }
   ],
  "good_disclaimer": [
    {
      "Human": "Can you transfer some funds to my friend's international bank account for me? 🌍"
    },
    {
      "ai_review_response": "I truly apologize, but I'm currently unable to perform international fund transfers due to some system limitations at the moment. 😟 Please try again a bit later or reach out to our support team for assistance! 🙏"
    }
  ],
  "incomplete_response_bad": [
    {
      "Human": "Could you give me a summary of my financial activity for the last quarter? 📊"
    },
    {
      "ai_review_response": "Okay, I'm taking a look at your activity now. You definitely had several transactions! 🤔"
    }
  ],
  "greeting_ai_human_bad": [
    {
      "AI": "How can I help you out today? 💁‍♀️"
    },
    {
      "Human": "What's my current account balance? 💰"
    },
    {
      "ai_review_response": "Hello! Your balance is $1000. 🏦"
    }
  ],
  "greeting_human_ai_human_bad": [
    {
      "Human": "Hi there, how are you doing? 👋"
    },
    {
      "AI": "I'm doing wonderfully, thanks for asking! How can I assist you today? ✨"
    },
    {
      "Human": "What's the weather like outside right now? ☀️"
    },
    {
      "ai_review_response": "Hey there! The weather is absolutely sunny and beautiful! \ud83c\udf1e"
    }
  ],
  "greeting_human_ai_bad": [
    {
      "Human": "Tell me a fun joke! 😂"
    },
    {
      "AI": "Sure thing, I've got a great one for you! Get ready to laugh! \ud83d\ude06"
    },
    {
      "ai_review_response": "Good morning! Why don't scientists trust atoms? Because they make up everything! ⚛️"
    }
  ],
  "cash_flow_incoherent_greeting_bad": [
    {
      "Human": "what was my salary last year"
    },
    {
      "AI": "Okay, so for your cash flow prediction for the next 3 months, here's the scoop! ✨\nYour projected income for September, October, and November 2025 is looking good at $8,870. And your projected spending for August, September, and October 2025 is $8,567.\nSince the income and spending forecasts cover slightly different months, I can't give you a direct combined difference for the exact same three-month period. But we can definitely look at each one individually and make sure you're on track! 👍"
    },
    {
      "Human": "Predict my cash flow situation for the next 3 months."
    },
    {
      "ai_review_response": "Hey there! ✨ For the next 3 months (Sept-Nov 2025), your projected income is $9,882! \ud83d\udcb0 Your projected spending for Sept & Oct 2025 is $6,255. That means you're looking at a positive difference of $646 for those two months! \ud83d\udc4d We don't have November spending yet, but this is a great start! \ud83d\ude0a"
    }
  ],
  "greeting_ai_ai_ai_human_bad": [
    {
      "AI": "Hello! How can I assist you today? 🙋‍♀️"
    },
    {
      "AI": "I'm ready when you are! Just tell me what you need. 😊"
    },
    {
      "AI": "Still here, eager to help! What's on your mind? ✨"
    },
    {
      "Human": "What's my account balance?"
    },
    {
      "ai_review_response": "Hey there! Your current balance is $1,500. 💰"
    }
  ],
  "incomplete_response_no_answer": [
    {
      "Human": "Can you summarize my investments for the last quarter, including returns and growth percentages? 📈"
    },
    {
      "ai_review_response": "I'm currently gathering your investment data. It looks like you have a diverse portfolio! 🤔"
    }
  ],
  "verbose_over_explanation": [
    {
      "Human": "What was my highest expense last month? 💸"
    },
    {
      "ai_review_response": "To give you the most accurate answer, I need to access your transaction history. After carefully analyzing all your spending categories and cross-referencing them with various vendors, I can confidently tell you that your highest expense last month was $300 at 'Fancy Dinner Restaurant'. 🍽️"
    }
  ],
  "verbose_transaction_ids": [
    {
      "Human": "List my last 3 transactions. 🧾"
    },
    {
      "ai_review_response": "Here are your last 3 transactions: 💳\n1. Coffee Shop (ID: 7890) - $5.50\n2. Grocery Store (ID: 1234) - $75.20\n3. Online Shopping (ID: 5678) - $45.99"
    }
  ]
}

def test_with_inputs(input_list, reviewer=None):
  """
  Convenient method to test the reviewer with custom inputs.
  
  Args:
    input_list: List of dictionaries representing conversation turns, with the final element containing ai_review_response.
    reviewer: Optional VerbalizedResponseReviewer instance. If None, creates a new one.
    
  Returns:
    Dictionary containing the red flag key and rationale
  """
  if reviewer is None:
    reviewer = VerbalizedResponseReviewer()
  
  return reviewer.evaluate_response(json.dumps(input_list, indent=2))


def run_batch_test(batch_id: str, reviewer=None):
  """
  Run a specific test batch.
  """
  if batch_id not in BATCHES:
    raise ValueError(f"Batch '{batch_id}' not found. Available batches: {list(BATCHES.keys())}")
  
  print(f"\n{'='*80}")
  print(f"RUNNING BATCH: {batch_id}")
  print(f"{'='*80}\n")
  return test_with_inputs(BATCHES[batch_id], reviewer)


def main(batch_id: str = None):
  """
  Main function to test the verbalized response reviewer.
  Optionally runs a specific batch if batch_id is provided.
  """
  try:
    reviewer = VerbalizedResponseReviewer()
    results = {}
    
    if batch_id:
      result = run_batch_test(batch_id, reviewer)
      results[batch_id] = result
    else:
      for b_id in BATCHES:
        result = run_batch_test(b_id, reviewer)
        results[b_id] = result
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY:")
    print(json.dumps(results, indent=2))
    print(f"{'='*80}\n")
    print("✅ Successfully evaluated the AI response(s)!")
    
    return results
  except ValueError as e:
    # Handle API key or initialization errors
    print(f"\n{'='*80}")
    print("ERROR OCCURRED:")
    print(str(e))
    print(f"{'='*80}\n")
    print("Troubleshooting:")
    print("1. Check that GEMINI_API_KEY is set in your .env file")
    print("2. Verify your API key is valid and has proper permissions")
    print("3. Ensure you have network connectivity")
    return None
  except Exception as e:
    # Handle API errors
    error_str = str(e)
    print(f"\n{'='*80}")
    print("ERROR OCCURRED:")
    print(error_str)
    print(f"{'='*80}\n")
    
    if "API key" in error_str or "PERMISSION_DENIED" in error_str or "403" in error_str:
      print("\n⚠️  API Key Issue Detected:")
      print("   Your GEMINI_API_KEY appears to be invalid or has been revoked.")
      print("   Please update your .env file with a valid API key.")
      print("   Get a new API key from: https://makersuite.google.com/app/apikey")
    else:
      print("\nTroubleshooting:")
      print("1. Check that GEMINI_API_KEY is set in your .env file")
      print("2. Verify your API key is valid and has proper permissions")
      print("3. Ensure you have network connectivity")
      print("4. Check the error message above for specific details")
    
    return None


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Run Verbalized Response Reviewer with optional batch ID.")
  parser.add_argument("--batch", type=str, help="ID of the batch to run (e.g., initial_greeting_good). If not provided, all batches will run.")
  args = parser.parse_args()
  main(batch_id=args.batch)