from google import genai
from google.genai import types
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()


# Output Schema - array of result objects
SCHEMA = types.Schema(
  type=types.Type.ARRAY,
  items=types.Schema(
    type=types.Type.OBJECT,
    properties={
      "match_id": types.Schema(
        type=types.Type.STRING,
        description="Unique identifier for the pair. Same as input."
      ),
      "reasoning": types.Schema(
        type=types.Type.STRING,
        description="Brief 1-2 sentence explanation focusing on decisive factors for the classification"
      ),
      "result": types.Schema(
        type=types.Type.STRING,
        enum=["same", "different"],
        description="Classification result: 'same' if names represent the same entity/establishment, 'different' if they represent different entities/establishments"
      ),
      "confidence": types.Schema(
        type=types.Type.STRING,
        enum=["high", "medium", "low"],
        description="Confidence level: 'high' for strong evidence, 'medium' for good evidence with some ambiguity, 'low' for weak/conflicting evidence"
      )
    },
    required=["match_id", "result", "confidence", "reasoning"]
  )
)


SYSTEM_PROMPT = """You are an expert at determining if two transaction names represent the same kind of charge from the same entity, for the purpose of financial categorization.

## Core Task

Analyze pairs of transaction sets and determine if they should be classified as "same" or "different". The primary decision should be based on this question: **Could the `short_name` from the left item AND the `short_name` from the right item EACH be used to accurately and completely describe all transactions in BOTH sets?**

- If **yes**, the result is **"same"**. This means the names are effectively interchangeable, and renaming would not cause a loss of critical information.
- If **no**, the result is **"different"**. This applies when at least one of the names is too specific or too generic to correctly describe the other set of transactions.

## Key Considerations

- **Ignore Geographic Locations**: Transactions at different physical locations should be considered the same.

## Analysis Heuristics

**1. Marketplaces vs. Payment Processors:**
- **Marketplaces**: Transactions made through a distinct marketplace or platform (e.g., DoorDash, eBay, Best Buy) are **different** from a direct transaction with a merchant. The marketplace is a critical part of the transaction's context.
- **Payment Processors**: The involvement of a pure payment processor (e.g., Stripe, Paypal, Square) should be **ignored**. These do not change the fundamental nature of the transaction.

**2. Product/Service Distinction:**
- Transactions are **different** if they represent distinct products, service tiers, or types of charges from the same company. Renaming one to the other would be inaccurate.

**3. Payment & Transfer Specificity:**
- A more specific payment or transfer type is **different** from a general one, as essential detail is lost in generalization.
- For Person-to-Person (P2P) transfers, a transaction with a specific purpose (text after a colon) is **different** from a general transfer to the same person.

**4. Sub-brands and Departments:**
- Sub-brands or departments of the same parent company should generally be considered the **same**, as the primary entity is the same.

**5. Name & Description Analysis (Supporting Evidence):**
- Use `short_name`, `raw_names`, and `description` to determine if one name is simply a variation of the other or if they represent fundamentally different things.

## Output Format

JSON array where each element contains:
- `match_id`: Same as input
- `reasoning`: 1-2 sentence explanation focusing on decisive factors
- `result`: "same" or "different"
- `confidence`: "high", "medium", or "low"

## Rules
- Process all pairs and maintain input order.
- Base your decision on the Core Task and Analysis Heuristics.
- Be conservative with "high" confidence.
- Reasoning must be 1-2 sentences and focus only on the most decisive factors.
"""





class SameSummarizedNameClassifierOptimizer:
  """Handles all Gemini API interactions for detecting if names are from the same entity or establishment"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for similarity detection"""
    # API Configuration
    self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    
    # Model Configuration
    self.thinking_budget = 4096
    self.model_name = model_name
    
    # Generation Configuration Constants
    self.temperature = 0.6
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

  def detect_similarity(self, transaction_history_pairs: list) -> list:
    """
    Detect if establishment pairs are from the same entity or establishment using Gemini API.
    
    Args:
      transaction_history_pairs: A list of dictionaries, each containing:
        - match_id: Unique identifier for the pair
        - left: Name object with short_name, raw_names (list), description, amounts
        - right: Name object with short_name, raw_names (list), description, amounts
      
    Returns:
      A list of dictionaries, each containing:
        - match_id: The same match_id from input
        - result: "same" (same entity/establishment) or "different" (different entities/establishments)
        - confidence: "high", "medium", or "low"
        - reasoning: Explanation string
    """
    import json
    
    # Convert input to JSON string
    input_json = json.dumps(transaction_history_pairs, indent=2)
    
    # Create request text
    request_text = types.Part.from_text(text=f"""input:
{input_json}

output:""")
    
    # Create content and configuration
    contents = [types.Content(role="user", parts=[request_text])]
    
    generate_content_config = types.GenerateContentConfig(
      temperature=self.temperature,
      top_p=self.top_p,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=self.system_prompt)],
      thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
      response_mime_type="application/json",
      response_schema=SCHEMA,
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
    
    # Parse JSON response
    try:
      # Extract JSON from the response (in case there's extra text)
      output_text = output_text.strip()
      # Try to find JSON array in the response
      if output_text.startswith("```"):
        # Remove markdown code blocks if present
        lines = output_text.split("\n")
        json_lines = []
        in_code_block = False
        for line in lines:
          if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue
          if in_code_block or (not in_code_block and line.strip()):
            json_lines.append(line)
        output_text = "\n".join(json_lines)
      
      result = json.loads(output_text)
      return result
    except json.JSONDecodeError as e:
      raise ValueError(f"Failed to parse JSON response: {e}\nResponse was: {output_text}")
  
  
  def get_available_models(self):
    """
    Get list of available Gemini models.
    
    Returns:
      List of available model names
    """
    try:
      models = genai.list_models()
      return [m.name for m in models if 'generateContent' in m.supported_generation_methods]
    except Exception as e:
      raise Exception(f"Failed to get models: {str(e)}")


def _run_test_with_logging(transaction_history_pairs: list, detector: SameSummarizedNameClassifierOptimizer = None):
  """
  Internal helper function that runs a test with consistent logging.
  
  Args:
    transaction_history_pairs: List of establishment pairs to analyze
    detector: Optional SameSummarizedNameClassifierOptimizer instance. If None, creates a new one.
    
  Returns:
    The detection results as a list
  """
  import json
  
  if detector is None:
    detector = SameSummarizedNameClassifierOptimizer()
  
  # Print the input
  print("=" * 80)
  print("INPUT:")
  print("=" * 80)
  print(json.dumps(transaction_history_pairs, indent=2))
  print("=" * 80)
  print()
  
  try:
    result = detector.detect_similarity(transaction_history_pairs)
    
    # Print the output
    print("=" * 80)
    print("OUTPUT:")
    print("=" * 80)
    print(json.dumps(result, indent=2))
    print("=" * 80)
    print()
    
    return result
  except Exception as e:
    print(f"**Error**: {str(e)}")
    import traceback
    print(traceback.format_exc())
    print("=" * 80)
    return None


def test_batch_1(detector: SameSummarizedNameClassifierOptimizer = None):
  """
  Test batch 1: Basic cases & Payment Processors
  """
  transaction_history_pairs = [
    {
      "match_id": "2112:2113",
      "left": {
        "short_name": "OpenAI",
        "raw_names": [
          "OPENAI",
          "OPENAI +14158799686 USA"
        ],
        "description": "sells access to its large language model API, including text generation, translation, and code completion",
        "amounts": [6.00, 5.32]
      },
      "right": {
        "short_name": "OpenAI Chatgpt",
        "raw_names": ["Openai Chatgpt Subscr"],
        "description": "sells subscriptions to use the AI chatbot for various tasks such as writing, coding, and problem-solving",
        "amounts": [21.28, 20.00]
      }
    },
    {
        "match_id": "SPOT-01",
        "left": {
            "short_name": "Spotify Silver Plan",
            "raw_names": ["Spotify Silver Plan"],
            "description": "Subscription for music streaming service.",
            "amounts": [9.99]
        },
        "right": {
            "short_name": "Spotify Platinum",
            "raw_names": ["SPOTIFY PLATINUM"],
            "description": "Premium subscription for music and podcast streaming.",
            "amounts": [15.99]
        }
    },
    {
        "match_id": "GRAB-01",
        "left": {
            "short_name": "Grab: Wendy's",
            "raw_names": ["GrabFood*Wendy's"],
            "description": "Food delivery from a fast-food chain.",
            "amounts": [12.50]
        },
        "right": {
            "short_name": "Wendy's",
            "raw_names": ["WENDY'S"],
            "description": "Fast-food restaurant specializing in hamburgers.",
            "amounts": [18.75]
        }
    }
  ]
  return _run_test_with_logging(transaction_history_pairs, detector)


def test_batch_2(detector: SameSummarizedNameClassifierOptimizer = None):
  """
  Test batch 2: Name variations
  """
  transaction_history_pairs = [
    {
        "match_id": "FB-01",
        "left": {
            "short_name": "Facebook.com",
            "raw_names": ["FACEBOOK.COM"],
            "description": "Social media platform for connecting with friends and family.",
            "amounts": [10.00]
        },
        "right": {
            "short_name": "Facebook",
            "raw_names": ["Facebook"],
            "description": "Online social networking service.",
            "amounts": [15.50]
        }
    },
    {
        "match_id": "PP-01",
        "left": {
            "short_name": "Paypal Retry Payment",
            "raw_names": ["PAYPAL *RETRY PYMT"],
            "description": "A second attempt for a PayPal transaction.",
            "amounts": [45.00]
        },
        "right": {
            "short_name": "Paypal Payment",
            "raw_names": ["PayPal Payment"],
            "description": "A standard payment made via PayPal.",
            "amounts": [45.00]
        }
    },
    {
      "match_id": "2124:2125",
      "left": {
        "short_name": "Marco's Pizza",
        "raw_names": ["Marco's Pizza"],
        "description": "pizza restaurant that sells a variety of pizzas, subs, wings, sides, and desserts",
        "amounts": [70.92, 31.37]
      },
      "right": {
        "short_name": "Marcos Pizza",
        "raw_names": ["MARCOS PIZZA REF# 505100025146 WAXAHACHIE,TX Card ending in 3458"],
        "description": "A payment for food and beverages at a pizza restaurant",
        "amounts": [16.23, 22.70, 27.48, 23.97]
      }
    }
  ]
  return _run_test_with_logging(transaction_history_pairs, detector)


def test_batch_3(detector: SameSummarizedNameClassifierOptimizer = None):
  """
  Test batch 3: Sub-brands and Locations
  """
  transaction_history_pairs = [
    {
      "match_id": "2130:2131",
      "left": {
        "short_name": "Old Navy Kids",
        "raw_names": ["Old Navy Kids"],
        "description": "sells clothing, shoes, and accessories for children",
        "amounts": [39.09, 44.03, 50.88]
      },
      "right": {
        "short_name": "Old Navy",
        "raw_names": ["Old Navy"],
        "description": "sells clothing and accessories for men, women, and children",
        "amounts": [145.54, 189.48, 227.18]
      }
    },
    {
      "match_id": "2114:2115",
      "left": {
        "short_name": "MGM Grand",
        "raw_names": ["MGM Grand"],
        "description": "sells hotel rooms, casino gaming, dining, entertainment, and other resort amenities",
        "amounts": [128.93]
      },
      "right": {
        "short_name": "MGM Grand Detroit",
        "raw_names": ["Mgm grand detroi"],
        "description": "sells various types of casino games, including slots, table games, and poker",
        "amounts": [506.23, 1023.44, 1057.96, 1166.57, 1181.45, 1168.88]
      }
    },
    {
      "match_id": "2126:2127",
      "left": {
        "short_name": "Cleveland Marriott",
        "raw_names": [
          "MOBILE PURCHASE 0501 CLE EMBERS 681115 CLEVELAND OH XXXXX8751XXXXXXXXXX5621",
          "MOBILE PURCHASE CLE EMBERS 681115 CLEVELAND OH ON 05/01",
          "MOBILE PURCHASE CLE SPORTS ST1576 CLEVELAND OH ON 05/01",
          "MOBILE PURCHASE CLEVELAND MARRIOT CLEVELAND OH ON 04/30" 
        ],
        "description": "sells hotel accommodations, meeting rooms, and other services",
        "amounts": [5.40, 46.04, 23.67, 10.72, 43.88]
      },
      "right": {
        "short_name": "Marriott",
        "raw_names": ["Marriott"],
        "description": "sells hotel accommodations, resort stays, and other travel-related services",
        "amounts": [23.47, 6.43, 7.54]
      }
    }
  ]
  return _run_test_with_logging(transaction_history_pairs, detector)
  
  
def test_batch_4(detector: SameSummarizedNameClassifierOptimizer = None):
  """
  Test batch 4: Edge Cases & Ambiguity
  """
  transaction_history_pairs = [
    {
        "match_id": "ZELLE-01",
        "left": {
            "short_name": "Zelle to Gabby",
            "raw_names": ["Zelle Transfer to Gabby"],
            "description": "Peer-to-peer money transfer.",
            "amounts": [50.00]
        },
        "right": {
            "short_name": "Zelle to Gabby: Jollibee",
            "raw_names": ["Zelle to Gabby: Jollibee"],
            "description": "Peer-to-peer money transfer with a memo.",
            "amounts": [25.30]
        }
    },
    {
      "match_id": "2128:2129",
      "left": {
        "short_name": "Sales",
        "raw_names": ["Sales"],
        "description": "Undetermined",
        "amounts": [1581.33, 2366.88, 3487.34, 1582.67, 3181.54, 439.55]
      },
      "right": {
        "short_name": "Seamless",
        "raw_names": ["Seamless"],
        "description": "sells food delivery services from various restaurants",
        "amounts": [84.29, 52.17, 78.39, 46.76]
      }
    },
    {
      "match_id": "2134:2135",
      "left": {
        "short_name": "Dollar Tree",
        "raw_names": [
          "DOLLARTREE REF# 504200037537 WAXAHACHIE,TX Card ending in 3466",
          "DOLLARTREE REF# 504500026805 DESOTO,TX Card ending in 3466",
          "DOLLARTREE REF# 505200026122 WAXAHACHIE,TX Card ending in 3466",
          "Dollartree"
        ],
        "description": "A discount retail store selling a variety of household goods, groceries, and seasonal items.",
        "amounts": [4.06, 18.34, 5.34, 11.13]
      },
      "right": {
        "short_name": "Dollar Tree Payment",
        "raw_names": ["DOLLARTREE REF# 502300022154 DESOTO,TX Card ending in 3466"],
        "description": "payment to Dollar Tree with reference number 502300022154",
        "amounts": [4.06]
      }
    }
  ]
  return _run_test_with_logging(transaction_history_pairs, detector)


def main(batch: int = 0):
  """
  Main function to test the similarity detector optimizer
  
  Args:
    batch: Batch number (1, 2, 3, or 4) to determine which tests to run. 0 runs all.
  """
  print("Testing SameSummarizedNameClassifierOptimizer\n")
  detector = SameSummarizedNameClassifierOptimizer()
  
  if batch == 1 or batch == 0:
    print("Test Batch 1: Basic cases")
    print("-" * 80)
    test_batch_1(detector)
    print("\n")
    
  if batch == 2 or batch == 0:
    print("Test Batch 2: Name variations")
    print("-" * 80)
    test_batch_2(detector)
    print("\n")
    
  if batch == 3 or batch == 0:
    print("Test Batch 3: Similar names")
    print("-" * 80)
    test_batch_3(detector)
    print("\n")
    
  if batch == 4 or batch == 0:
    print("Test Batch 4: Edge cases")
    print("-" * 80)
    test_batch_4(detector)
    print("\n")
    
  if batch not in [0, 1, 2, 3, 4]:
    raise ValueError("batch must be 0, 1, 2, 3, or 4")
  
  print("All tests completed!")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run tests in batches')
  parser.add_argument('--batch', type=int, default=0, choices=[0, 1, 2, 3, 4],
                      help='Batch number to run (1, 2, 3, or 4). 0 runs all batches.')
  args = parser.parse_args()
  main(batch=args.batch)
