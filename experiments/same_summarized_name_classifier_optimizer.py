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

## Decision Axes (maximize discrimination on these)

Apply these axes consistently. When any axis distinguishes left from right, the result is **"different"**:

1. **Transaction type:** Different payment or transfer methods are different (e.g. mobile vs physical check, check vs cash, retry vs standard payment).
2. **Corrected identity via raw_names:** `short_name` and `description` may be wrong. Use `raw_names` on both sides to infer the true merchant or transaction type; verify whether left-group transactions truly belong with right-group before judging.
3. **Inflows vs outflows:** If one side is inflow and the other outflow (or inferable as such), result is **"different"**.
4. **Specific vs generic:** A transaction that indicates a **specific product, service, or tier** from a provider is **different** from a transaction that indicates only the **provider or brand**. Same brand with only location or branch in the name is **same**.
5. **ACH vs non-ACH:** ACH transactions are **different** from non-ACH (card, wire, etc.). Do not merge them.
6. **Physical vs online:** For brands that have both physical and online presence, physical-store transactions and online-store transactions are **different**. If the establishment is online-only, ".com" or domain in the name alone does not make them different → **same**.
7. **P2P memo/purpose:** For person-to-person transfers, use the **memo or stated purpose** to decide: if the purpose makes one transfer semantically distinct from the other, treat as **different**. Same recipient with same generic purpose → **same**; specific purpose vs no purpose or different purpose → **different**.

## Core Task

**Step 1 — Correct using raw_names:** Use `raw_names` on both left and right to infer the true merchant or transaction type. Do not trust `short_name` or `description` alone. Verify that left-group and right-group transactions truly belong together; only then apply the interchangeability test.

**Step 2 — Interchangeability:** For the *corrected* understanding of left and right:
1. Can the left (corrected) name accurately and completely describe all transactions in the right set?
2. Can the right (corrected) name accurately and completely describe all transactions in the left set?

- If **both** are yes → **"same"**.
- If **either** is no → **"different"**.

## Critical Rules (apply before interchangeability)

- **Specificity:** Sub-brands, departments, and specific product or service lines are **different** from the establishment's generic name. Specific service from a provider ≠ transaction showing only the provider.

- **Location and branch only:** If the **only** difference is physical location, city, state, or branch (or one name has a location and the other is the generic brand), result is **"same"**. Do not treat location in the name as a sub-brand.

- **Transaction type:** Different payment or transfer types are **different**: mobile vs physical check, check vs cash, retry vs standard, ACH vs non-ACH. For **merchant** line items, ignore noise like "Payment", "Ref", reference numbers, card suffixes, and location — same merchant → **same** unless the transaction type is fundamentally different (pending vs settled, retry vs standard).

- **ACH vs non-ACH:** ACH and non-ACH are **different**. Do not merge.

- **Pending vs non-pending:** Pending (authorization) and posted/settled are **different**. Do not merge.

- **Inflows vs outflows:** One side inflow and the other outflow → **"different"**.

- **Online vs physical:** Unless the establishment is online-only, physical store and online store of the same brand are **different**. Online-only: ".com" or domain alone → **same**.

- **P2P memo:** For person-to-person transfers, use memo/purpose: distinct purpose → **different**; same recipient, same generic purpose → **same**; specific purpose vs no or different purpose → **different**.

## Analysis Heuristics

- **Marketplaces:** A transaction through a distinct marketplace or platform is **different** from a direct transaction with the merchant. Ignore pure payment processor names when comparing merchant identity only.
- **Product/Service tier:** Distinct products, service tiers, or charge types from the same company → **different**.
- **Name evidence:** Use `short_name`, `raw_names`, and `description` to correct misprocessed names and to apply the axes above.

## Output Format & Reasoning Guide

Output a JSON array. Each element: `match_id`, `reasoning` (one concise phrase), `result` ("same" or "different"), `confidence` ("high", "medium", "low").

## Rules
- Process all pairs in input order.
- Apply Critical Rules first, then Core Task.
- Be conservative with "high" confidence.
- Reasoning must be a single, concise phrase.
"""





class SameSummarizedNameClassifierOptimizer:
  """Handles all Gemini API interactions for detecting if names are from the same entity or establishment"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for similarity detection"""
    # API Configuration
    self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    
    # Model Configuration
    self.thinking_budget = 0
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


def _run_sandbox_check(transaction_history_pairs: list, result: list, label: str = "Sandbox"):
  """Run checker on classifier output (Sandbox Execution)."""
  if result is None:
    return
  try:
    from check_same_summarized_name_classifier_optimizer import (
      run_test_case,
      CheckSameSummarizedNameClassifier,
    )
    checker = CheckSameSummarizedNameClassifier()
    run_test_case(label, transaction_history_pairs, result, [], checker)
  except Exception as e:
    print(f"Sandbox check failed: {e}")
    import traceback
    print(traceback.format_exc())


def _run_test_with_logging(
  transaction_history_pairs: list,
  detector: SameSummarizedNameClassifierOptimizer = None,
  run_sandbox: bool = True,
):
  """
  Internal helper function that runs a test with consistent logging.

  Args:
    transaction_history_pairs: List of establishment pairs to analyze
    detector: Optional SameSummarizedNameClassifierOptimizer instance. If None, creates a new one.
    run_sandbox: If True, run checker on the classifier output after detection.

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

    if run_sandbox and result:
      print("=" * 80)
      print("SANDBOX EXECUTION (Checker):")
      print("=" * 80)
      _run_sandbox_check(transaction_history_pairs, result, "Optimizer run")
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
      "match_id": "MSFT-01",
      "left": {
        "short_name": "Microsoft",
        "raw_names": ["MICROSOFT", "Microsoft Corp Billing"],
        "description": "sells software licenses, cloud services, and developer tools",
        "amounts": [12.00, 8.50]
      },
      "right": {
        "short_name": "Microsoft Xbox Game Pass",
        "raw_names": ["Xbox Game Pass Ultimate"],
        "description": "sells gaming subscription for console and PC access",
        "amounts": [14.99, 14.99]
      }
    },
    {
      "match_id": "DISNEY-01",
      "left": {
        "short_name": "Disney+ Basic",
        "raw_names": ["Disney Plus Monthly"],
        "description": "Subscription for streaming Disney movies and series.",
        "amounts": [7.99]
      },
      "right": {
        "short_name": "Disney+ Bundle",
        "raw_names": ["DISNEY PLUS HULU ESPN"],
        "description": "Premium bundle including Disney, Hulu, and ESPN+ streaming.",
        "amounts": [14.99]
      }
    },
    {
      "match_id": "DD-01",
      "left": {
        "short_name": "DoorDash: Chipotle",
        "raw_names": ["DOORDASH *CHIPOTLE"],
        "description": "Food delivery order placed through DoorDash from a Mexican grill chain.",
        "amounts": [22.40]
      },
      "right": {
        "short_name": "Chipotle",
        "raw_names": ["CHIPOTLE MEXICAN GRILL"],
        "description": "Fast-casual Mexican restaurant for dine-in and takeout.",
        "amounts": [18.00, 14.25]
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
      "match_id": "NETFLIX-01",
      "left": {
        "short_name": "Netflix.com",
        "raw_names": ["NETFLIX.COM"],
        "description": "Streaming service for movies and television series.",
        "amounts": [15.49]
      },
      "right": {
        "short_name": "Netflix",
        "raw_names": ["Netflix"],
        "description": "Online video streaming subscription.",
        "amounts": [15.49, 15.49]
      }
    },
    {
      "match_id": "VENMO-01",
      "left": {
        "short_name": "Venmo Retry Payment",
        "raw_names": ["VENMO *RETRY PAYMENT"],
        "description": "A second attempt for a Venmo transaction.",
        "amounts": [32.00]
      },
      "right": {
        "short_name": "Venmo Payment",
        "raw_names": ["Venmo payment"],
        "description": "A standard payment made via Venmo.",
        "amounts": [32.00]
      }
    },
    {
      "match_id": "PAPA-01",
      "left": {
        "short_name": "Papa John's Pizza",
        "raw_names": ["Papa John's Pizza"],
        "description": "pizza restaurant that sells pizzas, sides, and drinks",
        "amounts": [28.50, 34.20]
      },
      "right": {
        "short_name": "Papa Johns Pizza",
        "raw_names": ["PAPA JOHNS REF# 601200089012 AUSTIN,TX Card ending in 8821"],
        "description": "A payment for food at a pizza restaurant",
        "amounts": [19.99, 24.00, 31.45]
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
      "match_id": "TGT-01",
      "left": {
        "short_name": "Target Optical",
        "raw_names": ["Target Optical"],
        "description": "sells eyewear, contact lenses, and eye exams within Target stores",
        "amounts": [89.00, 120.50]
      },
      "right": {
        "short_name": "Target",
        "raw_names": ["Target"],
        "description": "sells general merchandise, groceries, and household goods",
        "amounts": [45.00, 78.30, 112.00]
      }
    },
    {
      "match_id": "HILTON-01",
      "left": {
        "short_name": "Hilton Garden Inn",
        "raw_names": ["Hilton Garden Inn"],
        "description": "sells hotel rooms, breakfast, and meeting space at midscale properties",
        "amounts": [142.00]
      },
      "right": {
        "short_name": "Hilton Garden Inn Phoenix",
        "raw_names": ["HILTON GARDEN INN PHOENIX AIRPORT"],
        "description": "sells hotel accommodations and amenities at an airport location",
        "amounts": [168.00, 155.00]
      }
    },
    {
      "match_id": "SBUX-01",
      "left": {
        "short_name": "Starbucks Downtown Seattle",
        "raw_names": [
          "STARBUCKS STORE 18492 SEATTLE WA",
          "STARBUCKS 18492 DOWNTOWN SEATTLE",
          "STARBUCKS SEATTLE WA"
        ],
        "description": "sells coffee, espresso drinks, and light food at a café location",
        "amounts": [5.85, 7.20, 6.45, 4.90]
      },
      "right": {
        "short_name": "Starbucks",
        "raw_names": ["Starbucks"],
        "description": "sells coffee and café beverages and food",
        "amounts": [6.00, 8.50]
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
      "match_id": "CASHAPP-01",
      "left": {
        "short_name": "Cash App to Marcus",
        "raw_names": ["Cash App payment to Marcus"],
        "description": "Peer-to-peer money transfer.",
        "amounts": [75.00]
      },
      "right": {
        "short_name": "Cash App to Marcus: Concert tickets",
        "raw_names": ["Cash App to Marcus: Concert tickets"],
        "description": "Peer-to-peer money transfer with a memo.",
        "amounts": [120.00]
      }
    },
    {
      "match_id": "MISC-01",
      "left": {
        "short_name": "Income",
        "raw_names": ["Income"],
        "description": "Undetermined",
        "amounts": [2400.00, 2400.00]
      },
      "right": {
        "short_name": "Uber Eats",
        "raw_names": ["Uber Eats"],
        "description": "sells food delivery from restaurants via the Uber platform",
        "amounts": [35.20, 18.90]
      }
    },
    {
      "match_id": "CVS-01",
      "left": {
        "short_name": "CVS Pharmacy",
        "raw_names": [
          "CVS/PHARMACY #12345 BOSTON MA",
          "CVS PHARMACY BOSTON MA",
          "CVS"
        ],
        "description": "A pharmacy and retail store selling prescriptions, health, and convenience items.",
        "amounts": [12.99, 24.50, 8.75]
      },
      "right": {
        "short_name": "CVS Pharmacy Payment",
        "raw_names": ["CVS/PHARMACY REF# 789012 BOSTON MA Card ending in 4412"],
        "description": "payment to CVS Pharmacy with reference number 789012",
        "amounts": [12.99]
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
