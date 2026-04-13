from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SCHEMA = types.Schema(
  type=types.Type.ARRAY,
  items=types.Schema(
    type=types.Type.OBJECT,
    properties={
      "id": types.Schema(type=types.Type.NUMBER),
      "primary": types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(type=types.Type.STRING),
        description="List containing at least one of: Logistics, Shows, Attractions, Sports, Nightlife, Relaxation, Movies, Apps, Gaming, Literature, Crafts and Hobbies, Gear."
      ),
      "secondary": types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(type=types.Type.STRING),
        description="List of 3-7 HIGH-VALUE business categories or precise descriptors. Tags must define the establishment (e.g., 'Streaming Subscription' not just 'Subscription')."
      ),
    },
    required=["id", "primary", "secondary"]
  )
)

SYSTEM_PROMPT = """Task: Transform Leisure Establishment JSON to Attribute JSON utilizing standard business taxonomy.

Mapping Rules:
1. Copy `id`.
2. `primary`: List [1+ items]. MANDATORY: Every output must have at least one primary attribute. Options:
   - Logistics: Functional infrastructure and travel essentials (eg. flights, car rentals, trains, hotels, Airbnbs, travel insurance, passports, visas).
   - Shows: Ticketed live performances and scheduled events (eg. concerts, festivals, theater, live performances).
   - Attractions: Entry to points of interest or curated experiences (eg. zoos, museums, theme parks, guided city tours, scuba charters).
   - Sports: Physical participation and active recreation venues (eg. bowling alleys, ski resorts, gyms, golf courses).
   - Nightlife: Adult-oriented social consumption and evening entertainment (eg. bars, nightclubs, liquor stores, cannabis dispensaries, lounges).
   - Relaxation: Physical rejuvenation and mental calm (eg. day spas, massage therapy centers, saunas, hot springs).
   - Movies: The film-viewing experience across all platforms. Includes cinema tickets, theater concessions, and digital video streaming services.
   - Apps: Recurring fees or one-time purchases for non-gaming, non-movie, and non-literature digital services. Includes music streaming, productivity tools, and utility applications.
   - Gaming: Digital and interactive software, hardware, or services (eg. video game purchases, online multiplayer subscriptions, in-game content).
   - Literature: Purchase or subscription of written content (eg. physical bookstores, e-book subscriptions, newsstands, magazines).
   - Crafts and Hobbies: Materials and retailers for personal projects and skill-based interests (eg. art supply stores, musical instrument shops, photography equipment).
   - Gear: Durable physical goods for specific leisure pursuits (eg. camping equipment, luggage, specialized technical apparel).

3. `secondary`: List [3-7 items]. Extract highly specific, high-value category tags.

   ### CRITICAL TAG RULES:
   - RULE 1: SOURCE MATERIAL ONLY. Tags MUST be solely based on Name and Description provided. Do not use external knowledge.
   - RULE 2: DEFINE THE ESTABLISHMENT. Tags are only valid if they define what the establishment IS or DOES.
   - RULE 3: ABSOLUTELY NO STANDALONE GENERIC TERMS. Standalone generic words are STRICTLY FORBIDDEN. You MUST prepend a specific descriptor to these words:
     "Rental", "Pass", "Admission", "Card", "Fee", "Reservation", "Stay", "Booking", "Ticket", "Venue", "Travel", "Arena", "Concession", "Content", "Online", "Digital", "Service", "Subscription", "Round-trip".
     - BAD: "Ticket", "Food", "Travel"
     - GOOD: "Concert Ticket", "Travel Insurance"
   - RULE 4: ABSOLUTELY NO PRIMARY REUSE. Secondary tags MUST NOT contain any of the primary attribute names (e.g., if "Sports" is a primary attribute, the word "Sports" cannot appear in any secondary tag). Secondary attributes should not include any of the primary attribute options.
     - FORBIDDEN WORDS in secondary: Logistics, Shows, Attractions, Sports, Nightlife, Relaxation, Movies, Apps, Gaming, Literature, Crafts, Hobbies, Gear.
   - RULE 5: P2P PAYMENTS. For person-to-person payments (e.g., Venmo), only focus on the purpose (e.g., "Dinner", "Gift").
   - RULE 6: SPECIFICITY. eg. prefer "Basketball Game" over "Game".

Definitions:
- `primary`: Select one or more from the primary categories list.
- `secondary`: Extract 3-7 specific tags (Genre, Activity, Product, or Type) from the Name and Description.

Constraints:
- Use singular nouns for `secondary` tags where appropriate.
- Allow compound terms (e.g., "Video game", "Theme park").
- MUST NOT repeat `primary` categories in `secondary`.
- MANDATORY: Ensure at least 3 `secondary` tags, and at least 1 `primary` tag.
- Output JSON only."""

class LeisureSpendAttributesOptimizer:
  """Handles all Gemini API interactions for generating leisure establishment attributes based on establishment information"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for generating establishment attributes"""
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
    self.top_k = 40
    self.max_output_tokens = 4096
    self.response_mime_type = "application/json"
    
    # Safety Settings
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
    
    # System Prompt
    self.system_prompt = SYSTEM_PROMPT
    
    # Output Schema - array of result objects
    self.output_schema = SCHEMA
  
  def generate(self, establishments: list) -> list:
    """
    Generate attributes using Gemini API based on establishment information.
    
    Args:
      establishments: List of dictionaries, each containing id, name, and description.
      
    Returns:
      List of dictionaries, each containing id, primary, and secondary
    """
    # Create request text with the input structure
    request_text_str = f"""input: {json.dumps(establishments, indent=2)}
output: """
    
    print(f"\n{'='*80}")
    print("INPUT JSON:")
    print(json.dumps(establishments, indent=2))
    print("="*80)
    
    request_text = types.Part.from_text(text=request_text_str)
    
    # Create content and configuration
    contents = [types.Content(role="user", parts=[request_text])]
    
    generate_content_config = types.GenerateContentConfig(
      temperature=self.temperature,
      top_p=self.top_p,
      top_k=self.top_k,
      max_output_tokens=self.max_output_tokens,
      response_mime_type=self.response_mime_type,
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


TEST_CASES = [
  {
    "name": "Test 1: Basic entertainment types",
    "establishments": [
      {
        "id": 101,
        "name": "Netflix",
        "description": "monthly subscription for streaming movies and tv shows"
      },
      {
        "id": 102,
        "name": "AMC Theatres",
        "description": "movie theater."
      },
      {
        "id": 103,
        "name": "Total Wine & More",
        "description": "sells wine, spirits and beer"
      }
    ],
    "ideal_response": """[
  {"id": 101, "primary": ["Movies"], "secondary": ["Video Streaming", "TV Series", "Digital Subscription"]},
  {"id": 102, "primary": ["Movies"], "secondary": ["Cinema", "Film Screening", "Movie Concession"]},
  {"id": 103, "primary": ["Nightlife"], "secondary": ["Wine Shop", "Beer Retailer", "Spirit Store", "Alcohol Retail"]}
]"""
  },
  {
    "name": "Test 2: Basic entertainment types",
    "establishments": [
      {
        "id": 201,
        "name": "Michaels",
        "description": "sells art supplies, framing, and seasonal decor"
      },
      {
        "id": 202,
        "name": "Disneyland",
        "description": "Amusement park with rides and characters"
      },
      {
        "id": 203,
        "name": "Barnes & Noble",
        "description": "This establishment sells books, magazines, and gifts."
      }
    ],
    "ideal_response": """[
  {"id": 201, "primary": ["Crafts and Hobbies"], "secondary": ["Art Supplies", "Custom Framing", "Seasonal Decor", "DIY Materials"]},
  {"id": 202, "primary": ["Attractions"], "secondary": ["Theme Park", "Amusement Park", "Amusement Ride", "Tourist Attraction"]},
  {"id": 203, "primary": ["Literature"], "secondary": ["Bookstore", "Magazine Periodical", "Reading Material", "Gift Shop"]}
]"""
  },
  {
    "name": "Test 3: Multiple Establishments",
    "establishments": [
      {
        "id": 301,
        "name": "Ticketmaster",
        "description": "platform for buying concert and sports tickets"
      },
      {
        "id": 302,
        "name": "Steam",
        "description": "digital store for pc video games"
      },
      {
        "id": 303,
        "name": "Coachella",
        "description": "This is a payment for a music festival pass."
      }
    ],
    "ideal_response": """[
  {"id": 301, "primary": ["Shows", "Sports"], "secondary": ["Concert Ticket", "Sports Ticket", "Event Platform", "Live Performance"]},
  {"id": 302, "primary": ["Gaming"], "secondary": ["Video Game", "Digital Software", "Gaming Store", "PC Gaming"]},
  {"id": 303, "primary": ["Shows"], "secondary": ["Music Festival", "Live Music", "Festival Pass", "Concert Event"]}
]"""
  },
  {
    "name": "Test 4: Travel and Lodging",
    "establishments": [
      {
        "id": 401,
        "name": "United Airlines",
        "description": "Flight booking for vacation"
      },
      {
        "id": 402,
        "name": "Hilton Garden Inn",
        "description": "Hotel accommodation for 3 nights"
      },
      {
        "id": 403,
        "name": "Expedia",
        "description": "Online travel agency for booking flights and hotels"
      },
      {
        "id": 404,
        "name": "Royal Caribbean",
        "description": "Cruise line booking"
      }
    ],
    "ideal_response": """[
  {"id": 401, "primary": ["Logistics"], "secondary": ["Airline Flight", "Air Travel", "Flight Booking", "Transportation"]},
  {"id": 402, "primary": ["Logistics"], "secondary": ["Hotel Room", "Travel Accommodation", "Overnight Stay", "Hospitality"]},
  {"id": 403, "primary": ["Logistics"], "secondary": ["Travel Agency", "Online Booking", "Flight Ticket", "Hotel Reservation"]},
  {"id": 404, "primary": ["Logistics", "Attractions"], "secondary": ["Cruise Ship", "Sea Travel", "Ocean Voyage", "Guided Excursion"]}
]"""
  }
]


def test_with_inputs(test_name_or_index_or_dict, optimizer: LeisureSpendAttributesOptimizer = None):
  """
  Convenient method to test the optimizer with custom inputs.
  
  Args:
    test_name_or_index_or_dict: Test case index, name or dict (Wait, could be a list of establishments directly for backward compatibility)
    optimizer: Optional LeisureSpendAttributesOptimizer instance. If None, creates a new one.
    
  Returns:
    List of dictionaries, each containing id, primary, and secondary
  """
  if optimizer is None:
    optimizer = LeisureSpendAttributesOptimizer()
    
  test_case = None
  
  # Backward compatibility: handle a direct list of establishments
  if isinstance(test_name_or_index_or_dict, list):
    return optimizer.generate(test_name_or_index_or_dict)
    
  if isinstance(test_name_or_index_or_dict, dict):
    if "establishments" in test_name_or_index_or_dict:
      test_case = test_name_or_index_or_dict
  elif isinstance(test_name_or_index_or_dict, int):
    if 0 <= test_name_or_index_or_dict < len(TEST_CASES):
      test_case = TEST_CASES[test_name_or_index_or_dict]
  elif isinstance(test_name_or_index_or_dict, str):
    for tc in TEST_CASES:
      if tc["name"] == test_name_or_index_or_dict:
        test_case = tc
        break
        
  if not test_case:
    print(f"Test case '{test_name_or_index_or_dict}' not found.")
    return None


def run_test_set_1(optimizer: LeisureSpendAttributesOptimizer = None):
  return test_with_inputs([
    {
      "id": 1001,
      "name": "Delta Air Lines",
      "description": "Airline tickets for a round-trip flight to London."
    },
    {
      "id": 1002,
      "name": "Royal Caribbean International",
      "description": "Booking for a 7-night Caribbean cruise."
    },
    {
      "id": 1003,
      "name": "Hertz Car Rental",
      "description": "Rental car reservation for a weekend road trip."
    }
  ], optimizer)

def run_test_set_2(optimizer: LeisureSpendAttributesOptimizer = None):
  return test_with_inputs([
    {
      "id": 2001,
      "name": "Airbnb",
      "description": "Reservation for a vacation rental apartment in Paris."
    },
    {
      "id": 2002,
      "name": "Hilton Hotels & Resorts",
      "description": "Hotel stay for a family vacation."
    },
    {
      "id": 2003,
      "name": "KOA Campgrounds",
      "description": "Campsite reservation for an RV trip."
    }
  ], optimizer)

def run_test_set_3(optimizer: LeisureSpendAttributesOptimizer = None):
  return test_with_inputs([
    {
      "id": 3001,
      "name": "Madison Square Garden",
      "description": "Tickets for a basketball game."
    },
    {
      "id": 3002,
      "name": "San Diego Zoo",
      "description": "Admission tickets for the zoo."
    },
    {
      "id": 3003,
      "name": "Topgolf",
      "description": "Venue for golf games, food, and drinks."
    }
  ], optimizer)

def run_test_set_4(optimizer: LeisureSpendAttributesOptimizer = None):
  return test_with_inputs([
    {
      "id": 4001,
      "name": "Spotify",
      "description": "Monthly subscription for music streaming."
    },
    {
      "id": 4002,
      "name": "Kindle Unlimited",
      "description": "Subscription for ebook reading."
    },
    {
      "id": 4003,
      "name": "PlayStation Network",
      "description": "Purchase of digital games and online multiplayer access."
    }
  ], optimizer)

def main(batch: int = 1):
  """
  Main function to test the LeisureSpendAttributesOptimizer
  
  Args:
    batch: Batch number (1-4) to determine which tests to run
  """
  print(f"Testing LeisureSpendAttributesOptimizer - Batch {batch}\n")
  
  optimizer = LeisureSpendAttributesOptimizer()
  
  if batch == 1:
    run_test_set_1(optimizer)
  elif batch == 2:
    run_test_set_2(optimizer)
  elif batch == 3:
    run_test_set_3(optimizer)
  elif batch == 4:
    run_test_set_4(optimizer)
  else:
    raise ValueError("batch must be between 1 and 4")
  
  print("Test completed!")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run leisure spend attribute tests in batches')
  parser.add_argument('--batch', type=int, default=1, choices=[1, 2, 3, 4],
                      help='Batch number to run (1-4)')
  args = parser.parse_args()
  main(batch=args.batch)
