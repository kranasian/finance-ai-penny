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
        description="List containing at least one of: Event, Venue, Digital, Adult, Hobby, Media, Travel, Lodging. Borderline places can have multiple."
      ),
      "secondary": types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(type=types.Type.STRING),
        description="List of 2-5 HIGH-VALUE business categories or precise descriptors. MUST NOT contain: entertainment, recreation, leisure, activity, fun, travel, vacation."
      ),
    },
    required=["id", "primary", "secondary"]
  )
)

SYSTEM_PROMPT = """Task: Transform Leisure Establishment JSON to Attribute JSON utilizing standard business taxonomy.

Mapping Rules:
1. Copy `id`.
2. `primary`: List [1+ items]. Options: "Event", "Venue", "Digital", "Adult", "Hobby", "Media", "Travel", "Lodging".
   - Inference: "Concert/Game" -> "Event". "Theater/Park" -> "Venue". "Streaming/App" -> "Digital". "Bar/Casino" -> "Adult". "Crafts/Sports" -> "Hobby". "Book/Movie" -> "Media". "Flight/Train" -> "Travel". "Hotel/Resort" -> "Lodging".
   - IF borderline (e.g., music festival), INCLUDE ALL relevant (e.g., "Event", "Venue", "Media").
3. `secondary`: List [3-7 items]. Extract highly specific, high-value Google Places/Yelp-style category tags.
   - For Event: Extract specific Type (e.g. "Concert", "Sports Game", "Festival").
   - For Venue: Extract specific Type (e.g. "Theme Park", "Museum", "Arcade").
   - For Digital: Extract specific Service (e.g. "Streaming", "Video Game", "Subscription").
   - For Adult: Extract specific Activity (e.g. "Bar", "Nightclub", "Gambling").
   - For Hobby: Extract specific Activity (e.g. "Rock Climbing", "Knitting", "Painting").
   - For Media: Extract specific Genre/Format (e.g. "Sci-Fi", "Anime", "Novel").
   - For Travel: Extract specific Mode (e.g. "Airline", "Train", "Cruise").
   - For Lodging: Extract specific Type (e.g. "Hotel", "Resort", "Vacation Rental").
   - Sparse Descriptions: If info is sparse, infer high-value descriptive characteristics without generic nouns.
   - CRITICAL NEGATIVE CONSTRAINTS: Tags MUST NOT contain the following words (even as substrings):
     "entertainment", "leisure", "recreation", "activity", "fun", "place", "establishment", "service", "company".
   - ONLY use standard singular nouns for items, or established category names.
   - STANDALONE ENFORCEMENT: If a tag is a generic single word (e.g., "Ticket", "Rental", "Pass", "Admission", "Card", "Fee", "Subscription", "Apartment", "Reservation"), YOU MUST prepend a specific descriptor (e.g., "Airline ticket", "Car rental", "Day pass", "Music subscription", "Vacation apartment").

Definitions:
- `primary`: Select one or more from ["Event", "Venue", "Digital", "Adult", "Hobby", "Media", "Travel", "Lodging"].
- `secondary`: Extract 3-7 specific tags (Genre, Activity, Product, or Type) from the Name and Description.

Constraints:
- Use singular nouns for `secondary` tags.
- STRICTLY EXCLUDE standalone generic terms: "Entertainment", "Leisure", "Fun", "Activity", "Place", "Service", "App".
- Allow compound terms (e.g., "Video game", "Theme park").
- Infer attributes from the Name if relevant.
- MUST NOT repeat `primary` categories in `secondary`.
- MANDATORY: Ensure at least 3 `secondary` tags. If specific details are missing, infer broad categories (e.g., "Outdoor", "Indoor", "Social", "Solo").
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
  {"id": 101, "primary": ["Digital"], "secondary": ["Video Streaming", "Movie", "TV Series", "Subscription"]},
  {"id": 102, "primary": ["Venue"], "secondary": ["Cinema", "Movie"]},
  {"id": 103, "primary": ["Adult"], "secondary": ["Wine", "Beer", "Spirit", "Alcohol"]}
]
Key validations:
- Specific attributes for digital vs physical venues
- Adult category correctly identified"""
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
  {"id": 201, "primary": ["Hobby"], "secondary": ["Arts & Crafts", "Framing", "Decor", "DIY"]},
  {"id": 202, "primary": ["Venue"], "secondary": ["Theme Park", "Amusement Park", "Ride", "Attraction"]},
  {"id": 203, "primary": ["Media", "Hobby"], "secondary": ["Book", "Magazine", "Reading", "Literature"]}
]
Key validations:
- Distinction between hobby supplies and media
- Venue identification for theme parks"""
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
  {"id": 301, "primary": ["Event"], "secondary": ["Concert", "Sporting Event", "Ticket", "Live Music"]},
  {"id": 302, "primary": ["Digital"], "secondary": ["Video Game", "Software", "Gaming", "Esports"]},
  {"id": 303, "primary": ["Event"], "secondary": ["Music Festival", "Live Music", "Concert", "Festival"]}
]
Key validations:
- Event ticketing vs the event itself
- Digital gaming correctly categorized"""
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
  {"id": 401, "primary": ["Travel"], "secondary": ["Airline", "Flight", "Air Travel", "Transportation"]},
  {"id": 402, "primary": ["Lodging"], "secondary": ["Hotel", "Accommodation", "Stay", "Hospitality"]},
  {"id": 403, "primary": ["Travel", "Lodging"], "secondary": ["Travel Agency", "Booking", "Flight", "Hotel"]},
  {"id": 404, "primary": ["Travel", "Lodging"], "secondary": ["Cruise", "Vacation", "Sea Travel", "Ship"]}
]
Key validations:
- Travel and Lodging categories correctly identified
- Overlap handled for travel agencies and cruises"""
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
