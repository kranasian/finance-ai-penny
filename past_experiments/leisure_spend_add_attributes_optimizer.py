from google import genai
from google.genai import types
import os
import json
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SCHEMA = types.Schema(
  type=types.Type.ARRAY,
  items=types.Schema(
    type=types.Type.OBJECT,
    properties={
      "id": types.Schema(
        type=types.Type.NUMBER,
        description="The unique identifier for the establishment.",
      ),
      "primary": types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(type=types.Type.STRING),
        description="For entertainment- or travel-related rows: at least one of Logistics, Shows, Attractions, Sports, Nightlife, Relaxation, Movies, Apps, Gaming, Literature, Crafts and Hobbies, Gear. Non-leisure rows: [].",
      ),
      "secondary": types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(type=types.Type.STRING),
        description="For leisure rows: 3-7 specific descriptors grounded in name and description. Non-leisure rows: [].",
      ),
    },
    required=["id", "primary", "secondary"]
  )
)

SYSTEM_PROMPT = """Transform each establishment into attribute JSON: one object per input `id` (copy `id` unchanged). Output a JSON array only.

## Source material only
Use **only** `name` and `description` — no outside knowledge. Every `primary` and `secondary` tag must trace to the source.

## Classification
Default **entertainment- or travel-related** unless the purchase cannot connect to Entertainment (recreation, streaming, events, games, literature, hobbies, nightlife) or Travel & Vacations (hotels, flights, touring, excursions, visas, travel gear). Non-leisure only when impossible (salary, utilities, fuel, insurance, pure groceries/dining): `"primary": []`, `"secondary": []`.

## `primary` (leisure rows)
At least one per row; multiples only when the source clearly spans categories. Pick **only** when supported:

- **Logistics**: Functional infrastructure and travel essentials (eg. flights, car rentals, trains, hotels, Airbnbs, travel insurance, passports, visas).
- **Shows**: Ticketed live performances and scheduled events (eg. concerts, festivals, theater, live performances).
- **Attractions**: Entry to points of interest or curated experiences (eg. zoos, museums, theme parks, guided city tours, scuba charters).
- **Sports**: Physical participation and active recreation venues (eg. bowling alleys, ski resorts, gyms, golf courses).
- **Nightlife**: Adult-oriented social consumption and evening entertainment (eg. bars, nightclubs, liquor stores, cannabis dispensaries, lounges).
- **Relaxation**: Physical rejuvenation and mental calm (eg. day spas, massage therapy centers, saunas, hot springs).
- **Movies**: The film-viewing experience across all platforms. Includes cinema tickets, theater concessions, and digital video streaming services.
- **Apps**: Recurring fees or one-time purchases for non-gaming, non-movie, and non-literature digital services. Includes music streaming, productivity tools, and utility applications.
- **Gaming**: Digital and interactive software, hardware, or services (eg. video game purchases, online multiplayer subscriptions, in-game content).
- **Literature**: Purchase or subscription of written content (eg. physical bookstores, e-book subscriptions, newsstands, magazines).
- **Crafts and Hobbies**: Materials and retailers for personal projects and skill-based interests (eg. art supply stores, musical instrument shops, photography equipment).
- **Gear**: Durable physical goods for specific leisure pursuits (eg. camping equipment, luggage, specialized technical apparel).

## `secondary` (leisure rows; 3–7)
Tags cluster this merchant with similar leisure merchants.

**Grounding:** quote or paraphrase `name` and `description`; infer products or activities only when the text reasonably supports them. Do not use the merchant name as a tag.

**Bare generics:** fail only when the **entire tag** is one undescribed word (bare "Ticket", "Service", "Subscription", "Good"). Multi-word descriptors pass ("Concert Ticket", "Developer Service", "Public Transportation", "Administrative fee"). Tags with a space skip bare-generic checks — never split a tag into words.

**Primary reuse:** no secondary tag with identical spelling to a primary label; compounds pass ("Japanese Literature"). Keep tags unique.

**Sparse text:** still provide 3–7 tags by splitting distinct phrases from the description into grounded labels.

Output JSON only."""


def _source_blob(establishment: dict) -> str:
  return f"{establishment.get('name', '')} {establishment.get('description', '')}".lower()


def _primary_grounded_in_source(primary: str, establishment: dict) -> bool:
  if primary not in ALLOWED_PRIMARIES:
    return False
  source = _source_blob(establishment)
  return any(hint in source for hint in _PRIMARY_GROUNDING_HINTS.get(primary, ()))


def _primaries_from_source(establishment: dict) -> list[str]:
  return [p for p in ALLOWED_PRIMARIES_LIST if _primary_grounded_in_source(p, establishment)]


def _content_words(tag: str) -> list[str]:
  return [w for w in re.findall(r"[a-z0-9]+", tag.lower()) if len(w) >= 3]


def _tag_grounded_in_source(tag: str, establishment: dict) -> bool:
  source = _source_blob(establishment)
  words = _content_words(tag)
  if not words:
    return False
  return any(word in source for word in words)


def _phrase_tags_from_description(text: str, limit: int = 5) -> list[str]:
  text = text.strip().rstrip(".")
  if not text or re.fullmatch(r"establishment undetermined", text, flags=re.I):
    return []
  for prefix in (r"sells\s+", r"offers\s+", r"providing\s+", r"payment for\s+", r"booking for\s+"):
    match = re.search(prefix + r"(.+)", text, flags=re.I)
    if match:
      text = match.group(1)
      break
  segments = re.split(r"[,;]|\band\b", text, flags=re.I)
  tags: list[str] = []
  for segment in segments:
    cleaned = segment.strip()
    if len(cleaned) < 3:
      continue
    words = cleaned.split()
    if len(words) > 5:
      cleaned = " ".join(words[:5])
    tag = cleaned[0].upper() + cleaned[1:]
    if tag not in tags:
      tags.append(tag)
  return tags[:limit]


def _tag_allowed(tag: str, primary: list[str], establishment: dict) -> bool:
  if not tag or not tag.strip() or len(tag.split()) > 4:
    return False
  if tag in primary:
    return False
  lowered = tag.lower()
  if lowered in _ALLOWED_PRIMARY_LOWER:
    return False
  name_lower = establishment.get("name", "").lower()
  if lowered == name_lower:
    return False
  if not _tag_grounded_in_source(tag, establishment):
    return False
  return True


def _sanitize_row(row: dict, establishment: dict) -> dict:
  raw_secondary = [t for t in row.get("secondary", []) if isinstance(t, str)]
  primary = [p for p in row.get("primary", []) if _primary_grounded_in_source(p, establishment)]
  if not primary:
    primary = _primaries_from_source(establishment)[:2]
  if not primary:
    return {"id": row["id"], "primary": [], "secondary": []}

  secondary: list[str] = []
  seen: set[str] = set()
  for tag in raw_secondary:
    tag = tag.strip()
    if not _tag_allowed(tag, primary, establishment):
      continue
    key = tag.lower()
    if key in seen:
      continue
    seen.add(key)
    secondary.append(tag)

  if len(secondary) < 3:
    for phrase in _phrase_tags_from_description(establishment.get("description", "")):
      if len(secondary) >= 7:
        break
      if _tag_allowed(phrase, primary, establishment) and phrase.lower() not in seen:
        secondary.append(phrase)
        seen.add(phrase.lower())

  if len(secondary) < 3:
    for phrase in _phrase_tags_from_description(establishment.get("name", "")):
      if len(secondary) >= 7:
        break
      if _tag_allowed(phrase, primary, establishment) and phrase.lower() not in seen:
        secondary.append(phrase)
        seen.add(phrase.lower())

  return {"id": row["id"], "primary": primary, "secondary": secondary[:7]}


def _sanitize_attributes(establishments: list, rows: list) -> list:
  by_id = {est["id"]: est for est in establishments}
  sanitized = []
  seen_ids: set[int] = set()
  for row in rows:
    est = by_id.get(row.get("id"))
    if est is None:
      continue
    sanitized.append(_sanitize_row(row, est))
    seen_ids.add(row["id"])
  for est in establishments:
    if est["id"] not in seen_ids:
      sanitized.append(_sanitize_row({"id": est["id"], "primary": [], "secondary": []}, est))
  return sanitized


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
    return _sanitize_attributes(establishments, self._generate_raw(establishments))

  def _generate_raw(self, establishments: list) -> list:
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
