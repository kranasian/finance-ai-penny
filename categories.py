import numpy as np

TOP_LEVEL_CATEGORY_MAP = {
  41: 'Food',
  42: 'Others',
  43: 'Bills',
  44: 'Shopping',
  46: 'Income',
}
_TOP_LEVEL_CATEGORY_NAME_TO_ID = { value: key for key, value in TOP_LEVEL_CATEGORY_MAP.items() }

_CATEGORY_ID_TO_NAME = {
    -1: 'Uncategorized',
    1: 'Meals',
    2: 'Dining Out',
    3: 'Delivered Food',
    4: 'Groceries',
    5: 'Leisure',
    6: 'Entertainment',
    7: 'Travel & Vacations',
    9: 'Bills',
    10: 'Connectivity',
    11: 'Insurance',
    12: 'Taxes',
    13: 'Service Fees',
    14: 'Shelter',
    15: 'Home',
    16: 'Utilities',
    17: 'Upkeep',
    18: 'Education',
    19: 'Kids Activities',
    20: 'Tuition',
    21: 'Shopping',
    22: 'Clothing',
    23: 'Gadgets',
    24: 'Kids',
    8: 'Pets',
    25: 'Transport',
    26: 'Car & Fuel',
    27: 'Public Transit',
    28: 'Health',
    29: 'Medical & Pharmacy',
    30: 'Gym & Wellness',
    31: 'Personal Care',
    32: 'Donations & Gifts',
    33: 'Miscellaneous',
    41: 'Food',
    42: 'Others',
    43: 'Bills',
    44: 'Shopping',
    45: 'Transfer',
    46: 'Income',
    47: 'Income',
    36: 'Salary',
    37: 'Side-Gig',
    38: 'Business',
    39: 'Interest',
}
_CATEGORY_NAME_TO_ID = { value.lower(): key for key, value in _CATEGORY_ID_TO_NAME.items() }

from typing import Optional

def get_category_id(category_name: str) -> Optional[int]:
  return _CATEGORY_NAME_TO_ID.get(category_name.lower(), None)

_CATEGORY_ID_TO_LARAVEL_NAME = {
    41: "Food",
    42: "Others",
    43: "Bills",
    44: "Shopping",
    46: "Income",
    1 : "Meals",
    2 : "Meals: Dining Out",
    3 : "Meals: Delivered Food",
    4 : "Meals: Groceries",
    5 : "Leisure",
    6 : "Leisure: Entertainment",
    7 : "Leisure: Travel & Vacations",
    9 : "Bills",
    10: "Bills: Connectivity",
    11: "Bills: Insurance",
    12: "Bills: Taxes",
    13: "Bills: Service Fees",
    14: "Shelter",
    15: "Shelter: Home",
    16: "Shelter: Utilities",
    17: "Shelter: Upkeep",
    18: "Education",
    19: "Education: Kids Activities",
    20: "Education: Tuition",
    21: "Shopping",
    22: "Shopping: Clothing",
    23: "Shopping: Gadgets",
    24: "Shopping: Kids",
    8 : "Shopping: Pets",
    25: "Transport",
    26: "Transport: Car & Fuel",
    27: "Transport: Public Transit",
    28: "Health",
    29: "Health: Medical & Pharmacy",
    30: "Health: Gym & Wellness",
    31: "Health: Personal Care",
    32: "Donations & Gifts",
    45: "Transfer",
    47: "Income",
    36: "Income: Salary",
    37: "Income: Side-Gig",
    38: "Income: Business",
    39: "Income: Interest",
    -1: "Uncategorized",
    33: "Miscellaneous",
}

def get_laravel_name(category_id: int) -> Optional[str]:
  return _CATEGORY_ID_TO_LARAVEL_NAME.get(category_id, None)

_TOP_LEVEL_CATEGORIES = [
    41,
    42,
    43,
    44,
    46,
]

_TOP_LEVEL_TO_PARENT_CATEGORIES = {
    41: [1, 2, 3, 4,],
    # 45: Transfer is intentionally not included here as its not included in the budget
    42: [5, 6, 7, 18, 19, 25, 27, 28, 29, 30, 31, 32, 33, -1],
    43: [9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 26,],
    44: [21, 22, 23, 24, 8,],
    46: [47, 36, 37, 38, 39,],
}

_PARENT_CATEGORIES = [
    -1,
    1,
    5,
    9,
    14,
    18,
    21,
    25,
    28,
    32,
    45,
    47,
]

_PARENT_TO_LEAF_CATEGORIES = {
    -1: [-1, ],
    1: [1, 2, 3, 4],
    5: [5, 6, 7,],
    9: [9, 10, 11, 12, 13,],
    14: [14, 15, 16, 17,],
    18: [18, 19, 20,],
    21: [21, 22, 23, 24, 8,],
    25: [25, 26, 27,],
    28: [28, 29, 30, 31,],
    32: [32,],
    33: [33,],
    45: [45,],
    47: [36, 37, 38, 39,],
}

INCOME_CATEGORY_IDS = [47, 46, 36, 37, 38, 39,]

DISCRETIONARY_CATEGORY_IDS = [44, 21, 22, 23, 24, 8, 41, 1, 2, 3, 4,]

def get_leaves_ids() -> list[int]:
  leaves = []
  for parent_category_id in _PARENT_TO_LEAF_CATEGORIES:
    leaves_list = _PARENT_TO_LEAF_CATEGORIES[parent_category_id]
    if len(leaves_list) == 1:
      leaves.append(leaves_list[0])
    else:
      leaves.extend(leaves_list[1:])
  return leaves

def get_name(category_id: int) -> Optional[str]:
  return _CATEGORY_ID_TO_NAME.get(category_id, None)

from typing import Optional

def get_category_id(category_name: str) -> Optional[int]:
  return _CATEGORY_NAME_TO_ID.get(category_name.lower(), None)

def get_all_leaf_as_dict_categories() -> dict[str, list[int]]:
  leaf_categories = {}
  for parent_category in _PARENT_TO_LEAF_CATEGORIES:
    for leaf_category in _PARENT_TO_LEAF_CATEGORIES[parent_category]:
      if leaf_category != int(parent_category):
        leaf_categories[leaf_category] = [leaf_category]
  return leaf_categories

def get_all_parent_categories():
  return _PARENT_CATEGORIES

def get_parents_with_leaves_as_dict_categories():
  return _PARENT_TO_LEAF_CATEGORIES

def get_top_level_with_leaves_as_dict_categories():
  return _TOP_LEVEL_TO_PARENT_CATEGORIES

def get_top_level_categories():
  return _TOP_LEVEL_CATEGORIES

def get_mapped_top_level_categories():
  """
  Example: {'Food': [1, 2, 3, 4], ... }\n
  Returns:
      dict: A dictionary mapping top-level to parent-categories.
  """
  mapped_categories = {}
  for category, label in TOP_LEVEL_CATEGORY_MAP.items():
    if category in _TOP_LEVEL_TO_PARENT_CATEGORIES:
      mapped_categories[label] = _TOP_LEVEL_TO_PARENT_CATEGORIES[category]
  return mapped_categories

def get_top_level_category_id(category_id: int) -> int:
  """
  Get the top-level category ID for a given parent category ID.
  Args:
      category_id: The parent category ID to look up
  Returns:
      int: The corresponding top-level category ID
  """
  for top_id, parent_ids in _TOP_LEVEL_TO_PARENT_CATEGORIES.items():
    if category_id in parent_ids:
      return top_id
  return None

_EMBEDDING_MAP = {}

def get_embedding(text: str) -> list[float]:
  global _EMBEDDING_MAP
  if len(_EMBEDDING_MAP.keys()) == 0:
    _EMBEDDING_MAP = np.load("penny/grounders/categories_expansions.npy", allow_pickle=True).item()

  sanitized_text = text.lower().strip()
  if sanitized_text not in _EMBEDDING_MAP:
    print(f"Category {sanitized_text} not found in embeddings")
    return None
  return _EMBEDDING_MAP[sanitized_text]


CATEGORIES_MAP = {
  -1: {
    'sort_key': 570,
    'noun': 'uncategorized spending',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': [],
    'secondary_expansions': [],
    'attributes': ["skip_weekly_finding", "skip_monthly_finding", "skip_week_month_spend_notify"],
  },
  1: {
    'sort_key': 310,
    'noun': 'groceries and eating out',
    'noun_plural': True,
    'negate_amount': False,
    'primary_expansions': [],
    'secondary_expansions': [],
    'attributes': ["skip_weekly_finding", "skip_monthly_finding"],
  },
  2: {
    'sort_key': 311,
    'noun': 'eating out',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': ["diners, pubs, and fast-food", "restaurants & coffee shops", ],
    'secondary_expansions': ["leisure food and non-grocery food", "date night dining", "social eating and celebrations", "baby food and snacks", "bread shop & patisserie",],
    'attributes': [],
  },
  3: {
    'sort_key': 312,
    'noun': 'delivered food',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': ["food delivery service and apps", "virtual kitchen & online food orders",],
    'secondary_expansions': ["takeout and delivery costs", ],
    'attributes': [],
  },
  4: {
    'sort_key': 313,
    'noun': 'groceries',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': ["cooking supplies and ingredients", "groceries & supermarket",],
    'secondary_expansions': ["meat, poultry, produce and frozen foods", "fruits and vegetables", "pantry staples, snacks and beverages",],
    'attributes': [],
  },
  5: {
    'sort_key': 510,
    'noun': 'leisure activities',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': [],
    'secondary_expansions': [],
    'attributes': [],
  },
  6: {
    'sort_key': 511,
    'noun': 'entertainment',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': ["indoor and outdoor entertainment & recreation", "movies, concerts and event tickets",],
    'secondary_expansions': ["live performance & concerts", "festivals, theme parks and interactive entertainment venues", "streaming services and games", "alcohol, cannabis and cigarettes", "fiction books & magazines and other literature", "personal hobbies and crafts supplies", ],
    'attributes': [],
  },
  7: {
    'sort_key': 512,
    'noun': 'travel and vacation',
    'noun_plural': True,
    'negate_amount': False,
    'primary_expansions': ["activity & excursion fund", "trip insurance and incidentals", "travel & vacations",],
    'secondary_expansions': ["excursions and trip stash", "sightseeing and on-the-road spending", "cultural immersion, relaxation and rejuvenation trips", "passport & visa fees", "journey and exploration gear",],
    'attributes': [],
  },
  9: {
    'sort_key': 220,
    'noun': 'bills',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': [],
    'secondary_expansions': [],
    'attributes': [],
  },
  10: {
    'sort_key': 221,
    'noun': 'connectivity spending',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': ["phone and internet bills", "satellite and other connectivity",],
    'secondary_expansions': ["phone and mobile plan", "internet costs & mobile data", "social media spending",],
    'attributes': [],
  },
  11: {
    'sort_key': 222,
    'noun': 'insurance spending',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': ["life insurance and other insurance", "business insurance",],
    'secondary_expansions': [],
    'attributes': ["skip_weekly_finding", "skip_monthly_finding", "skip_week_month_spend_notify"],
  },
  12: {
    'sort_key': 223,
    'noun': 'tax spending',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': ["local, state and federal taxes", "business taxes and penalties",],
    'secondary_expansions': [],
    'attributes': ["skip_weekly_finding", "skip_monthly_finding", "skip_week_month_spend_notify"],
  },
  13: {
    'sort_key': 224,
    'noun': 'service fee spending',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': ["professional service fees and administrative costs",],
    'secondary_expansions': ["personal assistant and secretariat services", "laundry & household services", ],
    'attributes': ["skip_weekly_finding", "skip_week_month_spend_notify"],
  },
  14: {
    'sort_key': 210,
    'noun': 'shelter spending',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': [],
    'secondary_expansions': [],
    'attributes': [],
  },
  15: {
    'sort_key': 211,
    'noun': 'rent or mortgage',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': ["homeowners insurance & property tax", "mortgage or rentals",],
    'secondary_expansions': ["home association dues and county tax",],
    'attributes': ["skip_weekly_finding"],
  },
  16: {
    'sort_key': 212,
    'noun': 'utilities',
    'noun_plural': True,
    'negate_amount': False,
    'primary_expansions': ["water, electric and gas utilities",],
    'secondary_expansions': ["water & electricity", "natural gas billings", "sewage maintenance costs",],
    'attributes': [],
  },
  17: {
    'sort_key': 213,
    'noun': 'home upkeep',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': ["home improvement & home repair services", "gardening, home cleaning and hvac upkeep",],
    'secondary_expansions': ["furnitures and appliances", "bedroom and furnishings",],
    'attributes': [],
  },
  18: {
    'sort_key': 530,
    'noun': 'education spending',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': [],
    'secondary_expansions': [],
    'attributes': [],
  },
  19: {
    'sort_key': 531,
    'noun': 'kids activities',
    'noun_plural': True,
    'negate_amount': False,
    'primary_expansions': ["youth education, recreation and after-school activities", "sports and extra-curricular classes",],
    'secondary_expansions': ["summer camps and sports events", "educational child care and lessons", "summer camps and sports events", ],
    'attributes': [],
  },
  20: {
    'sort_key': 241,
    'noun': 'tuition spending',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': ["private and college tuition and lodging", "school supplies and academic fees", ],
    'secondary_expansions': ["testing/examination, enrollment and registration fees", "school textbooks, course materials and supplies", "online learning or tutoring", "higher education funds", ],
    'attributes': [],
  },
  21: {
    'sort_key': 410,
    'noun': 'shopping spending',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': [],
    'secondary_expansions': [],
    'attributes': ["skip_weekly_finding", "skip_monthly_finding"],
  },
  22: {
    'sort_key': 411,
    'noun': 'clothing spending',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': ["clothing, shoes, fashion and jewelery", "attire, wardrobe shopping and accessories",],
    'secondary_expansions': ["seasonal winter or summer clothes and rentals", "undergarments and hats",],
    'attributes': [],
  },
  23: {
    'sort_key': 412,
    'noun': 'gadget shopping',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': ["phones, gadgets and electronic devices", "cameras, drones and tech devices",],
    'secondary_expansions': ["electronic device rental and repair", "laptops, camera and accessories", "speakers, headphones and smart devices", "sleep and fitness trackers and electronic equipment",],
    'attributes': ["skip_weekly_finding", "skip_week_month_spend_notify"],
  },
  24: {
    'sort_key': 413,
    'noun': 'kids shopping',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': ["kids clothing, shoes and kids fashion", "kids toys and games", ],
    'secondary_expansions': ["infant clothing, diapers and sanitary items", ],
    'attributes': [],
  },
  8: {
    'sort_key': 414,
    'noun': 'pet costs',
    'noun_plural': True,
    'negate_amount': False,
    'primary_expansions': ["pets and animal food & supplies", "veterinarian and pet insurance costs",],
    'secondary_expansions': ["pet clothing, toys and accessories", "pet daycare, walkers and pet services",],
    'attributes': [],
  },
  25: {
    'sort_key': 540,
    'noun': 'transportation spending',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': [],
    'secondary_expansions': [],
    'attributes': [],
  },
  26: {
    'sort_key': 541,
    'noun': 'car spending',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': ["car insurance, upkeep, maintenance and repairs", "car fuel, gasoline/diesel or charging", "car parking and tolls",],
    'secondary_expansions': ["auto insurance, licensing, registration and fees", "car accessories & modifications", "electric vehicle charging fees",],
    'attributes': [],
  },
  27: {
    'sort_key': 231,
    'noun': 'transit cost',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': ["commute and public transit", "taxis, ubers and ride hailing apps",],
    'secondary_expansions': ["bus, taxi, subway and metro fares", "shuttle and commute passes",],
    'attributes': [],
  },
  28: {
    'sort_key': 550,
    'noun': 'health spending',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': [],
    'secondary_expansions': [],
    'attributes': ["skip_weekly_finding", "skip_week_month_spend_notify"],
  },
  29: {
    'sort_key': 551,
    'noun': 'medical and pharmacy',
    'noun_plural': True,
    'negate_amount': False,
    'primary_expansions': ["doctors, hospital and ambulance fees", "health, vision and dental insurance and appointment copays", "pharmacy and drug copays and over-the-counter medicine",],
    'secondary_expansions': ["doctor consultation, hospital fees and health insurance", "maintenance medications, therapy, and counselling", "hospital, oral and eye care", "diagnostic tests, physicals and mental health", ],
    'attributes': ["skip_weekly_finding", "skip_week_month_spend_notify"],
  },
  30: {
    'sort_key': 552,
    'noun': 'gym and wellness spending',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': ["fitness gym and spa memberships", "workout classes, pilates and yoga sessions",],
    'secondary_expansions': ["personal trainors and wellness services", "retreats, spas and saunas",],
    'attributes': [],
  },
  31: {
    'sort_key': 553,
    'noun': 'personal care spending',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': ["hygiene, grooming and cosmetics", "hair-cuts, manicures", ],
    'secondary_expansions': ["waxing, tanning salons and make-up grooming", "cosmetic enhancements",],
    'attributes': [],
  },
  32: {
    'sort_key': 560,
    'noun': 'donations and gifts',
    'noun_plural': True,
    'negate_amount': False,
    'primary_expansions': ["gifts, donations and tokens", "holiday fund-raisers and sponsorships",],
    'secondary_expansions': ["celebratory presents, tokens to others", "charities, fundraisers and religious contributions",],
    'attributes': ["skip_weekly_finding", "skip_week_month_spend_notify"],
  },
  33: {
    'sort_key': 570,
    'noun': 'miscellaneous expense',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': ["--matches nothing--",],
    'secondary_expansions': [],
    'attributes': [],
  },
  36: {
    'sort_key': 111,
    'noun': 'salary',
    'noun_plural': False,
    'negate_amount': True,
    'primary_expansions': ["salary or regular hourly wage",],
    'secondary_expansions': ["part-time work",],
    'attributes': [],
  },
  37: {
    'sort_key': 112,
    'noun': 'side-gig income',
    'noun_plural': False,
    'negate_amount': True,
    'primary_expansions': ["freelance work or extra income",],
    'secondary_expansions': ["online selling, tutoring income",],
    'attributes': [],
  },
  38: {
    'sort_key': 113,
    'noun': 'business income',
    'noun_plural': False,
    'negate_amount': True,
    'primary_expansions': ["business profits, income and spending",],
    'secondary_expansions': ["business compliance and professional services", "business operating costs",],
    'attributes': ["skip_weekly_finding", "skip_week_month_spend_notify"],
  },
  39: {
    'sort_key': 114,
    'noun': 'interest income',
    'noun_plural': False,
    'negate_amount': True,
    'primary_expansions': ["savings and investment interest appreciation",],
    'secondary_expansions': ["dividends, capital gains and interest distributions",],
    'attributes': [],
  },
  41: {
    'sort_key': 300,
    'noun': 'food expense',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': [],
    'secondary_expansions': [],
    'attributes': [],
  },
  42: {
    'sort_key': 500,
    'noun': 'other expenses',
    'noun_plural': True,
    'negate_amount': False,
    'primary_expansions': [],
    'secondary_expansions': [],
    'attributes': ["skip_weekly_finding", "skip_monthly_finding", "skip_week_month_spend_notify", "skip_urgency_score"],
  },
  43: {
    'sort_key': 200,
    'noun': 'bills',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': [],
    'secondary_expansions': [],
    'attributes': ["skip_weekly_finding", "skip_monthly_finding", "skip_week_month_spend_notify", "skip_urgency_score"],
  },
  44: {
    'sort_key': 400,
    'noun': 'shopping expense',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': [],
    'secondary_expansions': [],
    'attributes': [],
  },
  45: {
    'sort_key': 600,
    'noun': 'transfers',
    'noun_plural': False,
    'negate_amount': False,
    'primary_expansions': [],
    'secondary_expansions': [],
    'attributes': ["skip_weekly_finding", "skip_monthly_finding", "skip_week_month_spend_notify", "skip_forecast"],
  },
  46: {
    'sort_key': 100,
    'noun': 'income',
    'noun_plural': False,
    'negate_amount': True,
    'primary_expansions': [],
    'secondary_expansions': [],
    'attributes': [],
  },
  47: {
    'sort_key': 110,
    'noun': 'income',
    'noun_plural': False,
    'negate_amount': True,
    'primary_expansions': [],
    'secondary_expansions': [],
    'attributes': ["skip_weekly_finding", "skip_monthly_finding"],
  },
}


def get_primary_expansions(leaf_id):
  category_obj = CATEGORIES_MAP.get(leaf_id, None)
  if category_obj:
    return category_obj["primary_expansions"] if category_obj["primary_expansions"] else [category_obj.get("noun", None)]
  return []

def get_secondary_expansions(leaf_id):
  category_obj = CATEGORIES_MAP.get(leaf_id, None)
  secondary = []
  if category_obj and category_obj["secondary_expansions"]:
    secondary = category_obj["secondary_expansions"]
  secondary.extend(get_primary_expansions(leaf_id))
  return secondary


def diff_category_json(l, dl, a_category_json: list, b_category_json: list, high_confidence_threshold: float = 0.41) -> float:
  """
  Calculate the difference between two category JSON objects.
  
  Args:
      a_category_json: List of dicts with 'id' and 'score' keys
      b_category_json: List of dicts with 'id' and 'score' keys  
      high_confidence_threshold: Threshold for high confidence scores (default 0.41)
  
  Returns:
      float: Difference score from 0.0 (perfect match) to 5.0 (completely different)
  """
  # Convert lists to dictionaries for easier lookup
  a_dict = {item['id']: item['score'] for item in a_category_json}
  b_dict = {item['id']: item['score'] for item in b_category_json}
  
  # Get all unique category IDs
  all_ids = set(a_dict.keys()) | set(b_dict.keys())
  
  if not all_ids:
    return 0.0  # Both empty, perfect match
  
  total_difference = 0.0
  total_weight = 0.0

  a_ids = [x["id"] for x in a_category_json]
  a_ids.sort()
  b_ids = [x["id"] for x in b_category_json]
  b_ids.sort()
  
  
  for category_id in all_ids:
    a_score = min(a_dict.get(category_id, 0.0), 0.50)
    b_score = min(b_dict.get(category_id, 0.0), 0.50)
    
    # Calculate weight based on the higher score (more important categories get more weight)
    weight = max(a_score, b_score)
    
    # Calculate difference for this category
    score_diff = abs(a_score - b_score)
    
    # If one side is missing this category entirely, apply stronger penalty
    if (a_score == 0.0 and b_score > 0.0) or (b_score == 0.0 and a_score > 0.0):
      present_score = max(a_score, b_score)
      # Apply much stronger penalties for missing categories
      if present_score < 0.05:
        # Very low scores get moderate penalty
        missing_penalty = present_score * 2.0
      elif present_score < 0.1:
        # Low scores get higher penalty
        missing_penalty = present_score * 3.0
      elif present_score < 0.15:
        # Medium-low scores get high penalty
        missing_penalty = present_score * 4.0
      elif present_score < 0.2:
        # Medium scores get very high penalty
        missing_penalty = present_score * 5.0
      elif present_score < high_confidence_threshold:
        # High scores get maximum penalty
        missing_penalty = present_score * 6.0
      else:
        # Very high scores get extreme penalty
        missing_penalty = present_score * 8.0
      score_diff = max(score_diff, missing_penalty)
    
    # Apply additional penalty for high-confidence mismatches
    if a_score >= high_confidence_threshold or b_score >= high_confidence_threshold:
      if score_diff > 0.05:  # Lower threshold for high-confidence categories
        score_diff *= 2.0  # Increased multiplier
    
    # Apply additional penalty for significant score differences
    if score_diff > 0.1:
      score_diff *= 1.3  # Boost significant differences
    
    total_difference += score_diff * weight
    total_weight += weight
  
  if total_weight == 0:
    return 0.0
  
  # Normalize the difference to 0-5 scale
  # Base normalization to 0-1, then scale to 0-5
  normalized_diff = total_difference / total_weight
  
  # Apply more aggressive non-linear scaling to make differences more pronounced
  # Use a more aggressive power function to map 0-1 to 0-5
  diff_score = 5.0 * (normalized_diff ** 0.5)  # Changed from 0.7 to 0.5 for more aggressive scaling
  
  return min(diff_score, 5.0)


CATEGORY_DESCRIPTIONS = [
  {
    "category": "Meals",
    "description": "all sources of food from supermarkets to restaurants and food deliveries",
    "children": [
      {
        "category": "Dining Out",
        "description": "eating out from restaurants or other food establishments, or buying prepared food",
      },
      {
        "category": "Delivered Food",
        "description": "food delivery apps prepared by restaurants.  Food Apps like Doordash, Grubhub UberEats, or delivered by the establishment itself",
      },
      {
        "category": "Groceries",
        "description": "items bought from convenience stores and supermarkets such as cooking materials and ingredients",
      }
    ]
  },
  {
    "category": "Leisure",
    "description": "all relaxation or recreation activities of the family.",
    "children": [
      {
        "category": "Entertainment",
        "description": "concerts, cable companies and watching movies and streaming services and other forms of entertainment.",
      },
      {
        "category": "Travel & Vacations",
        "description": "hotels, air fare and other spending categories like touring.",
      }
    ]
  },
  {
    "category": "Bills",
    "description": "essential payments for practical services and recurring costs.",
    "children": [
      {
        "category": "Connectivity",
        "description": "expenses for communication and internet, or services that let you connect with others.",
      },
      {
        "category": "Insurance",
        "description": "payments for financial protection, such as life insurance.",
      },
      {
        "category": "Taxes",
        "description": "obligatory contributions to the government such as income tax, state tax, business tax, etc.",
      },
      {
        "category": "Service Fees",
        "description": "payments for specific services rendered, like professional fees for lawyers or accountants, or the fees product or service.",
      }
    ]
  },
  {
    "category": "Shelter",
    "description": "all expenses needed for a place of residence and for happily living in it.",
    "children": [
      {
        "category": "Home",
        "description": "payments for rent, mortgage/debt, property tax, home insurance and other dues tied to permanent or temporary ownership of a place of residence",
      },
      {
        "category": "Utilities",
        "description": "payments to public utilities such as water, electricity, natural gas, sewage",
      },
      {
        "category": "Upkeep",
        "description": "expenses made for the purpose of maintaining the place of residence, including security or beautification and home improvement",
      }
    ]
  },
  {
    "category": "Education",
    "description": "all things for learning and development of the household",
    "children": [
      {
        "category": "Kids Activities",
        "description": "expenses for extra-curricular activities outside normal schooling, such as sports or after school care and camps.",
      },
      {
        "category": "Tuition",
        "description": "expenses for the necessary parts of schooling and recurring costs like dormitories",
      }
    ]
  },
  {
    "category": "Shopping",
    "description": "spending on discretionary purchases",
    "children": [
      {
        "category": "Clothing",
        "description": "spending on wearables such as clothes, shoes, accessories used everyday or for special occassions",
      },
      {
        "category": "Gadgets",
        "description": "technological devices, such as laptops, cameras, gaming consoles and phones",
      },
      {
        "category": "Kids",
        "description": "purchases made for kids, such as clothes, accessories, gadgets and toys.",
      },
      {
        "category": "Pets",
        "description": "expenses related to your pets, such as food, veterinarian fees, toys, grooming, boarding, and pet insurance",
      }
    ]
  },
  {
    "category": "Transport",
    "description": "expenditures related to moving from point a to point b",
    "children": [
      {
        "category": "Public Transit",
        "description": "expenses for riding trains, trams, cabs, buses",
      },
      {
        "category": "Car & Fuel",
        "description": "expenses for your own personal car including fuel, EV charging, and car maintenance such as oil change, tire change, car checkups and insurance",
      }
    ]
  },
  {
    "category": "Health",
    "description": "all things related to well-being both inside and out",
    "children": [
      {
        "category": "Medical & Pharmacy",
        "description": "includes health insurance payments, hospital/clinic visits, doctor fees, medications both curative and preventive",
      },
      {
        "category": "Gym & Wellness",
        "description": "activities relating to fitness, exercise and sports",
      },
      {
        "category": "Personal Care",
        "description": "services for personal grooming like getting a haircut, getting your nails done, eyelash threading, etc.",
      }
    ]
  },
  {
    "category": "Donations & Gifts",
    "description": "anything spent for the sake of others, like buying gifts, treating meals, donating to charities, etc."
  },

  {
    "category": "Transfers",
    "description": "movement of money from one account to another"
  },

  {
    "category": "Income",
    "description": "money earned from products or services sold, and returns on investments",
    "children": [
      {
        "category": "Salary",
        "description": "regular inflow of money from the primary source of income",
      },
      {
        "category": "Side-Gig",
        "description": "semi-regular inflow of money due to part-time work",
      },
      {
        "category": "Business",
        "description": "profits earned from owned businesses",
      },
      {
        "category": "Interest",
        "description": "appreciation of investments",
      }
    ]
  }
]