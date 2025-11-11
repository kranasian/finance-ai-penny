def get_params(kwargs):
  """Return the parameters from the kwargs."""
  return (kwargs.get("l"),
      kwargs.get("dl", {}),
      kwargs.get("debug_arr", []),
      kwargs.get("db_mconn"),
      kwargs.get("db_sconn"),
      kwargs.get("templates_df"),
      kwargs.get("user_id"),
      kwargs.get("fake_mutations", False))


_CATEGORY_TEXT_MAP = {
  "meals": 1,
  "meals_groceries": 4,
  "meals_dining_out": 2,
  "meals_delivered_food": 3,
  "leisure": 5,
  "leisure_entertainment": 6,
  "leisure_travel": 7,
  "bills": 9,
  "bills_connectivity": 10,
  "bills_insurance": 11,
  "bills_tax": 12,
  "bills_service_fees": 13,
  "shelter": 14,
  "shelter_home": 15,
  "shelter_utilities": 16,
  "shelter_upkeep": 17,
  "education": 18,
  "education_kids_activities": 19,
  "education_tuition":  20,
  "shopping": 21,
  "shopping_clothing": 22,
  "shopping_gadgets": 23,
  "shopping_kids": 24,
  "shopping_pets": 8,
  "transportation": 25,
  "transportation_public": 27,
  "transportation_car": 26,
  "health": 28,
  "health_medical_pharmacy": 29,
  "health_gym_wellness": 30,
  "health_personal_care": 31,
  "donations_gifts": 32,
  "income": 47,
  "income_salary": 36,
  "income_sidegig": 37,
  "income_business": 38,
  "income_interest": 39,
  "uncategorized": -1,
  "transfers": 45,
  "miscellaneous": 33,
}
_CATEGORY_ID_MAP = {v: k for k, v in _CATEGORY_TEXT_MAP.items()}

_ALL_CATEGORY_TEXT_MAP = {
  "top_income": 46,
  "top_bills": 43,
  "top_meals": 41,
  "top_shopping": 44,
  "others": 42
}
_ALL_CATEGORY_TEXT_MAP.update(_CATEGORY_TEXT_MAP)
_ALL_CATEGORY_ID_MAP = {v: k for k, v in _ALL_CATEGORY_TEXT_MAP.items()}

def to_category_id(category: str) -> int:
  return _CATEGORY_TEXT_MAP[category]

def to_category_name(category_id: int) -> str:
  if category_id is None:
    return "uncategorized"
  try:
    return _CATEGORY_ID_MAP[category_id]
  except KeyError:
    return "uncategorized"



def to_all_category_id(category: str) -> int:
  return _ALL_CATEGORY_TEXT_MAP[category]

def to_all_category_name(category_id: int) -> str:
  return _ALL_CATEGORY_ID_MAP[category_id]


OUTPUT_CATEGORY_TEXT_MAP = {
    "meals": "meals",
    "meals_groceries": "groceries",
    "meals_dining_out": "dining out",
    "meals_delivered_food": "delivered food",
    "leisure": "leisure",
    "leisure_entertainment": "entertainment",
    "leisure_travel": "travel",
    "bills": "bills",
    "bills_connectivity": "connectivity",
    "bills_insurance": "insurance",
    "bills_tax": "tax",
    "bills_service_fees": "service fees",
    "shelter": "overall shelter",
    "shelter_home": "rent/mortgage",
    "shelter_utilities": "utilities",
    "shelter_upkeep": "home upkeep",
    "education": "education",
    "education_kids_activities": "kids activities",
    "education_tuition": "tuition",
    "shopping": "all shopping",
    "shopping_clothing": "clothing",
    "shopping_gadgets": "gadgets",
    "shopping_kids": "kids shopping",
    "shopping_pets": "pets costs",
    "transportation": "transportation",
    "transportation_public": "public transit",
    "transportation_car": "car or fuel",
    "health": "overall health",
    "health_medical_pharmacy": "medical and pharmacy",
    "health_gym_wellness": "gym and wellness",
    "health_personal_care": "personal care",
    "donations_gifts": "donations and gifts",
    "income": "all income",
    "income_salary": "salary",
    "income_sidegig": "sidegig",
    "income_business": "business cashflow",
    "income_interest": "interest",
    "uncategorized": "uncategorized",
    "transfers": "transfers",
    "miscellaneous": "miscellaneous",
    "others": "others",
}

def to_output_category_name(category_name: str) -> str:
  try:
    return OUTPUT_CATEGORY_TEXT_MAP[category_name]
  except KeyError:
    return "uncategorized"

_ALL_OUTPUT_CATEGORY_TEXT_MAP = {
  "top_income": "all income",
  "top_bills": "all bills",
  "top_meals": "food",
  "top_shopping": "all shopping",
  "others": "others",
}
_ALL_OUTPUT_CATEGORY_TEXT_MAP.update(OUTPUT_CATEGORY_TEXT_MAP)

def to_output_all_category_name(category_name: str) -> str:
  return _ALL_OUTPUT_CATEGORY_TEXT_MAP[category_name]
