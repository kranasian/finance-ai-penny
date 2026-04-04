from datetime import date

OFFICIAL_CATEGORY_VALUES: frozenset[str] = frozenset(
  {
    "income_salary",
    "income_sidegig",
    "income_business",
    "income_interest",
    "meals_groceries",
    "meals_dining_out",
    "meals_delivered_food",
    "leisure_entertainment",
    "leisure_travel",
    "bills_connectivity",
    "bills_insurance",
    "bills_tax",
    "bills_service_fees",
    "shelter_home",
    "shelter_utilities",
    "shelter_upkeep",
    "education_kids_activities",
    "education_tuition",
    "shopping_clothing",
    "shopping_gadgets",
    "shopping_kids",
    "shopping_pets",
    "transportation_public",
    "transportation_car",
    "health_medical_pharmacy",
    "health_gym_wellness",
    "health_personal_care",
    "donations_gifts",
    "uncategorized",
    "transfers",
    "miscellaneous",
  }
)


def lookup_transactions(
  start: date,
  end: date,
  name_contains: str = "",
  amount_larger_than: int | None = None,
  amount_less_than: int | None = None,
  in_category: list[str] | None = None,
) -> str:
  """Return a string listing matching transactions for the user (host supplies DB-backed implementation).

  Filters (all optional except date bounds):
  - name_contains: substring match on merchant/description (empty = no filter).
  - amount_larger_than: keep rows with amount >= this value (None = no lower bound).
  - amount_less_than: keep rows with amount <= this value (None = no upper bound).
  - in_category: restrict to these category slugs; each entry must be in OFFICIAL_CATEGORY_VALUES (empty/None = no filter).

  Host-produced row lines match top-transaction excerpts: ``- On YYYY-MM-DD, $N at Merchant as category_slug.``
  with whole-dollar amounts.

  This stub returns an explanatory placeholder when no datastore is wired.
  """
  if in_category:
    bad = [c for c in in_category if c not in OFFICIAL_CATEGORY_VALUES]
    if bad:
      raise ValueError(
        f"in_category contains non-official slugs: {bad}. "
        f"Use only values from OFFICIAL_CATEGORY_VALUES."
      )
  return (
    f"(lookup_transactions stub: {start.isoformat()}..{end.isoformat()} "
    f"name_contains={name_contains!r} amount>={amount_larger_than} amount<={amount_less_than} "
    f"in_category={in_category!r})\nNo rows in offline stub."
  )
