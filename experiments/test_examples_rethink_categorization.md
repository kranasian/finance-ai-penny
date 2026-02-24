# Test examples for RethinkTransactionCategorization checker
# 4 groups of 3 test examples each — varied EVAL_INPUT topics and REVIEW_NEEDED accuracy

---

## Group 1

<EVAL_INPUT>
[
  {
    "group_id": "1:2001",
    "establishment_name": "AMC Theatres",
    "establishment_description": "movie theater chain offering film screenings and concessions",
    "transactions": [
      {
        "transaction_id": 3001,
        "transaction_text": "AMC THEATRES #123",
        "amount": 45.50
      }
    ],
    "category_options": [
      "leisure_entertainment",
      "leisure_travel",
      "meals_dining_out",
      "bills_service_fees",
      "donations_gifts"
    ]
  },
  {
    "group_id": "1:2002",
    "establishment_name": "Home Depot",
    "establishment_description": "home improvement retailer selling tools, construction products, and services",
    "transactions": [
      {
        "transaction_id": 3002,
        "transaction_text": "HOME DEPOT #456",
        "amount": 89.99
      }
    ],
    "category_options": [
      "shelter_upkeep",
      "shelter_home",
      "shopping_gadgets",
      "bills_service_fees",
      "donations_gifts"
    ]
  },
  {
    "group_id": "1:2003",
    "establishment_name": "Chase Bank Interest",
    "establishment_description": "interest earned on a savings or checking account balance",
    "transactions": [
      {
        "transaction_id": 3003,
        "transaction_text": "INTEREST PAYMENT",
        "amount": -12.45
      }
    ],
    "category_options": [
      "income_interest",
      "income_salary",
      "income_sidegig",
      "transfer",
      "bills_service_fees"
    ]
  }
]
</EVAL_INPUT>
<REVIEW_NEEDED>
[
  {
    "group_id": "1:2001",
    "transaction_id": 3001,
    "reasoning": "Movie theater purchase for entertainment services.\n",
    "category": "leisure_entertainment",
    "confidence": "high"
  },
  {
    "group_id": "1:2002",
    "transaction_id": 3002,
    "reasoning": "Home improvement store purchase; likely for home maintenance or repairs.\n",
    "category": "shelter_upkeep",
    "confidence": "high"
  },
  {
    "group_id": "1:2003",
    "transaction_id": 3003,
    "reasoning": "Inflow representing interest earned on bank account balance.\n",
    "category": "income_interest",
    "confidence": "high"
  }
]
</REVIEW_NEEDED>

---

## Group 2

<EVAL_INPUT>
[
  {
    "group_id": "2:3001",
    "establishment_name": "Equinox",
    "establishment_description": "luxury fitness club offering gym access, classes, and personal training",
    "transactions": [
      {
        "transaction_id": 4001,
        "transaction_text": "EQUINOX CLUB MEMBERSHIP",
        "amount": 250.00
      }
    ],
    "category_options": [
      "health_gym_wellness",
      "health_medical_pharmacy",
      "leisure_entertainment",
      "bills_service_fees",
      "donations_gifts"
    ]
  },
  {
    "group_id": "2:3002",
    "establishment_name": "Apple Store",
    "establishment_description": "retail location for Apple products including iPhones, Macs, and accessories",
    "transactions": [
      {
        "transaction_id": 4002,
        "transaction_text": "APPLE STORE #R123",
        "amount": 1299.00
      }
    ],
    "category_options": [
      "shopping_gadgets",
      "shopping_clothing",
      "leisure_entertainment",
      "bills_service_fees",
      "education_tuition"
    ]
  },
  {
    "group_id": "2:3003",
    "establishment_name": "Kumon Learning Center",
    "establishment_description": "after-school math and reading programs for children",
    "transactions": [
      {
        "transaction_id": 4003,
        "transaction_text": "KUMON CENTER FEES",
        "amount": 180.00
      }
    ],
    "category_options": [
      "education_kids_activities",
      "education_tuition",
      "shopping_kids",
      "bills_service_fees",
      "donations_gifts"
    ]
  }
]
</EVAL_INPUT>
<REVIEW_NEEDED>
[
  {
    "group_id": "2:3001",
    "transaction_id": 4001,
    "reasoning": "Monthly membership fee for a fitness and wellness club.\n",
    "category": "health_gym_wellness",
    "confidence": "high"
  },
  {
    "group_id": "2:3002",
    "transaction_id": 4002,
    "reasoning": "High-value purchase at an electronics retailer; likely a computer or phone.\n",
    "category": "shopping_gadgets",
    "confidence": "high"
  },
  {
    "group_id": "2:3003",
    "transaction_id": 4003,
    "reasoning": "Payment for children's supplemental education and learning activities.\n",
    "category": "education_kids_activities",
    "confidence": "high"
  }
]
</REVIEW_NEEDED>

---

## Group 3

<EVAL_INPUT>
[
  {
    "group_id": "3:4001",
    "establishment_name": "MTA MetroCard",
    "establishment_description": "public transportation system for New York City including subways and buses",
    "transactions": [
      {
        "transaction_id": 5001,
        "transaction_text": "MTA*METROCARD VENDING",
        "amount": 33.00
      }
    ],
    "category_options": [
      "transportation_public",
      "transportation_car",
      "leisure_travel",
      "bills_service_fees",
      "donations_gifts"
    ]
  },
  {
    "group_id": "3:4002",
    "establishment_name": "Verizon Wireless",
    "establishment_description": "telecommunications company providing mobile phone and internet services",
    "transactions": [
      {
        "transaction_id": 5002,
        "transaction_text": "VERIZON*WIRELESS PYMT",
        "amount": 115.20
      }
    ],
    "category_options": [
      "bills_connectivity",
      "bills_utilities",
      "shopping_gadgets",
      "bills_service_fees",
      "donations_gifts"
    ]
  },
  {
    "group_id": "3:4003",
    "establishment_name": "Sallie Mae Loan Payment",
    "establishment_description": "payment toward a student loan balance held by the account holder",
    "transactions": [
      {
        "transaction_id": 5003,
        "transaction_text": "SALLIE MAE PYMT",
        "amount": 400.00
      }
    ],
    "category_options": [
      "transfer",
      "education_tuition",
      "bills_service_fees",
      "income_business",
      "donations_gifts"
    ]
  }
]
</EVAL_INPUT>
<REVIEW_NEEDED>
[
  {
    "group_id": "3:4001",
    "transaction_id": 5001,
    "reasoning": "Purchase of a transit pass for public subway or bus travel.\n",
    "category": "transportation_public",
    "confidence": "high"
  },
  {
    "group_id": "3:4002",
    "transaction_id": 5002,
    "reasoning": "Monthly bill for mobile phone and data connectivity services.\n",
    "category": "bills_connectivity",
    "confidence": "high"
  },
  {
    "group_id": "3:4003",
    "transaction_id": 5003,
    "reasoning": "Payment toward own student loan liability; net worth remains unchanged.\n",
    "category": "transfer",
    "confidence": "high"
  }
]
</REVIEW_NEEDED>

---

## Group 4

<EVAL_INPUT>
[
  {
    "group_id": "4:5001",
    "establishment_name": "The Cheesecake Factory",
    "establishment_description": "full-service casual dining restaurant chain",
    "transactions": [
      {
        "transaction_id": 6001,
        "transaction_text": "CHEESECAKE FACTORY #789",
        "amount": 62.35
      }
    ],
    "category_options": [
      "meals_dining_out",
      "meals_groceries",
      "meals_delivered_food",
      "leisure_entertainment",
      "donations_gifts"
    ]
  },
  {
    "group_id": "4:5002",
    "establishment_name": "Petco",
    "establishment_description": "retailer specializing in pet supplies, food, and grooming services",
    "transactions": [
      {
        "transaction_id": 6002,
        "transaction_text": "PETCO #321",
        "amount": 42.18
      }
    ],
    "category_options": [
      "shopping_pets",
      "shopping_general",
      "health_medical_pharmacy",
      "bills_service_fees",
      "donations_gifts"
    ]
  },
  {
    "group_id": "4:5003",
    "establishment_name": "PayPal to John Doe",
    "establishment_description": "person-to-person payment sent via PayPal to an individual named John Doe",
    "transactions": [
      {
        "transaction_id": 6003,
        "transaction_text": "PAYPAL TRANSFER TO John Doe",
        "amount": 50.00
      }
    ],
    "category_options": [
      "donations_gifts",
      "meals_dining_out",
      "leisure_entertainment",
      "transfer",
      "income_business"
    ]
  }
]
</EVAL_INPUT>
<REVIEW_NEEDED>
[
  {
    "group_id": "4:5001",
    "transaction_id": 6001,
    "reasoning": "Restaurant meal purchase at a full-service dining establishment.\n",
    "category": "meals_dining_out",
    "confidence": "high"
  },
  {
    "group_id": "4:5002",
    "transaction_id": 6002,
    "reasoning": "Purchase of pet-related supplies or services from a specialty retailer.\n",
    "category": "shopping_pets",
    "confidence": "high"
  },
  {
    "group_id": "4:5003",
    "transaction_id": 6003,
    "reasoning": "P2P payment to an individual with no stated purpose in the transaction text.\n",
    "category": "unknown",
    "confidence": "low"
  }
]
</REVIEW_NEEDED>
