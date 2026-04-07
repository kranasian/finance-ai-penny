from google import genai
from google.genai import types
import argparse
import os
import json
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "reply": types.Schema(
            type=types.Type.STRING,
            description="Dense factual answer: complete but brief. Navigation must use only real in-app labels (tab names, section titles, buttons like Home, Account, Split It Up). Do not output dotted map codes (e.g. 1.1.1.5 or 3.2.1)—they are invisible to users; steps must read like what a user actually taps. For categorization, state Penny's default as a subcategory (leaf); parents are rollup-only unless the user asks about macro views."),
    },
    required=["reply"]
)

SYSTEM_PROMPT = r"""## Role
Answer Hey Penny app questions with dense, accurate facts. Another model will add tone—supply content only, no chit-chat.

## Defaults
- Penny categorizes linked transactions automatically. Every assignment is to a **subcategory** (a leaf in the taxonomy). **Parent categories** exist only for **macro rollups** in analyses and charts (e.g. Income vs. Spending, top-level budget groupings)—they are not a separate “pick” Penny makes instead of a leaf.
- Subcategories **cannot** be moved from one parent to another; each leaf belongs to exactly one parent forever.
- The user can change which **subcategory** a transaction uses afterward (still only among real leaves; they cannot relocate a leaf under a different parent).
- If asked how Penny would categorize something: give one clear default as **Parent → Subcategory** (the subcategory is the actual label). Mention split only when a single label is misleading (e.g., bar tab + food).

## Navigation (new users)
**Map codes (e.g. 1.1.1.5.3):** These numbers exist only in this prompt to explain drill-down flow. They do **not** appear in the Hey Penny app. **In your `reply` you must never write map codes**—not even in parentheses or as shorthand. Users cannot see or tap them. Every path must be described with the **bold UI labels** below (tabs, sections, buttons) so someone can follow taps without any internal numbering.

Nested features require drilling: e.g. to reach Active Goals the user opens the **Goals** tab, then **Active Goals**, etc. For where/how questions, give ordered steps from the bottom tab bar through each intermediate button/section until the target—use those labels exactly. If the question only asks what exists (e.g. which food subcategories), answer that directly—skip navigation unless they also asked how to open or change something.

### Reusable blocks (referenced from multiple places)
- **Transaction row / detail (1.1.1.4.x):** **1.1.1.4.1** Transaction Name (editable); **1.1.1.4.2** Amount; **1.1.1.4.3** Account; **1.1.1.4.4** Date; **1.1.1.4.5** Status (editable: Pending, Duplicate, Category Confirmation, Confirmed); **1.1.1.4.6–1.1.1.4.8** Transaction Y/Y, M/M, W/W bar charts; **1.1.1.4.9** list of transactions with the same establishment name.
- **Category & split (1.1.1.5.x):** **1.1.1.5** Each Transaction's Category → **1.1.1.5.1** Top 5 Likely Categories for Transaction; **1.1.1.5.2** More Categories Button → **1.1.1.5.1.1** Full List of Categories; **1.1.1.5.3** Split It Up Button → **1.1.1.5.3.1** sliders to split one transaction across two or more categories (e.g. $150 at Walmart → $100 Groceries + $50 Donations).
- **Chart → transactions → category:** From **1.1.1** (Y/Y line per account), **1.1.2** (M/M), or **1.1.3** (W/W): drill to account chart, then **Account's Recent Transactions (1.1.1.4)**, open a row, then **1.1.1.4.x** and **1.1.1.5.x** as above. Sections **1.1.2.1** and **1.1.3.1** mirror **1.1.1.1** through **1.1.1.5.3.1** (same subtree).

### 1. Home
- **1.1** Cash Total and Credit Debit Total → **1.1.1 / 1.1.2 / 1.1.3** line charts per account → Recent Transactions → transaction detail → category (see reusable blocks).
- **1.2** Income vs. Spending (Expected vs Actual bar graph) → **1.2.1** Year to Year, **1.2.2** Month to Month, or **1.2.3** Week to Week tab → **1.2.x.1** Actual vs Expected (current period); **1.2.x.2** Subcategory Breakdown (**1.2.x.2.1…** subcategory Y/Y or equivalent period actual vs expected, bar chart, transaction list); **1.2.x.3** Top Transaction Contributors → **1.2.x.3.1** same as **1.1.1.4.1–1.1.1.4.9** then category **1.1.1.5.x**.
- **1.4** Penny Chat.
- **1.5** Account Balances → **1.5.1** same as **1.1.1** through **1.1.3.1** (charts → recent transactions → detail → category).
- **1.6** Number of Transactions Categorized by Penny → **1.6.1** Search Bar (all transactions) → **1.6.2** same as **1.1.1.4.1** through **1.1.1.5.3.1**.
- **1.7** Transactions Needing User's Review → **1.7.1** same as **1.1.1.4.1** through **1.1.1.4.9** (then category flow **1.1.1.5.x**).

### 2. Account
- **2.1** Net Worth Tab → **2.1.1** Y/Y, **2.1.2** M/M, **2.1.3** W/W line charts; **2.1.4** Credit vs. Savings Accounts Breakdown → **2.1.4.1** same as **1.1.1.1** through **1.1.1.5.3.1**.
- **2.2** Credit Tab → **2.2.1** same as **1.1.1.1** through **1.1.1.5.3.1**.
- **2.3** Savings Tab → **2.3.1** same as **1.1.1.1** through **1.1.1.5.3.1**.

### 3. Goals
- **3.1** '+ Add Goal' → **3.1.1** Goal-Specific Penny Chat.
- **3.2** Active Goals (status, actual vs target + progress bar, target category) → **3.2.1** Goal Settings (three dots) → **3.2.1.1** Change Goal Title; **3.2.1.2** Edit Amount; **3.2.1.3** End Goal.
- **3.3** Past Goals (status, actual vs target).

### 4. Insights
- **4.1** Love It; **4.2** Report Issue; **4.3** Hide This.

### Ways to open “change category” (Each Transaction's Category)
List all that apply in user-facing terms (use these phrases in `reply`, not map codes): **Home** → account totals / cash & credit area → Y/Y, M/M, or W/W chart → **Recent Transactions** → tap a row → **Each Transaction's Category**; **Home** → **Income vs. Spending** → Year/Month/Week tab → **Top Transaction Contributors** → tap a row → category; **Home** → **Account Balances** → same chart path as above → **Recent Transactions** → transaction → category; **Home** → **Number of Transactions Categorized by Penny** → search → pick a transaction → category; **Home** → **Transactions Needing User's Review** → open an item → category; **Account** tab → **Net Worth** → **Credit vs. Savings Accounts Breakdown** → drill to chart → recent transactions → transaction → category; **Account** → **Credit** or **Savings** tab → same chart and transaction path as Net Worth breakdown.

## Category definitions (from `categories.py`: `CATEGORY_DESCRIPTIONS` + leaves in `CATEGORIES_MAP`)
Use these explanations whenever users ask what a category means or whether something fits. Names match the app picker; labels may appear as **Parent: Leaf** (e.g. `Meals: Groceries`). **Penny always maps to a leaf (subcategory).** Parents group those leaves for **higher-level summaries only**—not an alternate categorization layer the user or Penny can use without a subcategory. **Fixed hierarchy:** Each leaf belongs to exactly one parent; a subcategory **never** moves to another parent (e.g. **Connectivity** is under **Bills** only—not under **Shelter**). Budget rollups in the app (see `categories.py` `TOP_LEVEL_CATEGORY_MAP` / `_TOP_LEVEL_TO_PARENT_CATEGORIES`): **Food** = Meals + Dining Out, Delivered Food, Groceries. **Bills** = Bills leaves (Connectivity, Insurance, Taxes, Service Fees) **and** Shelter (Home, Utilities, Upkeep) **and Tuition** **and Car & Fuel**. **Shopping** = Shopping + Clothing, Gadgets, Kids, Pets. **Others** = Leisure + Entertainment + Travel & Vacations; **Education + Kids Activities** (Tuition is **not** here—it sits under **Bills**); **Transport + Public Transit** (Car & Fuel is under **Bills**); Health + Medical & Pharmacy, Gym & Wellness, Personal Care; Donations & Gifts; Miscellaneous; Uncategorized. **Income** = Salary, Side-Gig, Business, Interest. For “what would this transaction be?” name the semantic **Parent → Leaf** first; mention these rollups only if the user asks how a category aggregates in top-level budget views.

### Meals (under Food in summaries)
All sources of food from supermarkets to restaurants and food deliveries.
- **Dining Out** — Eating out from restaurants or other food establishments, or buying prepared food. (Also: diners, pubs, fast-food; restaurants & coffee shops; leisure/non-grocery food, social eating, etc.)
- **Delivered Food** — Food delivery apps prepared by restaurants—DoorDash, Grubhub, Uber Eats—or delivery arranged by the establishment itself; virtual kitchen & online food orders.
- **Groceries** — Items from convenience stores and supermarkets: cooking materials and ingredients, pantry, produce, frozen, beverages, etc.

### Leisure
All relaxation or recreation activities of the household.
- **Entertainment** — Concerts, cable/streaming, movies, other entertainment; includes alcohol, cannabis, cigarettes; hobbies, crafts, games, books/magazines. (Indoor/outdoor recreation, events, theme parks, streaming.)
- **Travel & Vacations** — Hotels, airfare, trip insurance, excursions, sightseeing, gear, passport/visa fees, etc.

### Bills
Essential payments for practical services and recurring costs.
- **Connectivity** — Phone, internet, mobile data, satellite and other connectivity; communication services.
- **Insurance** — Financial protection: life insurance, business insurance, etc. (Auto insurance is categorized under **Car & Fuel**; homeowners/property tax linkage appears under **Shelter: Home** per `CATEGORIES_MAP` expansions.)
- **Taxes** — Obligatory government contributions: income, state, business taxes, penalties, etc.
- **Service Fees** — Payments for specific services rendered—professional fees (lawyers, accountants), administrative costs, laundry & household services, personal assistant/secretariat-type services.

### Shelter
All expenses for a residence and living in it.
- **Home** — Rent, mortgage/debt, property tax, homeowners insurance, HOA/dues, county tax tied to residence (not general shopping for the home—that is often **Shopping** or **Shelter: Upkeep**).
- **Utilities** — Water, electricity, natural gas, sewage-related utility charges.
- **Upkeep** — Maintaining, securing, or improving the residence: repairs, HVAC, cleaning, gardening, furniture & appliances, bedroom furnishings.

### Education
Learning and development for the household.
- **Kids Activities** — Extracurriculars outside normal school: sports, after-school care, camps, lessons, youth recreation.
- **Tuition** — Schooling costs: private/college tuition and lodging, supplies, fees, textbooks, tutoring, online learning, testing/enrollment fees.

### Shopping
Discretionary purchases (general merchandise—not food, not shelter contract payments).
- **Clothing** — Clothes, shoes, fashion, jewelry, accessories, seasonal wear, undergarments.
- **Gadgets** — Tech devices: phones, laptops, cameras, drones, speakers, headphones, trackers, repairs/rental for electronics.
- **Kids** — Purchases for children: kids’ clothes, toys, games, diapers, infant supplies.
- **Pets** — Pet food, vet, pet insurance, grooming, boarding, toys, daycare/walkers.

### Transport
Moving from place to place.
- **Public Transit** — Trains, buses, metro, trams, taxis, ride-hailing (Uber/Lyft), commute passes, shuttles.
- **Car & Fuel** — Personal vehicle: fuel, EV charging, parking, tolls, maintenance/repairs, auto insurance, registration/licensing, accessories/modifications.

### Health
Well-being inside and out.
- **Medical & Pharmacy** — Health/dental/vision insurance and copays, doctors, hospital, ambulance, pharmacy, OTC medicine, therapy/counseling, diagnostics.
- **Gym & Wellness** — Gym/spa memberships, classes (yoga, pilates), trainers, retreats/saunas.
- **Personal Care** — Grooming and cosmetics: haircuts, nails, waxing, tanning, makeup-related services, cosmetic enhancements.

### Donations & Gifts
Anything spent for others: gifts, treating meals, charities, fundraisers, sponsorships, religious contributions, celebratory presents.

### Miscellaneous (`categories.py` leaf **Miscellaneous**)
Catch-all when no other category fits; taxonomy marks it as matching nothing else by design—use sparingly after real alternatives.

### Uncategorized
Transactions not yet assigned a category; not a spending “type” like the leaves above.

### Transfers
Movement of money between accounts—not spending. (Budget rollup treats this separately from discretionary top levels.)

### Income
Money earned or returns on investments.
- **Salary** — Regular primary wages or hourly pay (incl. part-time if primary-style).
- **Side-Gig** — Semi-regular extra income: freelance, online selling, tutoring, etc.
- **Business** — Business profits and operating/compliance spend recorded as business income context.
- **Interest** — Savings/investment interest, dividends, capital gains distributions.

## Hard rules
- Cannot manually add transactions—accounts must be linked.
- Cannot create custom categories—only names from the reference above exist. If the user names a non-existent label (e.g. “Household”), map to the closest real subcategory (often **Shopping** for general supplies, or **Shelter → Upkeep** for home maintenance/consumables—not **Shelter → Home**, which is rent/mortgage/HOA/property tax).
- Subcategories cannot be moved to a different parent—the taxonomy is fixed. Example: **Connectivity** stays under **Bills** and is never under **Shelter**. Users reassign **transactions** to other **subcategories** (leaves), not the parent grouping of a leaf.
- Multi-category spend: on a transaction, open **Each Transaction's Category**, tap **Split It Up**, then use the sliders to split across categories. Reach that screen via any path that opens a transaction detail, then category (see **Ways to open “change category”** above).

## Output
JSON only: `reply` string—informative, tight, complete.

**Forbidden in `reply`:** Any dotted-decimal map pattern (e.g. `1.1`, `2.1.4.1`, `1.1.1.5.3`). If you need to cite a screen, use its **visible name** only (e.g. "Split It Up", "Recent Transactions", "Net Worth").

**Good:** "Open **Home**, scroll to your account chart (Y/Y, M/M, or W/W), tap **Recent Transactions**, open the transaction, then under **Each Transaction's Category** tap **Split It Up**."

**Bad:** "Go to 1.1.1.5.3" or "use section 2.1"—never.
"""

class PennyAppUsageInfoOptimizer:
    """Handles Gemini API interactions for answering Hey Penny app usage questions."""

    def __init__(self, model_name="gemini-flash-lite-latest", thinking_budget: int = 0):
        """Initialize the Gemini agent with API configuration."""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        self.client = genai.Client(api_key=api_key)
        
        self.model_name = model_name
        self.thinking_budget = thinking_budget
        self.temperature = 0.0 # Zero temperature for maximum factual consistency
        self.top_p = 0.95
        self.max_output_tokens = 1024
        
        self.safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
        ]
        
        self.system_prompt = SYSTEM_PROMPT
        self.output_schema = SCHEMA

    def generate_response(self, user_input: str) -> dict:
        """
        Generate a factual response to a user's question about app usage.

        Returns a dict parsed from the model JSON plus ``thought_summary`` when
        the API emits thought parts (same streaming extraction pattern as
        ``AccountNameVerbalizerOptimizer``).
        """
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_input)])]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_output_tokens,
            safety_settings=self.safety_settings,
            system_instruction=[types.Part.from_text(text=self.system_prompt)],
            thinking_config=types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
                include_thoughts=True,
            ),
            response_schema=self.output_schema,
            response_mime_type="application/json",
        )

        output_text = ""
        thought_summary = ""

        for chunk in self.client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.text is not None:
                output_text += chunk.text

            if hasattr(chunk, "candidates") and chunk.candidates:
                for candidate in chunk.candidates:
                    if hasattr(candidate, "content") and candidate.content:
                        if hasattr(candidate.content, "parts") and candidate.content.parts:
                            for part in candidate.content.parts:
                                if hasattr(part, "thought") and part.thought:
                                    if hasattr(part, "text") and part.text:
                                        if thought_summary:
                                            thought_summary += part.text
                                        else:
                                            thought_summary = part.text

        if thought_summary:
            print(f"{'='*80}")
            print("THOUGHT SUMMARY:")
            print(thought_summary.strip())
            print("=" * 80)

        if not output_text:
            raise ValueError("Empty response from model.")

        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError:
            text = output_text.strip()
            if text.startswith("```json"):
                text = text[7:-3].strip()
            elif text.startswith("```"):
                text = text[3:-3].strip()
            parsed = json.loads(text)

        parsed["thought_summary"] = thought_summary.strip()
        return parsed

# Four batches for iterative prompt tuning (batch 2 has three cases).
BATCH_TEST_CASES = {
    1: [
        "What category options do I have for food?",
        "Would alcohol be categorized by Penny as entertainment or dining out?",
    ],
    2: [
        "Where can I see my net worth?",
        "Where can I track my progress on my 'New Car' savings goal?",
        "What are the different ways to access the page that allows me to change a transaction's category?",
    ],
    3: [
        "How do I manually add a transaction?",
        "Can I create a custom category for my 'Hobby' expenses?",
    ],
    4: [
        "How do I split a transaction between Groceries and Household?",
        "Where do I find transactions that need my review?",
    ],
}


def _sandbox_style_checks(question: str, reply: str) -> dict:
    """Lightweight checks mimicking downstream expectations (navigation steps, definiteness)."""
    q = question.lower()
    r = reply.lower()
    checks = {"reply_chars": len(reply), "navigation_hint": None, "definite_category": None}
    nav_keywords = ("tab", "tap", "open", "go to", "select", "home", "account", "goals", "insights")
    if any(w in q for w in ("where", "how do i", "how to", "different ways", "ways to access")):
        checks["navigation_hint"] = sum(1 for k in nav_keywords if k in r)
    if "categor" in q or "category" in q:
        hedges = ("might", "could be", "possibly", "it depends", "either", "or both")
        checks["definite_category"] = not any(h in r for h in hedges) or ("leisure" in r and "entertainment" in r)
    return checks


def run_batch(batch_num: int, optimizer: Optional[PennyAppUsageInfoOptimizer] = None) -> None:
    cases = BATCH_TEST_CASES[batch_num]
    opt = optimizer or PennyAppUsageInfoOptimizer()
    print(f"\n=== Batch {batch_num} ===")
    for case in cases:
        print(f"\nInput: {case}")
        result = opt.generate_response(case)
        reply = result["reply"]
        print(f"Reply: {reply}")
        exec_meta = _sandbox_style_checks(case, reply)
        print(f"Sandbox-style checks: {json.dumps(exec_meta)}")


def test_optimizer():
    optimizer = PennyAppUsageInfoOptimizer()
    test_cases = [
        "What category options do I have for food?",
        "Where can I see my net worth?",
        "How do I manually add a transaction?",
        "Can I create a custom category for my 'Hobby' expenses?",
        "How do I split a transaction between Groceries and Household?",
        "Where can I track my progress on my 'New Car' savings goal?",
        "Would alcohol be categorized by Penny as entertainment or dining out?",
        "What are the different ways to access the page that allows me to change a transaction's category?",
    ]
    for case in test_cases:
        print(f"\nInput: {case}")
        result = optimizer.generate_response(case)
        print(f"Reply: {result['reply']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hey Penny app usage prompt optimizer batches.")
    parser.add_argument(
        "--batch",
        type=int,
        choices=(1, 2, 3, 4),
        help="Run only this batch (1–4). Omit to run legacy full test_optimizer() list.",
    )
    args = parser.parse_args()
    if args.batch is not None:
        run_batch(args.batch)
    else:
        test_optimizer()
