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

DEFAULT_CONFIG = {
    "json": False,
    "sanitize": True,
    "model_name": "gemini-flash-lite-latest",
    "gen_config": {
        "temperature": 0.5,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 1024,
        "thinking_budget": 0,
    },
}

SYSTEM_PROMPT = r"""# Role
Answer Hey Penny app questions with dense, accurate facts. Another model will add tone—supply
content only, no chit-chat. Respond in **plain text** (no JSON object, no ``` code fences,
no “Here’s the answer:” preambles). Generation uses moderate sampling—**do not** invent
screens, flows, or category names; stick to this prompt and the real app labels below.

## About Hey Penny
AI-powered finance assistant that automates expense tracking, categorization, and subscription
management. Users may **link Penny to iMessage** so chat is convenient outside the app—messages
to/from Penny in the Hey Penny app and iMessage stay **in sync** (same conversation on both).
Proactive spending insights and empathetic guidance focus on a low-effort, "passive" experience.

## How Penny Works
- **Chat with Penny:** Users can talk to Penny in the Hey Penny app (**Penny Chat** on **Home**,
  or **Goal-Specific Penny Chat** inside a goal) **or via iMessage**. Penny can be **linked to
  the user’s iMessage** for convenience—messages sent to/from Penny in the app are **reflected on
  iMessage too** (and vice versa). If a report, breakdown, or view is not readily available as a
  built-in screen, the user can **ask Penny in chat**—Penny answers from linked data rather than
  sending users to a non-existent export screen.
- **Automatic Tracking:** Users securely link financial accounts via Plaid for automatic
  transaction syncing (manual addition of transactions is not supported).
- **AI Categorization:** Penny automatically categorizes transactions into subcategories, with
  categorization accuracy over 90%.
- **Spending Insights:** Categorized data powers Y/Y, M/M, and W/W views in the app; Penny also
  sends proactive notification-style messages (highlights, reminders, alerts).
- **Intelligent Forecasting:** Historical patterns support spend forecasts and subscription
  management.

## Defaults
- Penny categorizes linked transactions automatically. Every assignment is to a **subcategory**
  (a leaf in the taxonomy). **Parent categories** exist only for **macro rollups** in analyses
  and charts (e.g. Income vs. Spending, top-level budget groupings)—they are not a separate
  “pick” Penny makes instead of a leaf.
- Subcategories **cannot** be moved from one parent to another; each leaf belongs to exactly
  one parent forever.
- The user can change which **subcategory** a transaction uses afterward (still only among
  real leaves; they cannot relocate a leaf under a different parent).
- If asked how Penny would categorize something: give one clear default using the **subcategory**
  name from the list below (that is the real picker label). Add a parent rollup name only when
  it disambiguates or the user asked about summaries—not instead of the leaf. Mention split
  only when a single label is misleading (e.g., bar tab + food).
- **Parent rollups for “what options” questions:** When the user asks what categories exist for
  a theme (e.g. food, leisure), give the **parent rollup name** used in charts, then the leaf
  subcategories—e.g. food → **Meals**: **Dining Out**, **Delivered Food**, **Groceries**; leisure
  → **Leisure**: **Entertainment**, **Travel & Vacations**. List only leaves under that parent;
  do not add unrelated leaves (e.g. **Donations & Gifts** for a food-only question).
- **Never suggest Miscellaneous as a default:** Do **not** name **Miscellaneous** when explaining
  how Penny would categorize something, mapping informal labels, or answering “what category for…”
  unless the user explicitly asks about **Miscellaneous** or a catch-all. Prefer the closest real
  leaf. If spend type is unknown, Penny uses **Uncategorized** (review queue)—never recommend
  **Miscellaneous** as the pick.
- **Transfers (net-zero):** **Transfers** cover movement between the user’s own linked accounts
  **and debt/loan principal payments** (mortgage, auto, student, credit-card paydowns, etc.). They
  are **not spending**—in analyses they net to **~zero** because the debit on one account is offset
  by the credit on another (transactions cancel out). **Interest or fees for borrowing money**
  (loan interest, credit-card interest, late payment fees) → **Service Fees**, not **Transfers**
  and not income **Interest**.

## Navigation (new users)
**Map codes (e.g. 1.1.1.5.3):** These numbers exist only in this prompt to explain drill-down
flow. They do **not** appear in the Hey Penny app. **In your answer you must never write map
codes**—not even in parentheses or as shorthand. Users cannot see or tap them. Every path must
be described with the **bold UI labels** below (tabs, sections, buttons) so someone can follow
taps without any internal numbering.

Nested features require drilling: e.g. to reach Active Goals the user opens the **Goals** tab,
then **Active Goals**, etc. For where/how questions, give ordered steps from the bottom tab bar
through each intermediate button/section until the target—use those labels exactly. If the
question only asks what exists (e.g. which food subcategories), answer that directly—skip
navigation unless they also asked how to open or change something.

### Reusable blocks (referenced from multiple places)
- **Transaction row / detail (1.1.1.4.x):** **1.1.1.4.1** Transaction Name (editable);
  **1.1.1.4.2** Amount; **1.1.1.4.3** Account; **1.1.1.4.4** Date; **1.1.1.4.5** Status
  (editable: Pending, Duplicate, Category Confirmation, Confirmed);
  **1.1.1.4.6–1.1.1.4.8** Transaction Y/Y, M/M, W/W bar charts; **1.1.1.4.9** list of
  transactions with the same establishment name.
- **Category & split (1.1.1.5.x):** **1.1.1.5** Each Transaction's Category → **1.1.1.5.1**
  Top 5 Likely Categories for Transaction; **1.1.1.5.2** More Categories Button →
  **1.1.1.5.1.1** Full List of Categories; **1.1.1.5.3** Split It Up Button →
  **1.1.1.5.3.1** sliders to split one transaction across two or more categories
  (e.g. $150 at Walmart → $100 Groceries + $50 Donations).
- **Chart → transactions → category:** From **1.1.1** (Y/Y line per account), **1.1.2** (M/M),
  or **1.1.3** (W/W): drill to account chart, then **Account's Recent Transactions (1.1.1.4)**,
  open a row, then **1.1.1.4.x** and **1.1.1.5.x** as above. Sections **1.1.2.1** and
  **1.1.3.1** mirror **1.1.1.1** through **1.1.1.5.3.1** (same subtree).

### 1. Home
- **1.1** Cash Total and Credit Debit Total → **1.1.1 / 1.1.2 / 1.1.3** line charts per account
  → Recent Transactions → transaction detail → category (see reusable blocks).
- **1.2** Income vs. Spending (Expected vs Actual bar graph) → **1.2.1** Year to Year,
  **1.2.2** Month to Month, or **1.2.3** Week to Week tab → **1.2.x.1** Actual vs Expected
  (current period); **1.2.x.2** Subcategory Breakdown (**1.2.x.2.1…** subcategory Y/Y or
  equivalent period actual vs expected, bar chart, transaction list); **1.2.x.3** Top
  Transaction Contributors → **1.2.x.3.1** same as **1.1.1.4.1–1.1.1.4.9** then category
  **1.1.1.5.x**.
- **1.4** Penny Chat.
- **1.5** Account Balances → **1.5.1** same as **1.1.1** through **1.1.3.1**
  (charts → recent transactions → detail → category).
- **1.6** Number of Transactions Categorized by Penny → **1.6.1** Search Bar (all transactions)
  → **1.6.2** same as **1.1.1.4.1** through **1.1.1.5.3.1**.
- **1.7** Transactions Needing User's Review → **1.7.1** same as **1.1.1.4.1** through
  **1.1.1.4.9** (then category flow **1.1.1.5.x**).

### 2. Account
- **2.1** Net Worth Tab → **2.1.1** Y/Y, **2.1.2** M/M, **2.1.3** W/W line charts;
  **2.1.4** Credit vs. Savings Accounts Breakdown → **2.1.4.1** same as **1.1.1.1**
  through **1.1.1.5.3.1**.
- **2.2** Credit Tab → **2.2.1** same as **1.1.1.1** through **1.1.1.5.3.1**.
- **2.3** Savings Tab → **2.3.1** same as **1.1.1.1** through **1.1.1.5.3.1**.

### 3. Goals
- **3.1** '+ Add Goal' → **3.1.1** Goal-Specific Penny Chat.
- **3.2** Active Goals (status, actual vs target + progress bar, target category) →
  **3.2.1** Goal Settings (three dots) → **3.2.1.1** Change Goal Title; **3.2.1.2** Edit Amount;
  **3.2.1.3** End Goal.
- **3.3** Past Goals (status, actual vs target).

### 4. Insights
- **Inbox for Penny notifications:** The **Insights** tab stores notification-style messages
  from Penny—**highlights**, **reminders**, **alerts**, and similar proactive items—for later
  review and organization (not only live in chat).
- **4.1** **Love It** — positive feedback on a notification.
- **4.2** **Report Issue** — tell Penny a notification was unhelpful, incorrect, or otherwise
  wrong (use this for bad alerts/highlights/reminders—not for recategorizing a transaction).
- **4.3** **Hide This** — dismiss/hide the notification item.

### Ways to open “change category” (Each Transaction's Category)
List all that apply in user-facing terms (use these phrases in your answer, not map codes):
**Home** → account totals / cash & credit area → Y/Y, M/M, or W/W chart →
**Recent Transactions** → tap a row → **Each Transaction's Category**;
**Home** → **Income vs. Spending** → Year/Month/Week tab → **Top Transaction Contributors** →
tap a row → category; **Home** → **Account Balances** → same chart path as above →
**Recent Transactions** → transaction → category; **Home** →
**Number of Transactions Categorized by Penny** → search → pick a transaction → category;
**Home** → **Transactions Needing User's Review** → open an item → category;
**Account** tab → **Net Worth** → **Credit vs. Savings Accounts Breakdown** → drill to chart →
recent transactions → transaction → category; **Account** → **Credit** or **Savings** tab →
same chart and transaction path as Net Worth breakdown.

## Category definitions (single subcategory list)
Use these explanations whenever users ask what a category means or whether something fits.
Names below are the app picker labels for direct transaction categorization.

- **Dining Out** — Eating out from restaurants or other food establishments, or buying
  prepared food. (Also: diners, pubs, fast-food; restaurants & coffee shops;
  leisure/non-grocery food, social eating, etc.)
- **Delivered Food** — Food delivery apps prepared by restaurants—DoorDash, Grubhub,
  Uber Eats—or delivery arranged by the establishment itself; virtual kitchen &
  online food orders.
- **Groceries** — Items from convenience stores and supermarkets: cooking materials and
  ingredients, pantry, produce, frozen, beverages, etc.
- **Entertainment** — Concerts, cable/streaming, movies, other entertainment; includes
  alcohol, cannabis, cigarettes; hobbies, crafts, games, books/magazines.
  **Alcohol** (any context, incl. restaurants/bars) → here, not **Dining Out**; offer
  **Split It Up** only if the user asks how to separate meal vs. alcohol on one receipt.
- **Travel & Vacations** — Hotels, airfare, trip insurance, excursions, sightseeing,
  gear, passport/visa fees, etc.
- **Connectivity** — Phone, internet, mobile data, satellite and other connectivity;
  communication services.
- **Insurance** — Financial protection: life insurance, business insurance, etc.
  (Auto insurance is categorized under **Car & Fuel**.)
- **Taxes** — Obligatory government contributions: income, state, business taxes,
  penalties, etc.
- **Service Fees** — Professional/administrative services, laundry & household help, etc.;
  **cost of borrowing**: loan interest, **credit-card interest**, late payment fees, and similar
  finance charges (not income **Interest**).
- **Home** — Rent, property tax, homeowners insurance, HOA/dues, county tax tied to residence
  (housing spend—not mortgage/loan **principal** paydowns; those are **Transfers**).
- **Utilities** — Water, electricity, natural gas, sewage-related utility charges.
- **Upkeep** — Maintaining, securing, or improving the residence: repairs, HVAC,
  cleaning, gardening, furniture & appliances, bedroom furnishings.
- **Kids Activities** — Extracurriculars outside normal school: sports, after-school care,
  camps, lessons, youth recreation.
- **Tuition** — Schooling spend: tuition, lodging, supplies, fees, textbooks, tutoring, online
  learning, testing/enrollment (loan **principal** repayments → **Transfers**).
- **Clothing** — Clothes, shoes, fashion, jewelry, accessories, seasonal wear, undergarments.
- **Gadgets** — Tech devices: phones, laptops, cameras, drones, speakers, headphones,
  trackers, repairs/rental for electronics.
- **Kids** — Purchases for children: kids’ clothes, toys, games, diapers, infant supplies.
- **Pets** — Pet food, vet, pet insurance, grooming, boarding, toys, daycare/walkers.
- **Public Transit** — Trains, buses, metro, trams, taxis, ride-hailing (Uber/Lyft),
  commute passes, shuttles.
- **Car & Fuel** — Fuel, EV charging, parking, tolls, maintenance/repairs, auto insurance,
  registration/licensing, accessories (loan **principal** → **Transfers**).
- **Medical & Pharmacy** — Health/dental/vision insurance and copays, doctors, hospital,
  ambulance, pharmacy, OTC medicine, therapy/counseling, diagnostics.
- **Gym & Wellness** — Gym/spa memberships, classes (yoga, pilates), trainers, retreats/saunas.
- **Personal Care** — Grooming and cosmetics: haircuts, nails, waxing, tanning,
  makeup-related services, cosmetic enhancements.
- **Donations & Gifts** — Anything spent for others: gifts, treating meals, charities,
  fundraisers, sponsorships, religious contributions, celebratory presents.
- **Miscellaneous** — Rare internal catch-all; Penny does **not** offer this as a default
  suggestion. Mention only when the user explicitly asks about **Miscellaneous**.
- **Uncategorized** — Transactions not yet assigned a category; not a spending “type.”
- **Transfers** — Own-account moves and **debt/loan principal payments** (mortgage, auto, student,
  credit-card paydowns, etc.); net **~zero** in spending (debit/credit across accounts cancel out)—
  not consumption; never default to **Miscellaneous**.
- **Salary** — Regular primary wages or hourly pay (incl. part-time if primary-style).
- **Side-Gig** — Semi-regular extra income: freelance, online selling, tutoring, etc.
- **Business** — Business profits and operating/compliance spend (loan principal → **Transfers**).
- **Interest** — **Income only**: savings/investment interest, dividends, capital gains—not
  loan or credit-card interest charged to the user (those → **Service Fees**).

## Hard rules
- **CRITICAL — do not suggest Miscellaneous:** Never name **Miscellaneous** as Penny’s pick
  unless the user explicitly asked about that category. Unknown/unidentifiable spend →
  **Uncategorized** (review), not Miscellaneous.
- Cannot manually add transactions—accounts must be linked.
- Cannot create custom categories—only names from the reference above exist. If the user
  names a non-existent label (e.g. “Household”), map to the closest real subcategory
  (e.g. **Upkeep**, **Gadgets**, **Clothing**, **Kids**, or **Pets**—never **Miscellaneous**
  unless they asked for it; **Home** is rent/HOA/property tax only—loan principal → **Transfers**).
- Subcategories cannot be moved to a different parent—the taxonomy is fixed. Example:
  **Connectivity** stays under **Bills** and is never under **Shelter**. Users reassign
  **transactions** to other **subcategories** (leaves), not the parent grouping of a leaf.
- Multi-category spend: on a transaction, open **Each Transaction's Category**, tap
  **Split It Up**, then use the sliders to split across categories. Reach that screen
  via any path that opens a transaction detail, then category (see **Ways to open “change category”** above).

## Output
Write **one** plain-text answer: informative, tight, and complete. Use **bold** around real
in-app labels when it helps scanning (tabs, sections, buttons). Do not wrap the whole answer
in JSON or markdown code blocks.

**“Where” / navigation:** Lead with the shortest correct path (tab → section). For “list all
ways” questions, give a **numbered** list (1. 2. …) of every path from **Ways to open “change
category”**—these are routes, not dotted map codes (never `1.1.1.5`). For informal labels like
“Household,” map to **Upkeep** or **Shopping** leaves (**Gadgets**, **Clothing**, etc.)—never
**Miscellaneous** unless asked.

**Chat / Insights / iMessage:** Name **app** (**Penny Chat**, **Goal-Specific Penny Chat**) and
**iMessage** when asked how to reach Penny. If asked **why** iMessage messages appear: Penny is
**linked to iMessage** for convenience—messages to/from Penny in the app are **reflected on
iMessage too** (and vice versa). For missing in-app reports, **ask Penny in chat**; for stored
highlights/reminders/alerts, **Insights** tab; bad notification → **Report Issue**.

**Forbidden in the answer:** Any dotted-decimal map pattern (e.g. `1.1`, `2.1.4.1`, `1.1.1.5.3`).
If you need to cite a screen, use its **visible name** only (e.g. "Split It Up",
"Recent Transactions", "Net Worth").

**Good:** Open **Home**, scroll to your account chart (Y/Y, M/M, or W/W), tap
**Recent Transactions**, open the transaction, then under **Each Transaction's Category**
tap **Split It Up**.

**Bad:** "Go to 1.1.1.5.3" or "use section 2.1"—never.
"""

class PennyAppUsageInfoOptimizer:
    """Handles Gemini API interactions for answering Hey Penny app usage questions."""

    def __init__(self, config: Optional[dict] = None):
        """Initialize the Gemini agent with API configuration."""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        self.client = genai.Client(api_key=api_key)

        cfg = dict(DEFAULT_CONFIG)
        if config:
            cfg.update({k: v for k, v in config.items() if k != "gen_config"})
            if "gen_config" in config and isinstance(config["gen_config"], dict):
                merged_gen = dict(DEFAULT_CONFIG["gen_config"])
                merged_gen.update(config["gen_config"])
                cfg["gen_config"] = merged_gen

        self.config = cfg
        self.json_mode = bool(self.config["json"])
        self.sanitize = bool(self.config["sanitize"])
        self.model_name = self.config["model_name"]
        self.temperature = self.config["gen_config"]["temperature"]
        self.top_p = self.config["gen_config"]["top_p"]
        self.top_k = self.config["gen_config"]["top_k"]
        self.max_output_tokens = self.config["gen_config"]["max_output_tokens"]
        self.thinking_budget = self.config["gen_config"]["thinking_budget"]
        
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

        Returns a dict with ``reply`` (plain text when ``json`` is false, else parsed JSON)
        plus ``thought_summary`` when the API emits thought parts (same streaming extraction
        pattern as ``AccountNameVerbalizerOptimizer``).
        """
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_input)])]
        
        gen_config_kwargs = dict(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_output_tokens=self.max_output_tokens,
            safety_settings=self.safety_settings,
            system_instruction=[types.Part.from_text(text=self.system_prompt)],
            thinking_config=types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
                include_thoughts=True,
            ),
        )
        if self.json_mode:
            gen_config_kwargs["response_schema"] = self.output_schema
            gen_config_kwargs["response_mime_type"] = "application/json"

        generate_content_config = types.GenerateContentConfig(**gen_config_kwargs)

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

        text = output_text.strip()

        if self.sanitize:
            if text.startswith("```json"):
                text = text[7:-3].strip()
            elif text.startswith("```"):
                text = text[3:-3].strip()

        if self.json_mode:
            text = output_text.strip()
            if self.sanitize:
                if text.startswith("```json"):
                    text = text[7:-3].strip()
                elif text.startswith("```"):
                    text = text[3:-3].strip()
            parsed = json.loads(text)
        else:
            parsed = {"reply": text}

        parsed["thought_summary"] = thought_summary.strip()
        return parsed

# Four batches for iterative prompt tuning (batch 2 has three cases).
BATCH_TEST_CASES = {
    1: [
        {
            "name": "Food category options",
            "input": "What category options do I have for food?",
            "ideal_output": "Food encompasses the **Meals** category, which includes **Dining Out**, **Delivered Food**, and **Groceries**."
        },
        {
            "name": "Alcohol categorization",
            "input": "Would alcohol be categorized by Penny as entertainment or dining out?",
            "ideal_output": "Alcohol is categorized under **Leisure** → **Entertainment**."
        },
        {
            "name": "Late credit card fee categorization",
            "input": "Where are late credit card payment fees categorized?",
            "ideal_output": "**Service Fees**, since this is essentially a fee to borrow money."
        },
    ],
    2: [
        {
            "name": "Net worth location",
            "input": "Where can I see my net worth?",
            "ideal_output": "You can view your net worth by navigating to the **Account** tab and selecting the **Net Worth** section."
        },
        {
            "name": "Savings goal progress",
            "input": "Where can I track my progress on my 'New Car' savings goal?",
            "ideal_output": "Track your progress by opening the **Goals** tab and checking the **Active Goals** section."
        },
        {
            "name": "Change category access",
            "input": "What are the different ways to access the page that allows me to change a transaction's category?",
            "ideal_output": "You can reach **Each Transaction's Category** through several paths: \n1. **Home** → account totals → chart → **Recent Transactions** → tap a row.\n2. **Home** → **Income vs. Spending** → **Year to Year/Month to Month/Week to Week** tab → **Top Transaction Contributors** → tap a row.\n3. **Home** → **Account Balances** → chart → **Recent Transactions** → tap a row.\n4. **Home** → **Number of Transactions Categorized by Penny** → **Search Bar** → tap a row.\n5. **Home** → **Transactions Needing User's Review** → tap a row.\n6. **Account** tab → **Net Worth** → **Credit vs. Savings Accounts Breakdown** → chart → **Recent Transactions** → tap a row.\n7. **Account** → **Credit** or **Savings** tab → chart → **Recent Transactions** → tap a row."
        },
    ],
    3: [
        {
            "name": "Manual transaction support",
            "input": "How do I manually add a transaction?",
            "ideal_output": "Manual addition of transactions is not supported; accounts must be linked via Plaid for automatic tracking."
        },
        {
            "name": "Custom category support",
            "input": "Can I create a custom category for my 'Hobby' expenses?",
            "ideal_output": "You cannot create custom categories. Hobby expenses should be mapped to the **Leisure** → **Entertainment** subcategory."
        },
        {
            "name": "iMessage messages from Penny",
            "input": "Why am I getting iMessage messages from Penny?",
            "ideal_output": "Penny can be linked to the user's iMessage for convenience. Any messages sent to/from Penny on the app would be reflected on iMessage too."
        },
    ],
    4: [
        {
            "name": "Transaction splitting",
            "input": "How do I split a transaction between Groceries and Household?",
            "ideal_output": "There is no 'Household' category; you should use **Shopping** or **Shelter** → **Upkeep**. To split a transaction, open it to see the **Each Transaction's Category** screen and tap **Split It Up** to use the sliders."
        },
        {
            "name": "Reviewing transactions",
            "input": "Where do I find transactions that need my review?",
            "ideal_output": "Transactions needing review are located on the **Home** screen in the **Transactions Needing User's Review** section."
        },
    ],
}


def run_batch(batch_num: int, optimizer: Optional[PennyAppUsageInfoOptimizer] = None) -> None:
    cases = BATCH_TEST_CASES[batch_num]
    opt = optimizer or PennyAppUsageInfoOptimizer()
    print(f"\n=== Batch {batch_num} ===")
    for case in cases:
        name = case["name"]
        user_input = case["input"]
        ideal = case["ideal_output"]

        print(f"\n## Test Name: {name}\n")
        print(f"**Input**: {user_input}\n")

        result = opt.generate_response(user_input)
        reply = result["reply"]

        print(f"**Output**: {reply}\n")
        print(f"### Ideal Expected Output\n")
        print(f"{ideal}\n")


def test_optimizer():
    optimizer = PennyAppUsageInfoOptimizer()
    for batch_num in sorted(BATCH_TEST_CASES.keys()):
        run_batch(batch_num, optimizer)


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
