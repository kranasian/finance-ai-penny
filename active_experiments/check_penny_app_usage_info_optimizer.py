from google import genai
from google.genai import types
import json
import os
from typing import Optional

from dotenv import load_dotenv

from penny_app_usage_info_optimizer import DEFAULT_CONFIG

load_dotenv()

CHECKER_OUTPUT_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "good_copy": types.Schema(
            type=types.Type.BOOLEAN,
            description="True if format/style rules pass.",
        ),
        "info_correct": types.Schema(
            type=types.Type.BOOLEAN,
            description=(
                "True if REVIEW_NEEDED is not inconsistent with the Hey Penny ground-truth rules "
                "in the checker system prompt."
            ),
        ),
        "eval_text": types.Schema(
            type=types.Type.STRING,
            description="Empty if both true; else concise fix bullets with single-quoted snippets.",
        ),
    },
    required=["good_copy", "info_correct", "eval_text"],
)

CHECKER_SYSTEM_PROMPT = r"""#### 1. Goal
Evaluate plain-text Hey Penny app usage model output (`REVIEW_NEEDED`) against user input
(`EVAL_INPUT`). You do **not** receive the full authoring system prompt—only the ground-truth
rules in **Section 3** below. Decide if `REVIEW_NEEDED` is acceptable and give concise fixes when not.

#### 2. Core Task
- Compare `REVIEW_NEEDED` to `EVAL_INPUT` using **Section 3** and the checks in **Sections 4–5**.
- Output a JSON object: `{"good_copy": boolean, "info_correct": boolean, "eval_text": string}`.

#### 3. Ground truth (Hey Penny usage — do not contradict)

**Product & behavior**
- Hey Penny: AI finance assistant; expense tracking, categorization, subscription-related help;
  chat-style interface (e.g. iMessage); low-effort “passive” money management.
- Accounts linked via **Plaid**; **manual addition of transactions is not supported**.
- Transactions are categorized into **subcategories** (taxonomy leaves). Parent groupings exist
  for rollups in analyses/charts, not as a separate pick instead of a leaf.
- User may change a transaction’s **subcategory** later among real leaves; leaves **cannot** be
  “moved” to another parent in the taxonomy (fixed hierarchy).
- Spending views include **Y/Y** (year to year), **M/M** (month to month), **W/W** (week to week)
  where relevant.
- Categorization accuracy is described as **over 90%** when that fact is relevant.

**Output style the usage model targets (for your `good_copy` check)**
- **Plain text** for the end user: no outer JSON object, no markdown code fence wrapping the
  whole answer, no long meta preambles (e.g. “Here is the answer:”).
- **Forbidden in user-facing copy:** dotted-decimal **map codes** (e.g. `1.1`, `2.1.4.1`,
  `1.1.1.5.3`) anywhere—including parentheses. Internal prompt numbering is not shown in the app.
- Prefer **real in-app labels** for navigation (tabs, sections, buttons). **Bold** around labels
  is fine; do not fail `good_copy` solely for reasonable bolding.
- Tone: dense facts, minimal chit-chat (another layer may add tone).

**Tabs & major surfaces (labels to expect)**
- Bottom tabs include **Home**, **Account**, **Goals** (and related sections below).
- **Home:** cash/credit totals; line charts per account (**Year to Year** / **Month to Month** /
  **Week to Week** where applicable); **Recent Transactions**; **Income vs. Spending** with
  period tabs (**Year to Year**, **Month to Month**, **Week to Week**); **Actual vs Expected**;
  **Subcategory Breakdown**; **Top Transaction Contributors**; **Penny Chat**; **Account Balances**;
  **Number of Transactions Categorized by Penny** with **Search Bar**; **Transactions Needing
  User's Review**.
- **Account:** **Net Worth** (with Y/Y, M/M, W/W charts); **Credit vs. Savings Accounts Breakdown**;
  **Credit** tab; **Savings** tab—drill paths mirror chart → recent transactions → transaction
  detail → category where applicable.
- **Goals:** **Add Goal** / **'+ Add Goal'**; **Active Goals**; **Past Goals**; goal settings
  (e.g. change title, edit amount, end goal); goal-specific Penny chat where applicable.
- **Insights:** **Love It**; **Report Issue**; **Hide This**.

**Transaction detail & category**
- Transaction row/detail includes editable **Transaction Name**, **Amount**, **Account**, **Date**,
  **Status** (e.g. Pending, Duplicate, Category Confirmation, Confirmed); period bar charts; list of
  same-establishment transactions as applicable.
- **Each Transaction's Category**; **Top 5 Likely Categories**; **More Categories** → full list;
  **Split It Up** → sliders to split one transaction across multiple categories.

**Ways to reach “change category” (user-facing phrases)**
Include paths that match the product when evaluating “where/how”: **Home** → account totals →
chart → **Recent Transactions** → row → **Each Transaction's Category**; **Home** → **Income vs.
Spending** → period tab → **Top Transaction Contributors** → row; **Home** → **Account Balances**
→ chart → **Recent Transactions**; **Home** → **Number of Transactions Categorized by Penny** →
search → row; **Home** → **Transactions Needing User's Review** → item; **Account** → **Net Worth**
→ **Credit vs. Savings Accounts Breakdown** → chart → recent transactions → row; **Account** →
**Credit** or **Savings** → same chart/transaction path as appropriate.

**Valid subcategory names (only these leaves; no custom categories)**
Dining Out; Delivered Food; Groceries; Entertainment; Travel & Vacations; Connectivity; Insurance;
Taxes; Service Fees; Home; Utilities; Upkeep; Kids Activities; Tuition; Clothing; Gadgets; Kids;
Pets; Public Transit; Car & Fuel; Medical & Pharmacy; Gym & Wellness; Personal Care;
Donations & Gifts; Miscellaneous; Uncategorized; Transfers; Salary; Side-Gig; Business; Interest.

**Category facts (selected)**
- Alcohol etc. → **Entertainment** (not a separate “alcohol” leaf).
- Auto insurance → **Car & Fuel** (not **Insurance** leaf for that case).
- Rent, mortgage, property tax, homeowners insurance, HOA → **Home**; general home supplies /
  maintenance consumables often **Upkeep** or shopping-type leaves—not treating **Home** as
  “any household spend.”
- **Connectivity** is a Bills-area leaf, not under shelter-style groupings in the taxonomy.

**Hard constraints**
- No manual transaction entry; linking required.
- **No custom categories**—map informal user labels to the closest real leaf (e.g. no “Household”
  as its own official leaf).
- Multi-category: **Each Transaction's Category** → **Split It Up** → sliders.

#### 4. `good_copy`
`true` if **Section 3** output-style rules pass: plain text contract, no map codes in the answer, no
whole-answer JSON/fence wrapper, reasonable use of real UI labels. `false` if any violation.

#### 5. `info_correct`
Judge whether `REVIEW_NEEDED` is **consistent** with **Section 3**. `REVIEW_NEEDED` may be **correct**
even when it does not restate every rule explicitly—**reasonable factual assumptions** and
compact summaries are **OK**. Mark `info_correct` **false** only when the text **contradicts**
**Section 3** (e.g. claims manual add exists, invents a screen or category name, wrong core navigation
for a direct “where” question, or states a capability the product does not have). When
`EVAL_INPUT` is ambiguous, prefer **lenient** `info_correct` unless there is a clear contradiction.

#### 6. Output schema
- `good_copy`: `true` if all **Section 4** checks pass.
- `info_correct`: `true` if **Section 5** passes.
- `eval_text`: **Must be `""`** if both are `true`. If either is `false`, one string (use `\n`
  between bullets), **single quotes** inside for snippets, concise, **start with the FIX action**,
  format: `- <Issue>: <Action> (Expected: <correct fact or phrasing>)`
"""


class PennyAppUsageInfoChecker:
    """Evaluates `REVIEW_NEEDED` from Penny usage-style models against embedded ground-truth rules."""

    def __init__(self, config: Optional[dict] = None):
        api_key = os.getenv("GEMINI_API_KEY")
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
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ]

        self.system_prompt = CHECKER_SYSTEM_PROMPT
        self.output_schema = CHECKER_OUTPUT_SCHEMA

    def check_output(self, eval_input: str, review_needed: str) -> dict:
        """Check `review_needed` against `eval_input` using embedded ground-truth rules."""
        request_text = f"""<EVAL_INPUT>

{eval_input}

</EVAL_INPUT>

<REVIEW_NEEDED>

{review_needed}

</REVIEW_NEEDED>

Output:"""

        contents = [
            types.Content(role="user", parts=[types.Part.from_text(text=request_text)])
        ]

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
            print(f"{'=' * 80}")
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

        try:
            if self.json_mode:
                return json.loads(text)

            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(text[json_start:json_end])
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON response: {e}\nResponse text: {output_text}"
            ) from e


def test_checker() -> None:
    checker = PennyAppUsageInfoChecker()

    test_cases = [
        {
            "name": "Clean navigation answer",
            "eval_input": "Where can I see my net worth?",
            "review_needed": (
                "Open the **Account** tab, then **Net Worth**. You can switch **Year to Year**, "
                "**Month to Month**, or **Week to Week** for the line charts."
            ),
        },
        {
            "name": "Map codes in reply (bad copy)",
            "eval_input": "How do I split a transaction?",
            "review_needed": (
                "Go to section **1.1.1.5.3** and use the sliders after opening **Split It Up**."
            ),
        },
        {
            "name": "Hallucinated feature",
            "eval_input": "How do I manually add a cash expense?",
            "review_needed": (
                "Tap **Home**, then **Add Transaction**, and enter the amount. Penny saves it "
                "without linking a bank."
            ),
        },
    ]

    for i, case in enumerate(test_cases):
        print(f"\n--- Test Case {i + 1}: {case['name']} ---")
        try:
            result = checker.check_output(case["eval_input"], case["review_needed"])
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    test_checker()
