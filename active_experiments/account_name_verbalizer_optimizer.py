from google import genai
from google.genai import types
import json
import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SCHEMA = types.Schema(
    type=types.Type.ARRAY,
    items=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "id": types.Schema(type=types.Type.INTEGER, description="Same as input id"),
            "purpose_name": types.Schema(
                type=types.Type.STRING,
                description="Optional. How the user identifies the account by purpose—only when customization is evident in inputs; omit key otherwise",
            ),
            "crisp_name": types.Schema(
                type=types.Type.STRING,
                description="Purpose (if any) plus product name—how the user distinguishes this account by use + product (see system prompt)",
            ),
            "bank_added_name": types.Schema(
                type=types.Type.STRING,
                description="Purpose (if any) plus bank—how the user distinguishes this account by use + institution (see system prompt)",
            ),
        },
        required=["id", "crisp_name", "bank_added_name"],
    ),
)

SYSTEM_PROMPT = r"""You return one JSON object per input row.

**Input (each row):** `id`, `account_name`, `bank_name` (nullable), `long_account_name`. These fields are **not standardized**: purpose cues, product wording, and bank names can appear **in any field** or be split across them. Read **`account_name`**, **`long_account_name`**, and **`bank_name` together as one picture**—do not assume a fixed slot per concept.

**Output (key order in each object):** `id` → `purpose_name` (only when customized—omit the key otherwise; never `"purpose_name": null`) → `crisp_name` → `bank_added_name`. Always emit `crisp_name` and `bank_added_name`. When `purpose_name` is included, it must appear **between** `id` and `crisp_name` in the JSON object.

---

## 1. Identify (per row)

Synthesize from the **entire row** (all text fields). Extract three things; each may be evidenced **anywhere** in `account_name`, `long_account_name`, or `bank_name`—including duplicates, fragments, or legal names you normalize away.

- **Purpose** — User-specific **use, goal, or nickname** if the **combined** text shows customization beyond a plain generic product line. If, taken as a whole, the row is just standard product naming for that FI, **there is no purpose** (uncustomized).
- **Product name** — Short, accurate **product/card/loan** wording **inferred from the full inputs** (not only `account_name`): strip mask tails (`…1234`, `****`); use real tokens (e.g. Total Checking, Sapphire Preferred, 360 Savings). No leading **institution** in `crisp_name` except true in-product sub-brands (`360`, `Venture`, etc.).
- **Bank** — **Short consumer brand**, taken from whichever field(s) carry the FI (often `bank_name`, but sometimes only `long_account_name` or `account_name`). Infer when missing (`Chase` not `JPMorgan Chase`; `Marcus` not `Marcus by Goldman Sachs`; keep `Ally Bank`, `Bank of America`, `Charles Schwab` when that is how people say the bank).

Ignore masked number tails when reasoning. Do not invent a purpose the inputs do not support.

---

## 2. Build (then apply step 3)

**`purpose_name` (optional in JSON)** — The **purpose string alone**: how the user identifies the account **by purpose**. Output this key **only** when step 1 found customization. It is **not** emitted when the account is uncustomized.

- Shortest faithful phrase from the display (you will **Title Case** it in step 3).
- **Never** use a generic product type **by itself** (solo Checking, Savings, Credit Card, Mortgage, Brokerage, Loan).

**`crisp_name`** — How the user identifies the account **by purpose and product name**.

- **No purpose:** product only — e.g. **`Total Checking`**, **`Sapphire Preferred`** (minimal tokens; disambiguate similar rows with the fewest real product words from inputs).
- **With purpose:** **purpose + product** — every word of `purpose_name` **verbatim once**, plus the smallest set of real product tokens needed — e.g. **`Gabby's Total Checking`**, **`Europe Trip 360 Savings`**. Keep possessives coherent (`Gabby's …`); for business-style labels, tight compounds like **`Citi Business`** / **`Business Citi`** are both OK—pick the shorter clear form.

**`bank_added_name`** — How the user identifies the account **by purpose and bank**.

- **No purpose:** **bank only** — e.g. **`Citi`**, **`Chase`** (no product words).
- **With purpose:** **purpose + bank**, **fewest words**, prefer **juxtaposition** over filler — e.g. **`Gabby's Citi`**, **`Truist Business Petty Cash`**. Prefer **`Citi Business`** over **`Business at Citi`**. **Personal / possessive:** **`Gabby's Chase`**, not **`Chase Gabby's`**. Avoid wordy **`… at [Bank]`** when a short compound is clear.

---

## 3. Clean up

Apply to every output string:

- **Title Case** with a **proper title shape** (capitalize major words; small words like prepositions stay lower when conventional unless they start the string).
- Preserve **brand/product casing** where standard (`360`, `Venture`, `Sapphire Preferred`).
- **Concision:** as short as possible without dropping required purpose/product/bank content.
- **Forbidden:** substring **Account** in any field. Preserve `id`.

---

<EXAMPLES>
- Uncustomized: product `Total Checking`, bank `Chase` → omit `purpose_name`; `crisp_name` **`Total Checking`**; `bank_added_name` **`Chase`**.
- Personal: purpose `Gabby's`, product `Total Checking`, bank `Citi` → `purpose_name` **`Gabby's`** (or as in source); `crisp_name` **`Gabby's Total Checking`**; `bank_added_name` **`Gabby's Citi`** (not **`Citi Gabby's`**).
- Business + bank: purpose `Business Petty Cash`, bank `Truist`, product tokens from inputs → `bank_added_name` **`Truist Business Petty Cash`** or compact equivalent; avoid **`Business Petty Cash at Truist`** when unnecessary.
</EXAMPLES>
"""

class AccountNameVerbalizerOptimizer:
    """Handles Gemini API interactions for simplifying and renaming bank account names."""

    def __init__(self, model_name="gemini-flash-lite-latest"):
        """Initialize the Gemini client with API configuration."""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        self.client = genai.Client(api_key=api_key)
        
        self.model_name = model_name
        self.temperature = 0
        self.top_p = 0.95
        self.top_k = 40
        self.max_output_tokens = 2048
        self.thinking_budget = 0
        
        self.safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
        ]
        
        self.system_prompt = SYSTEM_PROMPT
        self.output_schema = SCHEMA

    def generate_response(self, accounts: list) -> list:
        """
        Generate simplified names for a list of bank accounts.
        """
        user_input = json.dumps(accounts, indent=2)
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_input)])]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_output_tokens=self.max_output_tokens,
            safety_settings=self.safety_settings,
            system_instruction=[types.Part.from_text(text=self.system_prompt)],
            thinking_config=types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
                include_thoughts=True
            ),
            response_schema=self.output_schema,
            response_mime_type="application/json"
        )

        # Generate response using streaming to extract thoughts
        output_text = ""
        thought_summary = ""
        
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
                    if hasattr(candidate, 'content') and candidate.content:
                        if hasattr(candidate.content, 'parts') and candidate.content.parts:
                            for part in candidate.content.parts:
                                if hasattr(part, 'thought') and part.thought:
                                    if hasattr(part, 'text') and part.text:
                                        if thought_summary:
                                            thought_summary += part.text
                                        else:
                                            thought_summary = part.text
        
        if thought_summary:
            print(f"{'='*80}")
            print("THOUGHT SUMMARY:")
            print(thought_summary.strip())
            print("="*80)
            
        if not output_text:
            raise ValueError("Empty response from model.")
            
        try:
            return json.loads(output_text)
        except json.JSONDecodeError:
            text = output_text.strip()
            if text.startswith("```json"):
                text = text[7:-3].strip()
            elif text.startswith("```"):
                text = text[3:-3].strip()
            return json.loads(text)


def mechanical_sandbox_check(original_input: list, verbalizer_output: list) -> dict:
    """
    Lightweight deterministic checks: id alignment; forbid 'Account' in string outputs;
    when purpose_name is present, crisp_name must contain every purpose word and bank_added_name must contain the full purpose phrase.
    Does not judge whether purpose_name should exist (no custom-label field in input).
    """
    issues = []
    in_by_id = {r["id"]: r for r in original_input}
    out_by_id = {r["id"]: r for r in verbalizer_output}
    if set(in_by_id) != set(out_by_id):
        issues.append("Output id set does not match input.")

    for iid in in_by_id:
        row = out_by_id.get(iid)
        if not row:
            issues.append(f"ID {iid}: missing output row.")
            continue
        purpose_key = "purpose_name" in row and row["purpose_name"] is not None

        for field in ("crisp_name", "bank_added_name", "purpose_name"):
            if field not in row or row[field] is None:
                if field == "purpose_name":
                    continue
                issues.append(f"ID {iid}: missing required field {field}.")
                continue
            val = row[field]
            if not isinstance(val, str):
                continue
            if "Account" in val:
                issues.append(f"ID {iid}: substring 'Account' forbidden in {field}.")

        if purpose_key:
            purpose_raw = row.get("purpose_name")
            if purpose_raw is None or not str(purpose_raw).strip():
                issues.append(f"ID {iid}: purpose_name present but empty.")
            else:
                purpose = str(purpose_raw).strip()
                p_words = purpose.lower().split()
                crisp = str(row.get("crisp_name", "")).lower()
                if not all(w in crisp for w in p_words):
                    issues.append(f"ID {iid}: crisp_name must contain every word of purpose_name.")
                bank_added = str(row.get("bank_added_name", "")).strip()
                if purpose.lower() not in bank_added.lower():
                    issues.append(
                        f"ID {iid}: bank_added_name must include full purpose_name verbatim; got {bank_added!r}."
                    )

    good = len(issues) == 0
    return {
        "good_copy": good,
        "info_correct": good,
        "eval_text": "\n".join(f"- {x}" for x in issues) if issues else "",
    }


def test_optimizer(batch_index=None, run_sandbox=True):
    optimizer = AccountNameVerbalizerOptimizer()

    test_cases = [
        # Batch 0: same FI, mixed products + one purpose-style display name
        [
            {"id": 9512, "account_name": "Capital One 360 Checking", "bank_name": None, "long_account_name": "Capital One 360 Checking"},
            {"id": 9514, "account_name": "Europe Trip Capital One 360 Savings", "bank_name": None, "long_account_name": "Capital One 360 Savings"},
            {"id": 9513, "account_name": "Capital One Venture Rewards", "bank_name": "Capital One", "long_account_name": "Capital One Venture Rewards"},
        ],
        # Batch 1: heterogeneous + use-specific display name
        [
            {"id": 101, "account_name": "CHASE SAPPHIRE PREFERRED", "bank_name": "JPMorgan Chase", "long_account_name": "CHASE SAPPHIRE PREFERRED VISA SIGNATURE"},
            {"id": 102, "account_name": "Emergency Buffer", "bank_name": "Marcus by Goldman Sachs", "long_account_name": "Marcus High Yield Savings Account"},
            {"id": 103, "account_name": "Chase Secure Banking", "bank_name": "Chase", "long_account_name": "Chase Secure Banking"},
            {"id": 104, "account_name": "Brokerage Account", "bank_name": "Charles Schwab", "long_account_name": "Schwab Individual Brokerage Account"},
        ],
        # Batch 2: generic rows + business petty cash in display name
        [
            {"id": 201, "account_name": "Chase First Banking", "bank_name": "Chase", "long_account_name": "Chase First Banking"},
            {"id": 202, "account_name": "Savings", "bank_name": "Bank of America", "long_account_name": "Bank of America Advantage Savings"},
            {"id": 3217, "account_name": "Business Petty Cash Truist Core Checking", "bank_name": "Truist", "long_account_name": "Core Checking - 7891"},
        ],
        # Batch 3: loans + car payment in display name
        [
            {"id": 301, "account_name": "Mortgage", "bank_name": "Rocket Mortgage", "long_account_name": "Rocket Mortgage Conventional Fixed"},
            {"id": 302, "account_name": "Ally Car Payment Auto Loan", "bank_name": "Ally Bank", "long_account_name": "Ally Auto Finance Loan"},
            {"id": 303, "account_name": "Student Loan", "bank_name": "Navient", "long_account_name": "Navient Federal Student Loan"},
        ],
    ]
    
    if batch_index is not None:
        cases_to_run = [test_cases[batch_index]]
        start_idx = batch_index + 1
    else:
        cases_to_run = test_cases
        start_idx = 1

    for i, accounts in enumerate(cases_to_run):
        print(f"\n--- Test Case {start_idx + i} ---")
        print("Input:")
        print(json.dumps(accounts, indent=2))
        try:
            result = optimizer.generate_response(accounts)
            print("\nOutput:")
            print(json.dumps(result, indent=2))
            if run_sandbox:
                sand = mechanical_sandbox_check(accounts, result)
                print("\n--- Sandbox (mechanical) ---")
                print(json.dumps(sand, indent=2))
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else None
    test_optimizer(batch)
