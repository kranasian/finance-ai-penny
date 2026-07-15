from __future__ import annotations

import argparse
import json
import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

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

# Input
Bank account data. Ignore differentiation among `account_name`, `long_account_name`, and `bank_name`. Look at all three as a whole.

# Output
- `id`
- `purpose_name`
- `crisp_name`
- `bank_added_name`

# Processing

## 1. Identify bank name, product name, account type, and purpose.

Do not consider which part of the input the detail came from since the input may be mixed up. (Ex: bank name is not necessarily from `bank_name`)

- **Bank Name:** Commonly-known name of the bank that the account uses.
- **Product Name:** Brand of the bank's product that the account uses, excluding the bank's name.
- **Account Type:** Type of financial tool that the account is (eg. Checking, Credit Card, Auto Loan)
- **Customized Name (if any):** Name set by the accountholder. Blank if there are no identifiers additional to the bank name, product name, and account type.

## 2. Build the three name options.
1. `purpose_name`: customized name (do not output purpose_name at all if none; never `null`)
2. `crisp_name`: customized name + product name
3. `bank_added_name`: bank name + product name

### Guidelines
- Each name should independently be able to identify the account.
- Use Title Case while preserving brand casing for bank and product names.
- Keep as short as possible (5 words maximum) while still keeping all relevant information.
- Remove redundancies, if any (ie. word/s that do not add value/specificity to the name).
- Remove the word "Account".
- Remove unnecessary symbols.

## 3. Verification
- Does purpose_name just restate the bank name, product name, or account type? Remove purpose_name if yes.
- Does each outputted name work well even when taken independently? Reprocess output if no.
"""

_TEST_SEPARATOR = "=" * 72
_SECTION_RULE = "-" * 72


def _print_section_banner(title: str) -> None:
    print(f"\n{_SECTION_RULE}\n{title}\n{_SECTION_RULE}\n")


def _parse_expected_output(raw: str | None) -> list[dict[str, Any]] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else None
    except Exception:
        return None


TEST_CASES: list[dict[str, Any]] = [
    {
        "name": "capital_one_mixed_products_europe_trip",
        "batch": 0,
        "input": [
            {"id": 9512, "account_name": "Capital One 360 Checking", "bank_name": "Capital One", "long_account_name": "Capital One 360 Checking"},
            {"id": 9514, "account_name": "Europe Trip Capital One 360 Savings", "bank_name": "Capital One", "long_account_name": "Capital One 360 Savings"},
            {"id": 9513, "account_name": "Capital One Venture Rewards", "bank_name": "Capital One", "long_account_name": "Capital One Venture Rewards"},
        ],
        "output": """[
  {"id": 9512, "crisp_name": "360 Checking", "bank_added_name": "Capital One 360 Checking"},
  {"id": 9514, "purpose_name": "Europe Trip", "crisp_name": "Europe Trip 360 Savings", "bank_added_name": "Capital One 360 Savings"},
  {"id": 9513, "crisp_name": "Venture Rewards", "bank_added_name": "Capital One Venture Rewards"}
]""",
    },
    {
        "name": "chase_sapphire_emergency_buffer_heterogeneous",
        "batch": 1,
        "input": [
            {"id": 101, "account_name": "CHASE SAPPHIRE PREFERRED", "bank_name": "JPMorgan Chase", "long_account_name": "CHASE SAPPHIRE PREFERRED VISA SIGNATURE"},
            {"id": 102, "account_name": "Emergency Buffer", "bank_name": "Marcus by Goldman Sachs", "long_account_name": "Marcus High Yield Savings Account"},
            {"id": 103, "account_name": "Chase Secure Banking", "bank_name": "Chase", "long_account_name": "Chase Secure Banking"},
            {"id": 104, "account_name": "Brokerage Account", "bank_name": "Charles Schwab", "long_account_name": "Schwab Individual Brokerage Account"},
        ],
        "output": """[
  {"id": 101, "crisp_name": "Sapphire Preferred", "bank_added_name": "Chase Sapphire Preferred"},
  {"id": 102, "purpose_name": "Emergency Buffer", "crisp_name": "Emergency Buffer High Yield Savings", "bank_added_name": "Marcus High Yield Savings"},
  {"id": 103, "crisp_name": "Secure Banking", "bank_added_name": "Chase Secure Banking"},
  {"id": 104, "crisp_name": "Individual Brokerage", "bank_added_name": "Charles Schwab Individual Brokerage"}
]""",
    },
    {
        "name": "chase_first_banking_bofa_savings_truist_petty_cash",
        "batch": 2,
        "input": [
            {"id": 201, "account_name": "Chase First Banking", "bank_name": "Chase", "long_account_name": "Chase First Banking"},
            {"id": 202, "account_name": "Savings", "bank_name": "Bank of America", "long_account_name": "Bank of America Advantage Savings"},
            {"id": 3217, "account_name": "Business Petty Cash Truist Core Checking", "bank_name": "Truist", "long_account_name": "Core Checking - 7891"},
        ],
        "output": """[
  {"id": 201, "crisp_name": "First Banking", "bank_added_name": "Chase First Banking"},
  {"id": 202, "crisp_name": "Advantage Savings", "bank_added_name": "Bank of America Advantage Savings"},
  {"id": 3217, "purpose_name": "Business Petty Cash", "crisp_name": "Business Petty Cash Core Checking", "bank_added_name": "Truist Core Checking"}
]""",
    },
    {
        "name": "rocket_mortgage_ally_car_payment_navient_student_loan",
        "batch": 3,
        "input": [
            {"id": 301, "account_name": "Mortgage", "bank_name": "Rocket Mortgage", "long_account_name": "Rocket Mortgage Conventional Fixed"},
            {"id": 302, "account_name": "Ally - Car Payment Auto Loan", "bank_name": "Ally Bank", "long_account_name": "Ally - Auto Finance Loan"},
            {"id": 303, "account_name": "Student Loan", "bank_name": "Navient", "long_account_name": "Navient Federal Student Loan"},
        ],
        "output": """[
  {"id": 301, "purpose_name": "Mortgage", "crisp_name": "Mortgage Conventional Fixed", "bank_added_name": "Rocket Mortgage Conventional Fixed"},
  {"id": 302, "purpose_name": "Car Payment", "crisp_name": "Car Payment Auto Finance Loan", "bank_added_name": "Ally Bank Auto Finance Loan"},
  {"id": 303, "purpose_name": "Student Loan", "crisp_name": "Federal Student Loan", "bank_added_name": "Navient Federal Student Loan"}
]""",
    },
]


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


def _normalize_output_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {"id": row["id"]}
    for field in ("purpose_name", "crisp_name", "bank_added_name"):
        if field in row and row[field] is not None:
            normalized[field] = str(row[field]).strip()
    return normalized


def compare_to_ideal(actual: list, ideal: list) -> list[str]:
    failures: list[str] = []
    ideal_by_id = {r["id"]: _normalize_output_row(r) for r in ideal}
    actual_by_id = {r["id"]: _normalize_output_row(r) for r in actual}
    if set(ideal_by_id) != set(actual_by_id):
        failures.append(f"id set mismatch: got {sorted(actual_by_id)} expected {sorted(ideal_by_id)}")
        return failures

    for iid, want in ideal_by_id.items():
        got = actual_by_id[iid]
        want_fields = set(want) - {"id"}
        got_fields = set(got) - {"id"}
        if want_fields != got_fields:
            failures.append(
                f"ID {iid}: field set mismatch: got {sorted(got_fields)} expected {sorted(want_fields)}"
            )
            continue
        for field in sorted(want_fields):
            if got.get(field) != want.get(field):
                failures.append(
                    f"ID {iid}: {field} {got.get(field)!r} != expected {want.get(field)!r}"
                )
    return failures


def test_optimizer(
    batch_index: int | None = None,
    *,
    run_sandbox: bool = True,
    check: bool = False,
) -> list[str]:
    optimizer = AccountNameVerbalizerOptimizer()
    failures: list[str] = []

    if batch_index is not None:
        cases = [tc for tc in TEST_CASES if int(tc.get("batch", -1)) == int(batch_index)]
        if not cases:
            raise SystemExit(f"No tests found for batch={batch_index}")
    else:
        cases = TEST_CASES

    for i, tc in enumerate(cases):
        if i:
            print(f"\n{_TEST_SEPARATOR}\n")
        batch = tc.get("batch")
        batch_s = str(batch) if isinstance(batch, int) else "—"
        print(f"# Test: {tc['name']}  (batch {batch_s})\n")
        accounts = tc["input"]
        _print_section_banner("Input")
        print(json.dumps(accounts, indent=2))
        try:
            result = optimizer.generate_response(accounts)
            _print_section_banner("Output")
            print(json.dumps(result, indent=2))
            if tc.get("output") is not None:
                _print_section_banner("Expected Output")
                print(tc["output"])
            if run_sandbox:
                sand = mechanical_sandbox_check(accounts, result)
                _print_section_banner("Sandbox (mechanical)")
                print(json.dumps(sand, indent=2))
            if check:
                ideal = _parse_expected_output(tc.get("output"))
                if ideal is None:
                    failures.append(f"{tc.get('name')}: invalid expected output JSON")
                else:
                    mismatches = compare_to_ideal(result, ideal)
                    if mismatches:
                        failures.extend(f"{tc.get('name')}: {line}" for line in mismatches)
        except Exception as e:
            print(f"Error: {e}")
            if check:
                failures.append(f"{tc.get('name')}: {e}")

    if check:
        print(f"\n{_TEST_SEPARATOR}\n")
        print(f"# Total tests: {len(cases)}\n")
        if failures:
            print("# CHECK FAILURES\n")
            for line in failures:
                print(line)
            raise SystemExit(1)
        if cases:
            print("# CHECK: all outputs matched expected.\n")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "batch",
        nargs="?",
        type=int,
        default=None,
        help="Batch index (0-based). Omit to run all batches.",
    )
    parser.add_argument(
        "--no-sandbox",
        action="store_true",
        help="Skip mechanical sandbox checks.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Require model output to match expected JSON per test; exit non-zero on mismatch.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available test cases and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print("Available test cases:")
        for tc in TEST_CASES:
            batch = tc.get("batch")
            batch_s = str(batch) if isinstance(batch, int) else "—"
            print(f"  batch {batch_s}: {tc.get('name')}")
        return

    test_optimizer(
        args.batch,
        run_sandbox=not args.no_sandbox,
        check=args.check,
    )


if __name__ == "__main__":
    main()
