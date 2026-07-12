"""
Optimizer runner for `P:RationalizedPennyInsightsVerbalizer`.

Run from `finance-ai-penny` repo root:

  python3 active_experiments/rationalized_penny_insights_verbalizer_optimizer.py --test 0
  python3 active_experiments/rationalized_penny_insights_verbalizer_optimizer.py --batch 1
  python3 active_experiments/rationalized_penny_insights_verbalizer_optimizer.py --test all --no-thinking
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from typing import Any, Dict

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import ClientError
except Exception:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]
    ClientError = Exception  # type: ignore[misc,assignment]

if load_dotenv is not None:
    load_dotenv()

GEMINI_2_5_FLASH_MODEL = "gemini-flash-lite-latest"


def _extract_stream_chunk_text(chunk: Any) -> str:
    """Build text from a streaming chunk without reading ``chunk.text`` first.

    ``chunk.text`` triggers noisy SDK warnings when responses mix ``thought_signature``
    or other non-text parts with JSON text (google-genai streaming).
    """
    pieces: list[str] = []
    for cand in getattr(chunk, "candidates", None) or []:
        content = getattr(cand, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "thought", False):
                continue
            t = getattr(part, "text", None)
            if isinstance(t, str) and t:
                pieces.append(t)
    if pieces:
        return "".join(pieces)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*non-text parts in the response.*")
        agg = getattr(chunk, "text", None)
        return agg if isinstance(agg, str) else ""


def _build_output_schema() -> "types.Schema":
    if types is None:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency for Gemini optimizers. Install `google-genai` in this environment."
        )
    return types.Schema(
        type=types.Type.OBJECT,
        required=["insight", "insight_correct"],
        property_ordering=["insight", "insight_correct"],
        properties={
            "insight": types.Schema(
                type=types.Type.STRING,
                description=(
                    "Single-line Penny message from Drivers. Completeness over brevity: include "
                    "timeframe, direction, and driving establishments. Category link and amount "
                    "must share the same g or r color. ≥1 emoji (2 when long); no newlines."
                ),
            ),
            "insight_correct": types.Schema(
                type=types.Type.BOOLEAN,
                description=(
                    "false for any Drivers↔Insight inconsistency, including on-track/usual vs "
                    "significantly up/down; true only when Drivers fully affirm Insight. "
                    "Never mention inconsistency in insight."
                ),
            ),
        },
    )

def format_insight_input_for_llm(insight: Dict[str, Any]) -> str:
    """Build Markdown input for the model from structured fields (type, insight, drivers).

    Uses top-level Markdown headings `# Type`, `# Insight`, `# Drivers` (exact spellings).
    """
    for k in ("type", "insight", "drivers"):
        if k not in insight:
            raise ValueError(f"insight missing required key: {k!r}")
    return (
        f"# Type\n\n{insight['type']}\n\n"
        f"# Insight\n\n{insight['insight']}\n\n"
        f"# Drivers\n\n{insight['drivers']}"
    )


SYSTEM_PROMPT = """## Persona

You are Penny, a friendly personal finance advisor. Text a trusted friend: warm, direct, natural — never stiff or telegraphic.

## Objective

Return JSON only with keys in order: `insight`, then `insight_correct`.

## Priority

Required content beats brevity. Never omit **direction**, **timeframe**, or **driving establishments** (when Drivers name them) to stay short.

## Output

- **`insight`:** One line from Drivers covering the required fields. ≥1 emoji (second when long). No contrast with Insight.
- **`insight_correct`:** **false** whenever Drivers and Insight disagree in any way — including Insight saying significantly up/down while Drivers say on track / usual / typical / not a drop. **true** only when Drivers affirm Insight's direction and amount with no conflict.

## Format

- **Category link:** `g{[Display](/slug/period)}` or `r{[Display](/slug/period)}` — keep `[Display](/slug/period)` intact. Path `/weekly` if Type has `week_`, else `/monthly`.
- **Category amount:** **is at** / **was at** (or **only at** at `$0`) then `g{$N}` / `r{$N}` for the focal-window total — whole dollars, no commas, no cadence bands.
- **Same color:** link `g`/`r` and amount `g`/`r` must match exactly.
- **Color choice:** Outflows lower → **g**, higher → **r**. Inflows invert (zero/lower → **r**, higher → **g**). As expected / on track / usual nonzero outflow → **g**. Inflow quiet/zero → **r**.
- **Lead:** colored category link first.
- **Timeframe:** only **this month** / **last month** / **this week** / **last week**.

## Process

1. **Focus:** One Insight category; Drivers only. Prefer the active leaf.
2. **Decide insight_correct first:** if Drivers conflict with Insight on direction, amount, timing, or story → **false**; else **true**.
3. **Compose insight from Drivers:** direction (on track / usual / typical ⇒ **as expected**); timeframe; establishments (merchants/institutions only). For `*vs_goal*`, include over/under **limit**. **this** → **is at** only; **last** → **was at** only — no present-tense verbs with **last**.
4. **Verify:** identical `g`/`r` on link and amount; required fields present; focal `$` is the window total Drivers discuss for the period — never a cadence band; quiet/zero windows stay `$0`.

## Slugs (Display→slug)

Outflows: Meals→meals | Dining Out→meals_dining_out | Delivered Food→meals_delivered_food | Groceries→meals_groceries | Leisure→leisure | Entertainment→leisure_entertainment | Travel and Vacations→leisure_travel | Education→education | Kids Activities→education_kids_activities | Tuition→education_tuition | Transport→transportation | Public Transit→transportation_public | Car and Fuel→transportation_car | Health→health | Medical and Pharmacy→health_medical_pharmacy | Gym and Wellness→health_gym_wellness | Personal Care→health_personal_care | Donations and Gifts→donations_gifts | Uncategorized→uncategorized | Miscellaneous→miscellaneous | Bills→bills | Connectivity→bills_connectivity | Insurance→bills_insurance | Taxes→bills_taxes | Service Fees→bills_service_fees | Shelter→shelter | Home→shelter_home | Utilities→shelter_utilities | Upkeep→shelter_upkeep | Shopping→shopping | Clothing→shopping_clothing | Gadgets→shopping_gadgets | Kids→shopping_kids | Pets→shopping_pets | Transfers→transfers

Inflows: Income→income | Salary→income_salary | Sidegig→income_sidegig | Business→income_business | Interest→income_interest
"""


class RationalizedPennyInsightsVerbalizerOptimizer:
    """Gemini runner to iterate on the P:RationalizedPennyInsightsVerbalizer prompt shape."""

    def __init__(
        self,
        model_name: str = GEMINI_2_5_FLASH_MODEL,
        *,
        thinking_budget: int = 1024,
    ):
        if genai is None or types is None:  # pragma: no cover
            raise RuntimeError(
                "Gemini client dependencies not available. Install `google-genai` (and optionally `python-dotenv`)."
            )
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set. Set it in .env or environment.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.thinking_budget = thinking_budget

        self.temperature = 0.5
        self.top_p = 0.95
        self.top_k = 40
        # Enough for JSON + ≤320-char insight; keeps generation cheap vs 4k ceiling.
        self.max_output_tokens = 3072
        self.safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ]

        self.system_prompt = SYSTEM_PROMPT
        self.output_schema = _build_output_schema()
        self.quiet = False

    def generate_response(self, insight_input: str) -> Dict[str, Any]:
        if not isinstance(insight_input, str) or not insight_input.strip():
            raise ValueError(
                "insight_input must be a non-empty string (Markdown with # Type / # Insight / # Drivers)."
            )

        request_text = types.Part.from_text(text=f"input:\n{insight_input.strip()}\n\noutput:")
        contents = [types.Content(role="user", parts=[request_text])]

        cfg = types.GenerateContentConfig(
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
            response_mime_type="application/json",
            response_schema=self.output_schema,
        )

        output_text = ""
        thought_summary = ""
        try:
            for chunk in self.client.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=cfg,
            ):
                piece = _extract_stream_chunk_text(chunk)
                if piece:
                    output_text += piece
                if hasattr(chunk, "candidates") and chunk.candidates:
                    for candidate in chunk.candidates:
                        if hasattr(candidate, "content") and candidate.content:
                            if hasattr(candidate.content, "parts") and candidate.content.parts:
                                for part in candidate.content.parts:
                                    if getattr(part, "thought", False) and getattr(part, "text", None):
                                        t = part.text
                                        thought_summary = (thought_summary + t) if thought_summary else t
        except ClientError as e:
            if self.thinking_budget == 0 and "only works in thinking mode" in (str(e) or ""):
                print(
                    "\n[NOTE] This model requires thinking mode; use default (no --no-thinking) or a different model.",
                    flush=True,
                )
                sys.exit(1)
            raise

        if thought_summary and not self.quiet:
            print("\n" + "-" * 80)
            print("THOUGHT SUMMARY:")
            print("-" * 80)
            print(thought_summary.strip())
            print("-" * 80 + "\n")

        if not output_text or not output_text.strip():
            raise ValueError("Empty response from model. Check API key and model availability.")

        out_clean = output_text.strip()
        if out_clean.startswith("```"):
            lines = out_clean.split("\n")
            out_clean = "\n".join(lines[1:-1]) if len(lines) > 2 else out_clean
        if out_clean.startswith("```json"):
            lines = out_clean.split("\n")
            out_clean = "\n".join(lines[1:-1]) if len(lines) > 2 else out_clean

        result = json.loads(out_clean)
        if not isinstance(result, dict):
            raise ValueError(f"Expected JSON object, got {type(result)}")
        return result  # type: ignore[return-value]


_RE_LINK = re.compile(r"([gr])\{\[([^\]]+)\]\((/[^)]+)\)\}")
_RE_AMOUNT = re.compile(r"([gr])\{(\$[^}]+)\}")
_RE_MALFORMED_LINK = re.compile(r"[gr]\{\$[^[\]]*\]\(")
_RE_BANNED = re.compile(r"\b(currently|is currently)\b", re.I)
_RE_DISCREPANCY_ACK = re.compile(
    r"\b(although|actually|insight flags?|not \$0|rather than|instead of|correction|headline|discrepancy|mismatch)\b",
    re.I,
)
_RE_PARTIAL_TIMEFRAME = re.compile(
    r"\b(first \d+ days|through (the )?first|so far|as of early)\b",
    re.I,
)
_RE_EMOJI = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF](?:\uFE0F)?")
MAX_INSIGHT_WORDS = 20


def _insight_word_count(text: str) -> int:
    """Count words in insight prose, excluding ``{}``, ``()``, and ``[]`` markup."""
    plain = text or ""
    plain = re.sub(r"[gr]\{[^{}]*\}", " ", plain)
    plain = re.sub(r"\([^)]*\)", " ", plain)
    plain = re.sub(r"\[[^\]]*\]", " ", plain)
    plain = _RE_EMOJI.sub("", plain)
    plain = re.sub(r"[^\w\s'$-]+", " ", plain)
    return len([w for w in plain.split() if w])


def _emoji_count(text: str) -> int:
    return len(_RE_EMOJI.findall(text or ""))


def _min_emojis_for_length(char_count: int) -> int:
    if char_count < 140:
        return 1
    if char_count < 220:
        return 2
    return 2


def _strip_markup(text: str) -> str:
    return re.sub(r"[gr]\{([^}]*)\}", r"\1", text or "")


def _expected_period(input_type: str) -> str:
    return "weekly" if "week" in input_type else "monthly"


def _timeframe_phrase(insight_text: str) -> str | None:
    lower = (insight_text or "").lower()
    for phrase in ("last month", "this month", "last week", "this week"):
        if phrase in lower:
            return phrase
    return None


def _tense_issue(plain: str, timeframe: str | None) -> str | None:
    if not timeframe:
        return None
    if timeframe.startswith("last"):
        if re.search(r"\b(is only at|is at|are at|is right|is on)\b", plain, re.I):
            return f"use past tense with {timeframe} (was, landed, hit)"
    elif re.search(r"\b(was only at|was at|were at|landed at)\b", plain, re.I):
        return f"use present tense with {timeframe} (is at, hit)"
    return None


def mechanical_sandbox_check(
    *,
    ideal: dict[str, Any],
    actual: dict[str, Any],
    input_payload: dict[str, Any],
) -> dict[str, Any]:
    """Heuristic checks vs ideal output and prompt constraints."""
    issues: list[str] = []
    ideal_insight = str(ideal.get("insight", ""))
    actual_insight = str(actual.get("insight", ""))
    input_type = str(input_payload.get("type", ""))
    period = _expected_period(input_type)

    if ideal.get("insight_correct") != actual.get("insight_correct"):
        issues.append(
            f"insight_correct mismatch: ideal={ideal.get('insight_correct')} actual={actual.get('insight_correct')}"
        )
    if "\n" in actual_insight:
        issues.append("insight contains newline")
    if len(actual_insight) > 320:
        issues.append(f"insight over 320 chars ({len(actual_insight)})")
    word_count = _insight_word_count(actual_insight)
    if word_count > MAX_INSIGHT_WORDS:
        issues.append(f"insight over {MAX_INSIGHT_WORDS} words ({word_count}, excluding bracket markup)")
    if _RE_BANNED.search(actual_insight):
        issues.append("banned phrasing: currently")
    if _RE_DISCREPANCY_ACK.search(_strip_markup(actual_insight)):
        issues.append("insight acknowledges Insight↔Drivers discrepancy")
    if _RE_PARTIAL_TIMEFRAME.search(_strip_markup(actual_insight)):
        issues.append("use this week/month instead of partial-period phrasing")
    timeframe = _timeframe_phrase(str(input_payload.get("insight", "")))
    tense_problem = _tense_issue(_strip_markup(actual_insight), timeframe)
    if tense_problem:
        issues.append(tense_problem)
    if not _RE_LINK.search(actual_insight):
        issues.append("missing colored category link")
    if _RE_MALFORMED_LINK.search(actual_insight):
        issues.append("malformed link syntax")
    if not _RE_AMOUNT.search(actual_insight):
        issues.append("missing colored amount")

    actual_links = _RE_LINK.findall(actual_insight)
    ideal_links = _RE_LINK.findall(ideal_insight)
    actual_amounts = _RE_AMOUNT.findall(actual_insight)

    if len(actual_amounts) < len(actual_links):
        issues.append(
            f"colored amounts ({len(actual_amounts)}) fewer than category links ({len(actual_links)})"
        )
    for color, _display, path in actual_links:
        if not re.search(rf"/{period}$", path):
            issues.append(f"link period mismatch: {path} (expected /{period})")
        if not any(ac == color for ac, _ in actual_amounts):
            issues.append(f"link color {color} missing matching colored amount")
    for _color, _display, path in ideal_links:
        if not any(p == path for _c, _d, p in actual_links):
            issues.append(f"missing ideal link path: {path}")

    emoji_n = _emoji_count(actual_insight)
    min_emojis = _min_emojis_for_length(len(actual_insight))
    if emoji_n < 1:
        issues.append("missing emoji (minimum 1)")
    elif emoji_n < min_emojis:
        issues.append(f"too few emojis for length ({emoji_n}, expected ≥{min_emojis})")

    if "vs_goal" in input_type and not re.search(
        r"\b(budget|limit|over|exceeded|blew past)\b", _strip_markup(actual_insight), re.I
    ):
        issues.append("vs_goal missing budget/limit wording")

    ideal_plain = _strip_markup(ideal_insight).lower()
    actual_plain = _strip_markup(actual_insight).lower()
    ideal_tokens = {w for w in re.findall(r"[a-z]{4,}", ideal_plain)}
    overlap = len(ideal_tokens & {w for w in re.findall(r"[a-z]{4,}", actual_plain)})
    similarity = overlap / max(len(ideal_tokens), 1)

    good = not issues and similarity >= 0.35
    if similarity < 0.35 and not any("similarity" in i for i in issues):
        issues.append(f"low ideal similarity ({similarity:.0%})")

    return {
        "good_copy": good,
        "info_correct": ideal.get("insight_correct") == actual.get("insight_correct"),
        "similarity": round(similarity, 2),
        "issues": issues,
        "eval_text": "\n".join(f"- {x}" for x in issues) if issues else "OK",
    }


TEST_CASES: list[dict[str, Any]] = [
    # Batch 1
    {
        "name": "interest_ally_zero_last_month_on_track",
        "batch": 1,
        "input": {
            "type": "month_spent_vs_forecast",
            "insight": (
                "Interest was significantly down last month at $0.\n"
                "Category Taxonomy: parent Income (income), with leaf categories Salary (income_salary), "
                "Sidegig (income_sidegig), Business (income_business), Interest (income_interest)"
            ),
            "drivers": (
                "### Interest\n"
                "Although the insight flags Interest significantly down at $0, Interest is on track on the overall trail, "
                "as Ally Bank interest credits ($18–$24) post bimonthly on a steady cadence, not a drop in interest income."
            ),
        },
        "output": """{
  "insight": "r{[Interest](/income_interest/monthly)} was at r{$0} last month — Ally Bank interest credits were not received, as expected. 🏦",
  "insight_correct": false
}""",
    },
    {
        "name": "car_fuel_shell_chevron_up_this_month",
        "batch": 1,
        "input": {
            "type": "month_spend_vs_forecast",
            "insight": (
                "Car and Fuel is significantly up the first 9 days of this month at $168.\n"
                "(partial month Jul 1-31, 2026)\n"
                "Category Taxonomy: parent Transport (transportation), with leaf categories Public Transit "
                "(transportation_public), Car and Fuel (transportation_car)"
            ),
            "drivers": (
                "### Car and Fuel\n"
                "Car and Fuel is up on the trail—focal $168.40 (Jul 1–9) sits well above the usual $55–$90 band for "
                "early-month partials—with Shell ($94.12) and Chevron ($74.28 on Jul 7) driving the lift."
            ),
        },
        "output": """{
  "insight": "r{[Car and Fuel](/transportation_car/monthly)} is up at r{$168} this month — Shell and Chevron run the tab. ⛽",
  "insight_correct": true
}""",
    },
    {
        "name": "clothing_over_limit_nordstrom_zara",
        "batch": 1,
        "input": {
            "type": "month_spend_vs_goal",
            "insight": (
                "Clothing is over its monthly limit the first 12 days of this month at $241.\n"
                "(partial month Jul 1-31, 2026)\n"
                "Category Taxonomy: parent Shopping (shopping), with leaf categories Clothing (shopping_clothing), "
                "Gadgets (shopping_gadgets), Kids (shopping_kids), Pets (shopping_pets)"
            ),
            "drivers": (
                "### Clothing\n"
                "Clothing is above the user's monthly limit on the trail—focal $241.15 (Jul 1–12) exceeds the $200 "
                "cap—with Nordstrom ($132.40 across two visits) and Zara ($88.75) making up most of the overage."
            ),
        },
        "output": """{
  "insight": "r{[Clothing](/shopping_clothing/monthly)} is up at r{$241} this month because of trips to Nordstrom and Zara. 👕",
  "insight_correct": true
}""",
    },
    # Batch 2
    {
        "name": "entertainment_netflix_amc_week_usual",
        "batch": 2,
        "input": {
            "type": "week_spend_vs_forecast",
            "insight": (
                "Entertainment is significantly up the first 3 days of this week at $67.\n"
                " Leisure is thus significantly down the first 3 days of this week to $67.\n"
                "(partial week Jul 14-20, 2026)\n"
                "Category Taxonomy: parent Leisure (leisure), with leaf categories Entertainment (leisure_entertainment), "
                "Travel and Vacations (leisure_travel)"
            ),
            "drivers": (
                "**Entertainment is significantly up the first 3 days of this week at $67:** Entertainment is on track "
                "on the 6-week trail, bouncing between $0 and $85, with Netflix ($15.49) and AMC Theatres ($51.51) "
                "accounting for this week's spend—similar to a typical mid-July week last year.\n"
                "**Leisure is thus significantly down the first 3 days of this week to $67:** Although the insight flags "
                "Leisure as down, Leisure is on track on the 6-week trail after a Travel spike ($390.00) in late June; "
                "Entertainment ($67.00) is the only active leaf this week, and the total fits normal weekly swings."
            ),
        },
        "output": """{
  "insight": "g{[Entertainment](/leisure_entertainment/weekly)} is as expected at g{$67} this week — Netflix and AMC, usual mix for you. 🎬",
  "insight_correct": false
}""",
    },
    {
        "name": "salary_direct_deposit_up_this_month",
        "batch": 2,
        "input": {
            "type": "month_spent_vs_forecast",
            "insight": (
                "Salary is significantly up this month at $4,820.\n"
                "Category Taxonomy: parent Income (income), with leaf categories Salary (income_salary), "
                "Sidegig (income_sidegig), Business (income_business), Interest (income_interest)"
            ),
            "drivers": (
                "### Salary\n"
                "Salary is up on the trail—focal $4,820.00 sits above the usual $4,100–$4,200 band—"
                "with Acme Corp payroll ($4,820.00) including a one-time bonus driving the lift."
            ),
        },
        "output": """{
  "insight": "g{[Salary](/income_salary/monthly)} is at g{$4820} this month — Acme Corp payroll ran higher. 💰",
  "insight_correct": true
}""",
    },
    {
        "name": "dining_out_under_limit_week_chipotle",
        "batch": 2,
        "input": {
            "type": "week_spend_vs_goal",
            "insight": (
                "Dining Out is within its weekly limit this week at $42.\n"
                "(partial week Jul 14-20, 2026)\n"
                "Category Taxonomy: parent Meals (meals), with leaf categories Dining Out (meals_dining_out), "
                "Delivered Food (meals_delivered_food), Groceries (meals_groceries)"
            ),
            "drivers": (
                "### Dining Out\n"
                "Dining Out is under the user's weekly limit on the trail—focal $42.15 stays below the $75 "
                "cap—with Chipotle ($22.40) and a single Sweetgreen ($19.75) making up the spend."
            ),
        },
        "output": """{
  "insight": "g{[Dining Out](/meals_dining_out/weekly)} is at only g{$42} this week — under your usual at Chipotle and Sweetgreen. 🥗",
  "insight_correct": true
}""",
    },
    # Batch 3
    {
        "name": "public_transit_down_last_week_mta",
        "batch": 3,
        "input": {
            "type": "week_spend_vs_forecast",
            "insight": (
                "Public Transit is significantly down last week at $12.\n"
                "Category Taxonomy: parent Transport (transportation), with leaf categories Public Transit "
                "(transportation_public), Car and Fuel (transportation_car)"
            ),
            "drivers": (
                "### Public Transit\n"
                "Public Transit is down on the trail—focal $12.00 sits well below the usual $45–$60 band—"
                "with a single MTA MetroCard refill ($12.00) and no other rides last week."
            ),
        },
        "output": """{
  "insight": "g{[Public Transit](/transportation_public/weekly)} was down at g{$12} last week because of just one MTA refill. 🚇",
  "insight_correct": true
}""",
    },
    {
        "name": "utilities_pge_over_limit_this_month",
        "batch": 3,
        "input": {
            "type": "month_spend_vs_goal",
            "insight": (
                "Utilities is over its monthly limit the first 15 days of this month at $210.\n"
                "(partial month Jul 1-31, 2026)\n"
                "Category Taxonomy: parent Shelter (shelter), with leaf categories Home (shelter_home), "
                "Utilities (shelter_utilities), Upkeep (shelter_upkeep)"
            ),
            "drivers": (
                "### Utilities\n"
                "Utilities is above the user's monthly limit on the trail—focal $210.00 (Jul 1–15) exceeds the $160 "
                "cap—with PG&E ($168.40) and a Water Dept bill ($41.60) making up the overage."
            ),
        },
        "output": """{
  "insight": "r{[Utilities](/shelter_utilities/monthly)} is higher at r{$210} this month because of higher PG&E and Water Dept bills. 💡",
  "insight_correct": true
}""",
    },
    {
        "name": "business_stripe_zero_this_month_on_track",
        "batch": 3,
        "input": {
            "type": "month_spent_vs_forecast",
            "insight": (
                "Business is significantly down the first 10 days of this month at $0.\n"
                "(partial month Jul 1-31, 2026)\n"
                "Category Taxonomy: parent Income (income), with leaf categories Salary (income_salary), "
                "Sidegig (income_sidegig), Business (income_business), Interest (income_interest)"
            ),
            "drivers": (
                "### Business\n"
                "Although the insight flags Business significantly down at $0, Business is on track on the overall trail, "
                "as Stripe payouts ($1,200–$1,450) typically clear after the 15th, not a drop in business income."
            ),
        },
        "output": """{
  "insight": "r{[Business](/income_business/monthly)} is at r{$0} this month as Stripe payouts are expected to come later in the month. 🧾",
  "insight_correct": false
}""",
    },
    # Batch 4
    {
        "name": "medical_cvs_urgentcare_up_this_week",
        "batch": 4,
        "input": {
            "type": "week_spend_vs_forecast",
            "insight": (
                "Medical and Pharmacy is significantly up this week at $186.\n"
                "Category Taxonomy: parent Health (health), with leaf categories Medical and Pharmacy "
                "(health_medical_pharmacy), Gym and Wellness (health_gym_wellness), Personal Care (health_personal_care)"
            ),
            "drivers": (
                "### Medical and Pharmacy\n"
                "Medical and Pharmacy is up on the trail—focal $186.20 sits well above the usual $15–$40 band—"
                "with CVS Pharmacy ($48.20) and City Urgent Care ($138.00) driving the lift."
            ),
        },
        "output": """{
  "insight": "r{[Medical and Pharmacy](/health_medical_pharmacy/weekly)} is up at r{$186} this week since CVS and City Urgent Care payments ran higher. 💊",
  "insight_correct": true
}""",
    },
    {
        "name": "travel_airbnb_united_down_last_month",
        "batch": 4,
        "input": {
            "type": "month_spend_vs_forecast",
            "insight": (
                "Travel and Vacations is significantly down last month at $86.\n"
                "Category Taxonomy: parent Leisure (leisure), with leaf categories Entertainment (leisure_entertainment), "
                "Travel and Vacations (leisure_travel)"
            ),
            "drivers": (
                "### Travel and Vacations\n"
                "Travel and Vacations is down on the trail—focal $86.00 sits well below the usual $350–$500 band—"
                "with a lone United Airlines bag fee ($35.00) and Airbnb cleaning credit residual ($51.00)."
            ),
        },
        "output": """{
  "insight": "g{[Travel and Vacations](/leisure_travel/monthly)} was lower at g{$86} last month since transactions were just with United and Airbnb. ✈️",
  "insight_correct": true
}""",
    },
    {
        "name": "donations_gofundme_over_weekly_limit",
        "batch": 4,
        "input": {
            "type": "week_spend_vs_goal",
            "insight": (
                "Donations and Gifts is over its weekly limit this week at $120.\n"
                "Category Taxonomy: parent Donations and Gifts (donations_gifts)"
            ),
            "drivers": (
                "### Donations and Gifts\n"
                "Donations and Gifts is above the user's weekly limit on the trail—focal $120.00 exceeds the $50 "
                "cap—with GoFundMe ($75.00) and a Venmo gift to Maya ($45.00) making up the overage."
            ),
        },
        "output": """{
  "insight": "r{[Donations and Gifts](/donations_gifts/weekly)} is up at r{$120} this week because of donations through GoFundMe and Venmo. 🎁",
  "insight_correct": true
}""",
    },
]


def _run_test_with_logging(
    insight_input: Dict[str, Any] | str,
    optimizer: RationalizedPennyInsightsVerbalizerOptimizer | None = None,
    *,
    quiet: bool = False,
):
    if optimizer is None:
        optimizer = RationalizedPennyInsightsVerbalizerOptimizer()

    if not quiet:
        print("=" * 80)
    if isinstance(insight_input, str):
        input_text = insight_input.strip()
        if not input_text:
            raise ValueError("insight_input string is empty.")
    else:
        input_text = format_insight_input_for_llm(insight_input)
    if not quiet:
        print("LLM INPUT (markdown):")
        print("=" * 80)
        print(input_text)
        print("=" * 80 + "\n")

    result = optimizer.generate_response(input_text)

    if not quiet:
        print("=" * 80)
        print("LLM OUTPUT (rationalized verbalized):")
        print("=" * 80)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("=" * 80 + "\n")
    return result


def get_test_case(test_name_or_index):
    if isinstance(test_name_or_index, int):
        if 0 <= test_name_or_index < len(TEST_CASES):
            return TEST_CASES[test_name_or_index]
        return None
    if isinstance(test_name_or_index, str):
        for tc in TEST_CASES:
            if tc["name"] == test_name_or_index:
                return tc
        return None
    return None


def run_test(
    test_name_or_index_or_dict,
    optimizer: RationalizedPennyInsightsVerbalizerOptimizer | None = None,
    *,
    run_sandbox: bool = True,
    quiet: bool = False,
):
    if optimizer is None:
        optimizer = RationalizedPennyInsightsVerbalizerOptimizer()

    if isinstance(test_name_or_index_or_dict, dict):
        di = test_name_or_index_or_dict
        if "input" in di:
            payload = di["input"]
        elif "insight_input" in di:
            payload = di["insight_input"]
        else:
            print("Invalid test dict: must contain 'input' or 'insight_input' key.")
            return None
        name = di.get("name", "custom_test")
        if not quiet:
            print(f"\n{'='*80}\nRunning test: {name}\n{'='*80}\n")
        result = _run_test_with_logging(payload, optimizer, quiet=quiet)
        ideal_raw = di.get("output") or di.get("ideal_response")
        ideal = _parse_expected_output(ideal_raw) if ideal_raw else None
        if ideal and not quiet:
            print("\n" + "=" * 80 + "\nIDEAL RESPONSE:\n" + "=" * 80 + "\n" + ideal_raw + "\n" + "=" * 80 + "\n")
        if run_sandbox and result and ideal and isinstance(payload, dict):
            sand = mechanical_sandbox_check(ideal=ideal, actual=result, input_payload=payload)
            if not quiet:
                print("=" * 80)
                print("SANDBOX EXECUTION:")
                print("=" * 80)
                print(sand["eval_text"])
                print(
                    f"{'✅' if sand['good_copy'] else '❌'} good_copy={sand['good_copy']} "
                    f"info_correct={sand['info_correct']} similarity={sand['similarity']}"
                )
                print("=" * 80 + "\n")
            return {"result": result, "sandbox": sand, "ideal": ideal, "name": name}
        return result

    tc = get_test_case(test_name_or_index_or_dict)
    if tc is None:
        print(f"Test case '{test_name_or_index_or_dict}' not found.")
        return None
    return run_test(tc, optimizer, run_sandbox=run_sandbox, quiet=quiet)


def _parse_expected_output(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def run_batch(
    batch_num: int,
    optimizer: RationalizedPennyInsightsVerbalizerOptimizer | None = None,
    *,
    quiet: bool = True,
) -> list[dict[str, Any]]:
    cases = [tc for tc in TEST_CASES if int(tc.get("batch") or 0) == int(batch_num)]
    if not cases:
        raise ValueError(f"No tests found for batch={batch_num}")
    summaries: list[dict[str, Any]] = []
    if optimizer is not None:
        optimizer.quiet = quiet
    for tc in cases:
        out = run_test(tc, optimizer, run_sandbox=True, quiet=quiet)
        if isinstance(out, dict) and "sandbox" in out:
            summaries.append(
                {
                    "name": tc["name"],
                    "good_copy": out["sandbox"]["good_copy"],
                    "info_correct": out["sandbox"]["info_correct"],
                    "similarity": out["sandbox"]["similarity"],
                    "issues": out["sandbox"]["issues"],
                    "actual": out["result"],
                    "ideal": out["ideal"],
                }
            )
    return summaries


def main(test: str | None = None, *, batch: int | None = None, no_thinking: bool = False):
    thinking_budget = 2048
    optimizer = RationalizedPennyInsightsVerbalizerOptimizer(thinking_budget=thinking_budget)
    optimizer.quiet = False

    if batch is not None:
        summaries = run_batch(batch, optimizer, quiet=False)
        passed = sum(1 for s in summaries if s["good_copy"])
        print(f"\nBatch {batch}: {passed}/{len(summaries)} passed sandbox")
        return

    if test is not None:
        if test.strip().lower() == "all":
            print(f"\n{'='*80}\nRunning ALL test cases\n{'='*80}\n")
            for i in range(len(TEST_CASES)):
                run_test(i, optimizer)
                if i < len(TEST_CASES) - 1:
                    print("\n" + "-" * 80 + "\n")
            return
        test_val = int(test) if test.isdigit() else test
        run_test(test_val, optimizer)
        return

    _print_usage()


def _print_usage():
    print("Usage:")
    print("  Run a single test: --test <name_or_index>")
    print("  Run all tests: --test all")
    print("  Run batch: --batch <1-4>")
    print("  Disable thinking: --no-thinking (thinking_budget=0)")
    print("\nAvailable test cases:")
    for i, tc in enumerate(TEST_CASES):
        print(f"  {i}: {tc['name']} (batch {tc.get('batch', '?')})")
    print("  all: run all test cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run P:RationalizedPennyInsightsVerbalizer optimizer tests")
    parser.add_argument("--test", type=str, help='Test name or index (e.g. "spend_vs_forecast_leisure_down" or "0")')
    parser.add_argument("--batch", type=int, help="Run all tests in batch N (1-4)")
    parser.add_argument("--no-thinking", action="store_true", help="Set thinking_budget=0 (Thinking OFF)")
    args = parser.parse_args()
    main(test=args.test, batch=args.batch, no_thinking=args.no_thinking)

