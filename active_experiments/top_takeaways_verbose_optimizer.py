"""
Optimizer runner for multi-context **top takeaways** inputs (plain markdown user message).

The **user message** is built by ``build_top_takeaways_user_message_from_contexts`` (or the shorthand
``build_top_takeaways_user_message`` for all-rationalize runs). Each **context** has a required ``tap_link`` and
**exactly one** of: ``rationalize_agent_outcome`` (full rationalize markdown) or ``insight`` (plain text for
non-rationalized transaction alerts such as ``uncat_txn`` / ``large_txn``). Rationalize headings are renamed to
``# {Category} Insight`` / ``# {Category} Rationalization``; drill-down URLs appear under
``## Helpful Links to Information`` as Markdown bullets.

The **system instruction** is the canonical top-takeaways prompt (``SYSTEM_PROMPT``) for **P:TopTakeawaysVerbose**.
This one-shot Gemini path has **no** finance tools; the model should rely on the pasted rationalize markdown only.

Run from ``finance-ai-penny`` repo root:

  python3 active_experiments/top_takeaways_verbose_optimizer.py --test 0
  python3 active_experiments/top_takeaways_verbose_optimizer.py --test all

Thinking stays **on** (``thinking_budget`` 256, aligned with the production top-takeaways LLM template).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from typing import Any

_PENNY_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PENNY_REPO_ROOT not in sys.path:
  sys.path.insert(0, _PENNY_REPO_ROOT)

from categories import get_name

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

try:
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]

if load_dotenv is not None:
    load_dotenv()

GEMINI_FLASH_LITE = "gemini-flash-lite-latest"

# Canonical prompt for **P:TopTakeawaysVerbose** (rollup instructions for multi-context top takeaways).
SYSTEM_PROMPT = """You are **Penny**.

Context blocks (markdown, blank-line separated): rationalize body (`Explain:`, `## Figures`, `## Drivers`) or plain insight, plus `## Helpful Links to Information` (`- [name](/cashflow/...)`).

Synthesize **# Top Takeaways** with **## Highlights** (stability, savings, flat costs) and **## Lowlights** (spikes, overspending, miscategorization). Omit **## Lowlights** when none apply.

Each `- ` bullet: **Short headline:** + 2–4 sentences (what, amount, comparison, driver) from Figures/Drivers. The headline names the **story** (e.g. **Income Growth:**, **Utilities Spike:**, **Miscategorization:**)—not the category alone (**Salary:**, **Food:**). Put `[Category](/cashflow/...)` links in the sentence. When Explain disagrees with Figures or Drivers, use Figures/Drivers only—state the trend as a plain fact (e.g. "total spend reached $974" not "despite the insight flagging … as down"). **Never** use the word **insight**, **flagging**, **alert**, or meta phrases about upstream text, Explain, or disagreement. ≤3 bullets per section; one primary context per bullet.

**Amounts:** Whole dollars, comma thousands (`$6,369`).

**Periods:** **this month/week**; **last month/week** or **at this point last month/week**; prior week → **from April 26 to May 2**; older → **in March**. No years in prose; never "first X days", "so far", or "the previous week".

**Links:** `[display name](/cashflow/path)` woven into the sentence. Path from Helpful Links; anchor matches display name. Link each category discussed (parent + leaf when both matter, e.g. Food + Groceries, Shelter + Upkeep).

**Voice:** Observational Drivers tone—declarative facts, plain merchant names, no advice.

**Output (strict):** Your reply is **only** the markdown below—first line `# Top Takeaways`, then Highlights bullets, then Lowlights bullets when applicable. No preamble, postscript, YAML/JSON, properties, repeated input blocks, `## Helpful Links`, or any other headings or sections.

# Top Takeaways

## Highlights

- **Income Stability:** …

## Lowlights

- **Utilities Spike:** …
"""



def _category_id_from_cashflow_path(path: str) -> int | None:
    """Leading integer category id in ``/cashflow/<id>/…``, else ``None``."""
    p = (path or "").strip()
    if not p.startswith("/cashflow/"):
        return None
    rest = p[len("/cashflow/") :].lstrip("/")
    if not rest or rest.lower().startswith("transaction/"):
        return None
    m = re.match(r"^-?\d+", rest)
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def _tap_link_name_for_category_id(cat_id: int) -> str | None:
    """Display name for helpful links when derived from a Penny category id (id ``1`` is shown as **Food**)."""
    if cat_id == 1:
        return "Food"
    nm = get_name(cat_id)
    if isinstance(nm, str) and nm.strip():
        return nm.strip()
    return None


def _tap_link_category_display_name(*, tap_path: str, ctx: dict[str, Any]) -> str:
    raw = ctx.get("tap_link_category_name")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    meta = ctx.get("metadata")
    if isinstance(meta, dict):
        for k in ("tap_link_category_name", "category_display_name", "category_name"):
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        cid_raw = meta.get("category_id")
        if cid_raw is not None:
            try:
                cid = int(cid_raw)
            except (TypeError, ValueError):
                cid = None
            if cid is not None:
                nm = _tap_link_name_for_category_id(cid)
                if nm:
                    return nm
    cid = _category_id_from_cashflow_path(tap_path)
    if cid is not None:
        nm = _tap_link_name_for_category_id(cid)
        if nm:
            return nm
    return ""


_RE_RATIONALIZE_WHAT = re.compile(r"^#\s*Rationalize\s+What\s*$", re.IGNORECASE | re.MULTILINE)
_RE_RATIONALIZE_RESPONSE = re.compile(r"^#\s*Rationalize\s+Response\s*$", re.IGNORECASE | re.MULTILINE)
_RE_CATEGORY_TAXONOMY_LINE = re.compile(r"^Category Taxonomy:.*$", re.IGNORECASE | re.MULTILINE)
_RE_EXPLAIN_CATEGORY = re.compile(r"^Explain:\s*([A-Za-z][A-Za-z0-9 _/&-]*)", re.IGNORECASE | re.MULTILINE)
_RE_CASHFLOW_CATEGORY_PATH = re.compile(r"^(/cashflow/)(-?\d+)((?:/[^)\s]+)*)$")

# Same taxonomy as Hermes ``format_category_taxonomy_line`` (``rationalize_task.py``).
_TOP_LEVEL_CATEGORY_IDS = frozenset({41, 42, 43, 44, 46})
_PARENT_CATEGORY_IDS = frozenset({-1, 1, 5, 9, 14, 18, 21, 25, 28, 32, 45, 47})
_TOP_LEVEL_TO_DESCENDANTS: dict[int, tuple[int, ...]] = {
    41: (1, 2, 3, 4),
    42: (5, 6, 7, 18, 19, 25, 27, 28, 29, 30, 31, 32, 33, -1),
    43: (9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 26),
    44: (21, 22, 23, 24, 8),
    46: (47, 36, 37, 38, 39),
}
_PARENT_TO_LEAF_IDS: dict[int, tuple[int, ...]] = {
    -1: (-1,),
    1: (1, 2, 3, 4),
    5: (5, 6, 7),
    9: (9, 10, 11, 12, 13),
    14: (14, 15, 16, 17),
    18: (18, 19, 20),
    21: (21, 22, 23, 24, 8),
    25: (25, 26, 27),
    28: (28, 29, 30, 31),
    32: (32,),
    33: (33,),
    45: (45,),
    47: (36, 37, 38, 39),
}
_RE_HELPFUL_LINKS_SECTION = re.compile(
    r"##\s+Helpful\s+Links\s+to\s+Information\s*\n(.*?)(?=\n##\s+|\n#\s+|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_RE_MD_LINK_PATH = re.compile(r"\]\(\s*(/cashflow/[^)\s]+)\s*\)")


def _parse_category_from_explain(md: str) -> str | None:
    m = _RE_EXPLAIN_CATEGORY.search(md or "")
    if not m:
        return None
    label = (m.group(1) or "").strip()
    return label.split()[0].strip() if label else None


def _context_category_label(*, ctx: dict[str, Any], rationalize_md: str | None = None) -> str:
    nm = _tap_link_category_display_name(tap_path=str(ctx.get("tap_link") or ""), ctx=ctx)
    if nm:
        return nm
    if isinstance(rationalize_md, str) and rationalize_md.strip():
        from_explain = _parse_category_from_explain(rationalize_md)
        if from_explain:
            return from_explain
    return "Category"


def _strip_category_taxonomy(md: str) -> str:
    """Remove Hermes ``Category Taxonomy: …`` lines from rationalize markdown."""
    if not isinstance(md, str) or not md.strip():
        return md if isinstance(md, str) else ""
    s = _RE_CATEGORY_TAXONOMY_LINE.sub("", md)
    return re.sub(r"\n{3,}", "\n\n", s).strip("\n")


def _rename_rationalize_headings(md: str, category: str) -> str:
    cat = (category or "Category").strip()
    s = _RE_RATIONALIZE_WHAT.sub(f"# {cat} Insight\n\n", md)
    s = _RE_RATIONALIZE_RESPONSE.sub(f"# {cat} Rationalization\n\n", s)
    insight_h = re.compile(rf"^#\s*{re.escape(cat)}\s+Insight\s*$", re.IGNORECASE | re.MULTILINE)
    rat_h = re.compile(rf"^#\s*{re.escape(cat)}\s+Rationalization\s*$", re.IGNORECASE | re.MULTILINE)
    s = insight_h.sub(f"# {cat} Insight\n\n", s)
    s = rat_h.sub(f"# {cat} Rationalization\n\n", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip("\n")


def _prepare_rationalize_markdown_for_top_takeaways(md: str, category: str) -> str:
    """Strip taxonomy, rename headings, and ensure a blank line after each insight heading."""
    return _rename_rationalize_headings(_strip_category_taxonomy(md), category)


def _leaf_ids_for_top_level_rollups(top_id: int) -> list[int]:
    descendants = set(_TOP_LEVEL_TO_DESCENDANTS.get(top_id, ()))
    intermediate_parents = {p for p in _PARENT_CATEGORY_IDS if p in descendants}
    return sorted(descendants - intermediate_parents - {top_id})


def _leaf_ids_for_parent_category(parent_id: int) -> list[int]:
    leaves = _PARENT_TO_LEAF_IDS.get(parent_id, (parent_id,))
    return sorted(lid for lid in leaves if lid != parent_id)


def _taxonomy_leaf_category_ids(root_category_id: int) -> list[int]:
    try:
        cid = int(root_category_id)
    except (TypeError, ValueError):
        return []
    if cid in _TOP_LEVEL_CATEGORY_IDS:
        return _leaf_ids_for_top_level_rollups(cid)
    if cid in _PARENT_CATEGORY_IDS:
        return _leaf_ids_for_parent_category(cid)
    return []


def _root_category_id_from_ctx(ctx: dict[str, Any]) -> int | None:
    meta = ctx.get("metadata")
    if isinstance(meta, dict):
        raw = meta.get("category_id")
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                pass
    tl = ctx.get("tap_link")
    if isinstance(tl, str):
        return _category_id_from_cashflow_path(tl)
    return None


def _cashflow_path_for_category_id(base_path: str, category_id: int) -> str | None:
    m = _RE_CASHFLOW_CATEGORY_PATH.match((base_path or "").strip())
    if not m:
        return None
    return f"{m.group(1)}{int(category_id)}{m.group(3)}"


def _display_name_for_category_id(cat_id: int) -> str | None:
    nm = _tap_link_name_for_category_id(cat_id)
    if nm:
        return nm
    got = get_name(cat_id)
    return got.strip() if isinstance(got, str) and got.strip() else None


def _helpful_link_entries(ctx: dict[str, Any]) -> list[tuple[str, str]]:
    raw_links = ctx.get("helpful_links")
    if isinstance(raw_links, list) and raw_links:
        out: list[tuple[str, str]] = []
        for item in raw_links:
            if isinstance(item, dict):
                label = item.get("label") or item.get("name") or item.get("title")
                path = item.get("path") or item.get("tap_link") or item.get("href")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                label, path = item[0], item[1]
            else:
                continue
            if isinstance(label, str) and isinstance(path, str) and label.strip() and path.strip():
                out.append((label.strip(), path.strip()))
        if out:
            return out
    tl_raw = ctx.get("tap_link")
    if not isinstance(tl_raw, str) or not tl_raw.strip():
        raise ValueError("tap_link must be a non-empty string when helpful_links is omitted")
    base_path = tl_raw.strip()
    entries: list[tuple[str, str]] = []
    seen_paths: set[str] = set()

    def append(label: str, path: str) -> None:
        p = path.strip()
        if not p or p in seen_paths:
            return
        seen_paths.add(p)
        entries.append((label.strip(), p))

    parent_label = _tap_link_category_display_name(tap_path=base_path, ctx=ctx) or "Detail"
    append(parent_label, base_path)

    root_cid = _root_category_id_from_ctx(ctx)
    if root_cid is not None:
        for lid in _taxonomy_leaf_category_ids(root_cid):
            leaf_path = _cashflow_path_for_category_id(base_path, lid)
            if not leaf_path:
                continue
            leaf_label = _display_name_for_category_id(lid)
            if leaf_label:
                append(leaf_label, leaf_path)

    if not entries:
        raise ValueError("helpful_links could not be built from tap_link and category taxonomy")
    return entries


def _format_helpful_links_section(ctx: dict[str, Any]) -> str:
    lines = ["## Helpful Links to Information", ""]
    for label, path in _helpful_link_entries(ctx):
        lines.append(f"- [{label}]({path})")
    return "\n".join(lines)


def extract_helpful_link_paths_from_user_message(user_message: str) -> list[str]:
    """Paths from each ``## Helpful Links to Information`` section (canonical drill-down URLs)."""
    if not isinstance(user_message, str) or not user_message.strip():
        return []
    out: list[str] = []
    for m in _RE_HELPFUL_LINKS_SECTION.finditer(user_message):
        section = m.group(1) or ""
        for lm in _RE_MD_LINK_PATH.finditer(section):
            path = (lm.group(1) or "").strip()
            if path:
                out.append(path)
    return out


# Production top-takeaways template ``thinking_budget`` — do not set to 0.
TOP_TAKEAWAYS_THINKING_BUDGET = 256
# Generous cap so multi-context rollups (plus thinking-enabled models) are not truncated mid-markdown.
TOP_TAKEAWAYS_MAX_OUTPUT_TOKENS = 8192


def build_top_takeaways_user_message_from_contexts(*, contexts: list[dict[str, Any]]) -> str:
    """Build the top-takeaways user task from a list of per-context dicts (plain markdown).

    Each dict **must** include:

    - ``tap_link`` (str): relative drill-down URL (``/cashflow/...``), unless ``helpful_links`` is set.

    Optional:

    - ``helpful_links`` (list): ``[{label, path}, …]`` or ``[(label, path), …]`` for ``## Helpful Links to Information``.
    - ``tap_link_category_name`` (str): category label for headings and link text (defaults from ``metadata`` or path id).
    - ``metadata`` (dict): may include ``category_id``, ``category_name``, or ``tap_link_category_name``.

    And **exactly one** of:

    - ``rationalize_agent_outcome`` (str): full rationalize markdown; ``# Rationalize What`` / ``# Rationalize Response`` are renamed.
    - ``insight`` (str): non-rationalized insight text only (e.g. ``uncat_txn`` / ``large_txn``).
    """
    if not isinstance(contexts, list) or not contexts:
        raise ValueError("contexts must be a non-empty list")
    blocks: list[str] = []
    for i, ctx in enumerate(contexts, start=1):
        if not isinstance(ctx, dict):
            raise TypeError(f"contexts[{i - 1}] must be a dict")
        ra = ctx.get("rationalize_agent_outcome")
        ins = ctx.get("insight")
        has_ra = isinstance(ra, str) and bool(ra.strip())
        has_ins = isinstance(ins, str) and bool(ins.strip())
        if has_ra and has_ins:
            raise ValueError(f"contexts[{i - 1}]: set only one of rationalize_agent_outcome or insight, not both")
        if not has_ra and not has_ins:
            raise ValueError(f"contexts[{i - 1}]: set exactly one of rationalize_agent_outcome or insight")
        parts: list[str] = []
        if has_ra:
            raw = ra.strip().rstrip("\n")  # type: ignore[union-attr]
            cat = _context_category_label(ctx=ctx, rationalize_md=raw)
            parts.append(_prepare_rationalize_markdown_for_top_takeaways(raw, cat))
        else:
            parts.append(ins.strip())  # type: ignore[union-attr]
        parts.append(_format_helpful_links_section(ctx))
        blocks.append("\n\n".join(parts))
    return "\n\n".join(blocks) + "\n"


def build_top_takeaways_user_message(
    *,
    rationalize_agent_outcomes: list[str],
    tap_links: list[str],
) -> str:
    """Shorthand when every context is a full rationalize outcome (same as zip → ``build_top_takeaways_user_message_from_contexts``)."""
    if not isinstance(rationalize_agent_outcomes, list) or not rationalize_agent_outcomes:
        raise ValueError("rationalize_agent_outcomes must be a non-empty list")
    if not isinstance(tap_links, list) or len(tap_links) != len(rationalize_agent_outcomes):
        raise ValueError("tap_links must be a list with the same length as rationalize_agent_outcomes.")
    contexts: list[dict[str, Any]] = [
        {"rationalize_agent_outcome": ra, "tap_link": tl} for ra, tl in zip(rationalize_agent_outcomes, tap_links)
    ]
    return build_top_takeaways_user_message_from_contexts(contexts=contexts)



def _normalize_markdown_relative_links(text: str) -> str:
    """Remove stray whitespace after ``](`` before a relative path (model sometimes emits ``]( /foo``)."""
    return re.sub(r"\]\(\s+/", "](/", text)


def _extract_stream_chunk_text(chunk: Any) -> str:
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


class TopTakeawaysVerboseOptimizer:
    """One-shot Gemini runner: canonical top-takeaways system prompt plus ``<CONTEXTS>`` user message."""

    def __init__(
        self,
        model_name: str = GEMINI_FLASH_LITE,
        *,
        thinking_budget: int = TOP_TAKEAWAYS_THINKING_BUDGET,
        max_output_tokens: int = TOP_TAKEAWAYS_MAX_OUTPUT_TOKENS,
    ):
        if genai is None or types is None:  # pragma: no cover
            raise RuntimeError("Install `google-genai` (and optionally `python-dotenv`) for this optimizer.")
        if not isinstance(thinking_budget, int) or thinking_budget < 1:
            raise ValueError("thinking_budget must be a positive int (thinking must stay enabled).")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.thinking_budget = thinking_budget
        self.max_output_tokens = max_output_tokens
        self.temperature = 0.2
        self.top_p = 0.95
        self.top_k = 40
        self.safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ]
        self.system_prompt = SYSTEM_PROMPT

    def generate_markdown(self, user_message: str) -> str:
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("user_message must be a non-empty <CONTEXTS> bundle string.")
        request_text = types.Part.from_text(text=user_message.strip())
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
        )
        out: list[str] = []
        thought_summary = ""
        for chunk in self.client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=cfg,
        ):
            piece = _extract_stream_chunk_text(chunk)
            if piece:
                out.append(piece)
            if hasattr(chunk, "candidates") and chunk.candidates:
                for candidate in chunk.candidates:
                    if hasattr(candidate, "content") and candidate.content:
                        if hasattr(candidate.content, "parts") and candidate.content.parts:
                            for part in candidate.content.parts:
                                if getattr(part, "thought", False) and getattr(part, "text", None):
                                    t = part.text
                                    thought_summary = (thought_summary + t) if thought_summary else t
        if thought_summary:
            print("\n" + "-" * 80 + "\nTHOUGHT SUMMARY:\n" + "-" * 80 + "\n" + thought_summary.strip() + "\n" + "-" * 80 + "\n")
        text = _normalize_markdown_relative_links("".join(out).strip())
        if not text:
            raise ValueError("Empty model response.")
        return text


TEST_CASES: list[dict[str, Any]] = [
    {
        "name": "four_contexts_all_rationalizes_with_tap_links",
        "input": """
# Salary Insight

Explain: Salary is significantly down this month at $4200. (2026-05-01 to 2026-05-10)

# Salary Rationalization

## Figures

*   **May 1–10, 2026:** $6,369.24
*   **April 1–10, 2026:** $6,369.24
*   **March 1–10, 2026:** $17,369.24

## Drivers

The Explain line describes salary as significantly down at $4,200 this month, but Figures show $6,369 for May 1–10—the same as the $6,369 from the same window in April—so pay is flat month-over-month, not down. The "significant drop" compared to early March is due to a large, one-time bonus payout received in the first half of March 2026. Specifically, you received $8,800 in total "Genentech US Bonus" payments on March 11, alongside higher "CA State Payroll" amounts ($8,600 total) compared to your standard bi-weekly payroll cycle. Your income for the first 10 days of May is consistent with your regular pay cycle, mirroring the amount received during the same period in April.

## Helpful Links to Information

- [Salary](/cashflow/36/monthly/2026-05)

# Food Insight

Explain: Groceries is significantly down this month at $937.  Food is thus significantly down this month to $1859. (2026-05-01 to 2026-05-10)

# Food Rationalization

## Figures

*   **Total Meals (May 1–10, 2026):** $1,859.34
*   **Total Meals (Apr 1–30, 2026):** $4,039.28
*   **Total Meals (Mar 1–31, 2026):** $3,300.28
*   **Groceries (May 1–10, 2026):** $937.22
*   **Groceries (Apr 1–30, 2026):** $1,513.40
*   **Groceries (Mar 1–31, 2026):** $1,562.82

## Drivers

The significant decrease in food and grocery spending is primarily due to comparing only 10 days of May activity against full-month totals for April and March.

Additionally, your transaction history shows some potential miscategorization: several transactions at "Applebee's" (a dining venue) are currently categorized under `meals_groceries`. For example, $85.36 was recorded at Applebee's on May 2nd under groceries, whereas a typical grocery bill for a similar amount is expected from retailers like "Target" or "Trader Joe's".

## Helpful Links to Information

- [Food](/cashflow/1/monthly/2026-05)
- [Dining Out](/cashflow/2/monthly/2026-05)
- [Delivered Food](/cashflow/3/monthly/2026-05)
- [Groceries](/cashflow/4/monthly/2026-05)

# Uncategorized Insight

Explain: Uncategorized is significantly up last week at $6382. (2026-05-03 to 2026-05-09)

# Uncategorized Rationalization

## Figures

*   **Uncategorized:** $6,382.12 (May 3–9, 2026)
*   **Uncategorized:** $725.52 (Apr 26–May 2, 2026)
*   **Uncategorized:** $500.46 (Apr 19–25, 2026)

## Drivers

The spike in uncategorized spending is primarily driven by two large payments to "Property Group LLC" totaling $5,700 ($2,850 each). Additionally, duplicate entries appear for several recurring transactions, such as "Community Pool" ($83.06 x 2), "Costco" ($69.71 x 2), and "BP" ($49.56 x 2), which suggests some transactions might be showing up twice across different accounts.

## Helpful Links to Information

- [Uncategorized](/cashflow/-1/weekly/2026-05-03)

# Uncategorized Insight

Explain: Uncategorized is slightly down this month at $6644. (2026-05-01 to 2026-05-10)

# Uncategorized Rationalization

## Figures

* **May 1–10, 2026**: $6,644.34 (Uncategorized)
* **Apr 1–30, 2026**: $7,681.54 (Uncategorized)
* **Mar 1–31, 2026**: $8,111.90 (Uncategorized)

## Drivers

The "Uncategorized" spend is trending downward compared to prior months. A significant portion of this category consists of recurring transactions like "Property Group LLC," which appears twice monthly at $2,850.00 each time ($5,700 total). The lower total for May so far is primarily due to a lower volume of smaller miscellaneous transactions (18 transactions in May vs. 28 in April) rather than a change in the major recurring charges.

## Helpful Links to Information

- [Uncategorized](/cashflow/-1/monthly/2026-05)
""",
        "output": """
# Top Takeaways

## Highlights

- **Income Stability:** This month [Salary](/cashflow/36/monthly/2026-05) tracks at $6,369, which is consistent with the amount received at this point last month. While income appears significantly lower than in Mar, that variance is driven by a one-time $8,800 bonus and higher payroll amounts received during the first half of Mar, rather than a change in the standard bi-weekly pay cycle.
- **Transaction Volume:** The total [Uncategorized](/cashflow/-1/monthly/2026-05) spend is trending downward this month at $6,644, compared to $7,682 for the entirety of last month. This improvement is driven by a lower volume of smaller miscellaneous transactions, with 18 recorded so far in May compared to 28 in Apr, even though major recurring charges remain constant.

## Lowlights

- **Uncategorized Spikes:** Last week, [Uncategorized](/cashflow/-1/weekly/2026-05-03) spending surged to $6,382, a sharp increase from the $726 recorded from Apr 26 to May 2. This spike reflects two large payments to Property Group LLC totaling $5,700, alongside duplicate entries for recurring transactions like Community Pool, Costco, and BP, which suggests some items may be appearing twice across accounts.
- **Miscategorization:** While [Food](/cashflow/1/monthly/2026-05) spending is $1,859 and [Groceries](/cashflow/4/monthly/2026-05) is $937 this month, transaction history reveals potential miscategorization. For instance, an $85 charge at Applebee's was recorded under groceries on May 2, despite being a dining venue, which contrasts with typical grocery bills from retailers like Trader Joe's.
""",
    },
    {
        "name": "single_context_delivered_food_long_figures",
        "input": """
# Delivered Food Insight

Explain: Delivered Food is significantly down this month at $0. (2026-05-01 to 2026-05-23)

# Delivered Food Rationalization

## Figures

*   **Delivered Food (meals_delivered_food)**:
    *   May 2026: from May 1–23, $0 · entire month $0
    *   Apr 2026: from Apr 1–23, $0 · entire month $0
    *   Mar 2026: from Mar 1–23, $0 · entire month $0
    *   Feb 2026: from Feb 1–23, $0 · entire month $0
    *   Jan 2026: from Jan 1–23, $9.99 · entire month $9.99
    *   Dec 2025: from Dec 1–23, $149.88 · entire month $149.88
    *   Nov 2025: from Nov 1–23, $29.36 · entire month $29.36
    *   Oct 2025: from Oct 1–23, $0 · entire month $0
    *   Sep 2025: from Sep 1–23, $54.18 · entire month $54.18

## Drivers

Spending on **Delivered Food** is $0 this month, consistent with at this point last month and the same $0 pattern in Mar and Feb.

## Helpful Links to Information

- [Delivered Food](/cashflow/3/monthly/2026-05)
""",
        "output": """
# Top Takeaways
## Highlights
- **Spending Control:** You have maintained a clean streak in [Delivered Food](/cashflow/3/monthly/2026-05) this month, with total spending at $0.
This consistent behavior mirrors the $0 activity recorded at this point last month, as well as the same $0 patterns observed throughout Mar and Feb.
""",
    },
    {
        "name": "six_contexts_income_and_bills_mosaic",
        "input": """
# Salary Insight

Explain: Salary is slightly up this month at $4210. (2026-05-01 to 2026-05-10)

# Salary Rationalization

## Figures

*   **May 1–10, 2026:** $4,210.00
*   **April 1–10, 2026:** $4,105.50

## Drivers

Payroll landed on the usual bi-weekly schedule with no bonus lines this month.

## Helpful Links to Information

- [Salary](/cashflow/36/monthly/2026-05)

# Side-Gig Insight

Explain: Side-Gig is significantly up this month at $1850. (2026-05-01 to 2026-05-10)

# Side-Gig Rationalization

## Figures

*   **Side-Gig (May 1–10, 2026):** $1,850.00
*   **Side-Gig (Apr 1–10, 2026):** $420.00

## Drivers

A $1,200 consulting deposit from "Brightline Studio" posted May 6, plus two smaller Upwork payouts ($350 and $300) that did not appear at this point last month.

## Helpful Links to Information

- [Side-Gig](/cashflow/37/monthly/2026-05)

# Connectivity Insight

Explain: Connectivity is flat this month at $189. (2026-05-01 to 2026-05-10)

# Connectivity Rationalization

## Figures

*   **Connectivity (May 1–10, 2026):** $189.00
*   **Connectivity (Apr 1–30, 2026):** $189.00

## Drivers

Recurring Comcast and mobile plans unchanged.

## Helpful Links to Information

- [Connectivity](/cashflow/10/monthly/2026-05)

# Insurance Insight

Explain: Insurance is slightly up this month at $312. (2026-05-01 to 2026-05-10)

# Insurance Rationalization

## Figures

*   **Insurance (May 1–10, 2026):** $312.40
*   **Insurance (Apr 1–30, 2026):** $298.00

## Drivers

Auto premium installment rose $14.40 after policy renewal effective May 1; homeowners line unchanged.

## Helpful Links to Information

- [Insurance](/cashflow/11/monthly/2026-05)

# Utilities Insight

Explain: Utilities is significantly up this month at $428. (2026-05-01 to 2026-05-10)

# Utilities Rationalization

## Figures

*   **Utilities (May 1–10, 2026):** $428.15
*   **Utilities (Apr 1–30, 2026):** $241.80
*   **Utilities (Mar 1–31, 2026):** $265.10

## Drivers

PG&E bill jumped to $286.40 on May 4 versus $118.20 at this point last month, reflecting higher cooling use and a rate adjustment. Water district charge ($92.75) also posted earlier in the cycle than in Apr, when it landed on the 18th. Together those two lines explain most of the month-over-month lift; garbage service stayed flat at $49.00.

## Helpful Links to Information

- [Utilities](/cashflow/16/monthly/2026-05)

# Interest Insight

Explain: Interest is slightly up this month at $18. (2026-05-01 to 2026-05-10)

# Interest Rationalization

## Figures

*   **Interest (May 1–10, 2026):** $18.22
*   **Interest (Apr 1–10, 2026):** $12.05

## Drivers

Savings yield credit posted May 2.

## Helpful Links to Information

- [Interest](/cashflow/39/monthly/2026-05)
""",
        "output": """
# Top Takeaways

## Highlights

- **Income Growth:** Total [Salary](/cashflow/36/monthly/2026-05) is slightly up this month at $4,210 compared to $4,106 at this point last month, driven by standard bi-weekly payroll.
Additionally, [Interest](/cashflow/39/monthly/2026-05) income rose to $18 from $12 at this point last month, following a savings yield credit posted on May 2.
- **Side-Gig Expansion:** [Side-Gig](/cashflow/37/monthly/2026-05) earnings saw a significant increase this month, reaching $1,850 compared to $420 at this point last month. This lift is driven by a $1,200 consulting deposit from Brightline Studio on May 6, alongside two Upwork payouts totaling $650 that did not occur in the prior period.
- **Stable Expenses:** [Connectivity](/cashflow/10/monthly/2026-05) costs remain flat this month at $189, consistent with the total from last month, as recurring Comcast and mobile plan charges were unchanged.

## Lowlights

- **Utilities Spike:** [Utilities](/cashflow/16/monthly/2026-05) expenses are significantly higher this month at $428, compared to $242 last month. The increase is driven by a PG&E bill of $286—up from $118 at this point last month due to higher cooling use and a rate adjustment—and a water district charge of $93 that posted earlier in the cycle than it did in April.
- **Insurance Adjustment:** [Insurance](/cashflow/11/monthly/2026-05) costs are slightly up this month at $312, compared to $298 last month. This change reflects a $14 increase in the auto premium installment following a policy renewal effective May 1, while the homeowners line remained unchanged.
""",
    },
    {
        "name": "two_contexts_shelter_and_entertainment_deep_drivers",
        "input": """
# Shelter Insight

Explain: Utilities is significantly up this month at $428.  Shelter is thus significantly up this month to $2840. (2026-05-01 to 2026-05-10)

# Shelter Rationalization

## Figures

*   **Shelter (May 1–10, 2026):** $2,840.55
*   **Shelter (Apr 1–30, 2026):** $2,653.20
*   **Utilities (May 1–10, 2026):** $428.15
*   **Utilities (Apr 1–30, 2026):** $241.80
*   **Home (May 1–10, 2026):** $2,412.40
*   **Home (Apr 1–30, 2026):** $2,412.40

## Drivers

Shelter costs rose this month almost entirely because **Utilities** accelerated, not because rent moved. The **Home** (mortgage) payment is identical to last month at $2,412.40 and posted on May 1 as usual. **Utilities** jumped mainly on PG&E: $286.40 on May 4 compared with $118.20 at this point last month, which Drivers tie to earlier cooling use and a tariff step-up that took effect May 1. The water district bill ($92.75) also landed May 3 instead of mid-month as in Apr, front-loading the category. **Upkeep** was $0 this month versus $42.50 in Apr when a sprinkler repair posted — that absence partially offsets the utility spike but does not fully neutralize it. Net shelter is up $187.35 versus the full prior month even though fixed housing charges are flat.

## Helpful Links to Information

- [Shelter](/cashflow/14/monthly/2026-05)
- [Home](/cashflow/15/monthly/2026-05)
- [Utilities](/cashflow/16/monthly/2026-05)
- [Upkeep](/cashflow/17/monthly/2026-05)

# Entertainment Insight

Explain: Entertainment is significantly up last week at $186. (2026-05-03 to 2026-05-09)

# Entertainment Rationalization

## Figures

*   **Entertainment (May 3–9, 2026):** $186.40
*   **Entertainment (Apr 26–May 2, 2026):** $24.00
*   **Entertainment (Apr 19–25, 2026):** $31.50

## Drivers

Last week’s entertainment spike is concentrated in two tickets: **AMC Theaters** $68.50 on May 4 and **StubHub — Indie Fest** $94.00 on May 6. A $23.90 **Spotify** renewal also posted May 3; that recurring charge appears most weeks but the concert and movie lines drove the step-change versus the prior week’s $24.00 total, which was essentially streaming only.

## Helpful Links to Information

- [Entertainment](/cashflow/6/weekly/2026-05-03)
""",
        "output": """
# Top Takeaways

## Highlights

- **Fixed Housing:** This month, [Home](/cashflow/15/monthly/2026-05) mortgage payments were managed to remain flat at $2,412.

## Lowlights

- **Shelter Surge:** Shelter is up this month to $2,841, primarily because of a surge in [Utilities](/cashflow/16/monthly/2026-05) to $428. This is driven by a PG&E charge of $286—significantly higher than the $118 seen at this point last month—alongside an early water district bill of $93. Although the absence of [Upkeep](/cashflow/17/monthly/2026-05) costs, which were $43 in Apr, provides a minor offset, the overall [Shelter](/cashflow/14/monthly/2026-05) category remains elevated compared to the $2,653 total from last month.
- **Entertainment Spike:** Spending on entertainment rose to $186 last week, a sharp increase from the $24 recorded in the prior week. This jump in [Entertainment](/cashflow/6/weekly/2026-05-03) is concentrated in two specific transactions: $69 for AMC Theaters and $94 for StubHub, which combined with the recurring $24 Spotify renewal to drive the category well above the previous week's streaming-only baseline.
""",
    },
    {
        "name": "five_contexts_health_education_shopping_mixed_grains",
        "input": """
# Health Insight

Explain: Medical & Pharmacy is significantly up this month at $342.  Health is thus significantly up this month to $518. (2026-05-01 to 2026-05-10)

# Health Rationalization

## Figures

*   **Health (May 1–10, 2026):** $518.60
*   **Health (Apr 1–30, 2026):** $212.30
*   **Medical & Pharmacy (May 1–10, 2026):** $342.15
*   **Medical & Pharmacy (Apr 1–30, 2026):** $88.40
*   **Gym & Wellness (May 1–10, 2026):** $176.45
*   **Gym & Wellness (Apr 1–30, 2026):** $123.90

## Drivers

Medical & Pharmacy dominates the health lift: **CVS Pharmacy** $156.80 and **Kaiser Copay** $142.00 posted May 2–5 versus $88.40 for all of last month. Gym & Wellness rose mainly because annual **Peloton** renewal $176.45 hit May 1; last month gym lines were $123.90 with no renewal.

## Helpful Links to Information

- [Health](/cashflow/28/monthly/2026-05)
- [Medical & Pharmacy](/cashflow/29/monthly/2026-05)
- [Gym & Wellness](/cashflow/30/monthly/2026-05)
- [Personal Care](/cashflow/31/monthly/2026-05)

# Tuition Insight

Explain: Tuition is slightly up last week at $650. (2026-05-03 to 2026-05-09)

# Tuition Rationalization

## Figures

*   **Tuition (May 3–9, 2026):** $650.00
*   **Tuition (Apr 26–May 2, 2026):** $0.00

## Drivers

Spring semester installment to **State University** posted May 5.

## Helpful Links to Information

- [Tuition](/cashflow/20/weekly/2026-05-03)

# Clothing Insight

Explain: Clothing is significantly down this month at $45. (2026-05-01 to 2026-05-10)

# Clothing Rationalization

## Figures

*   **Clothing (May 1–10, 2026):** $45.00
*   **Clothing (Apr 1–30, 2026):** $312.80
*   **Clothing (Mar 1–31, 2026):** $189.50

## Drivers

No department-store runs this month; the only charge is **Target** socks $45.00 on May 7. Last month included **Nordstrom** $198.00 and **Old Navy** $114.80.

## Helpful Links to Information

- [Clothing](/cashflow/22/monthly/2026-05)

# Gadgets Insight

Explain: Gadgets is significantly up this month at $279. (2026-05-01 to 2026-05-10)

# Gadgets Rationalization

## Figures

*   **Gadgets (May 1–10, 2026):** $279.99
*   **Gadgets (Apr 1–30, 2026):** $0.00

## Drivers

**Apple Store** Magic Keyboard $279.99 on May 8.

## Helpful Links to Information

- [Gadgets](/cashflow/23/monthly/2026-05)

# Donations & Gifts Insight

Explain: Donations & Gifts is flat this month at $50. (2026-05-01 to 2026-05-10)

# Donations & Gifts Rationalization

## Figures

*   **Donations & Gifts (May 1–10, 2026):** $50.00
*   **Donations & Gifts (Apr 1–30, 2026):** $50.00

## Drivers

Recurring **Red Cross** $50.00 on May 1.

## Helpful Links to Information

- [Donations & Gifts](/cashflow/32/monthly/2026-05)
""",
        "output": """
# Top Takeaways

## Highlights

- **Clothing Savings:** Spending on clothing is significantly down this month at $45 compared to $313 last month, as the absence of department-store runs has left [Clothing](/cashflow/22/monthly/2026-05) with only a single $45 charge at Target on May 7, whereas last month included larger purchases at Nordstrom and Old Navy.
- **Consistent Giving:** Giving remains consistent this month at $50, matching the $50 total from last month, driven entirely by the recurring [Donations & Gifts](/cashflow/32/monthly/2026-05) payment to Red Cross on May 1.

## Lowlights

- **Health Surge:** Costs are significantly up this month at $519 compared to $212 last month, primarily due to a surge in [Medical & Pharmacy](/cashflow/29/monthly/2026-05) expenses—which reached $342 this month versus $88 last month—following CVS Pharmacy and Kaiser Copay charges posted May 2–5. Additionally, [Gym & Wellness](/cashflow/30/monthly/2026-05) spending rose to $176 this month, driven by an annual Peloton renewal on May 1, compared to $124 in gym lines last month.
- **Tuition Up:** Educational expenses are slightly up this week at $650, reflecting a spring semester installment to State University that posted May 5, whereas [Tuition](/cashflow/20/weekly/2026-05-03) saw no activity in the prior week.
- **Gadgets Splurge:** Spending is significantly up this month at $280, a sharp increase from $0 last month, driven by a single [Gadgets](/cashflow/23/monthly/2026-05) purchase of an Apple Store Magic Keyboard on May 8.
""",
    },
]

_RE_CENTS = re.compile(r"\$\d{1,3}(?:,\d{3})*\.\d{2}")
_RE_IDEAL_LINK = re.compile(r"\]\((/cashflow/[^)]+)\)")
_RE_MD_LINK = re.compile(r"\[[^\]]*\]\([^)]+\)")


def _prose_without_links(text: str) -> str:
    return _RE_MD_LINK.sub("", text or "")


def mechanical_sandbox_check(*, ideal: str, actual: str) -> dict[str, Any]:
    """Heuristic checks vs ideal output: structure, links, amounts, period wording."""
    issues: list[str] = []
    ideal_s = (ideal or "").strip()
    actual_s = (actual or "").strip()
    if "# Top Takeaways" not in actual_s:
        issues.append("Missing header: # Top Takeaways")
    if "## Highlights" not in actual_s:
        issues.append("Missing header: ## Highlights")
    if "## Lowlights" in ideal_s and "## Lowlights" not in actual_s:
        issues.append("Missing header: ## Lowlights")
    ideal_links = _RE_IDEAL_LINK.findall(ideal_s)
    for path in ideal_links:
        if path not in actual_s:
            issues.append(f"Missing ideal link path: {path}")
    prose = _prose_without_links(actual_s)
    for m in _RE_CENTS.findall(prose):
        issues.append(f"Unrounded amount (cents): {m}")
    banned = (
        "first 10 days",
        "first X days",
        " so far",
        "April",
        "February",
        "January ",
        "the previous week",
        "None.",
    )
    for phrase in banned:
        if phrase in prose:
            issues.append(f"Banned phrasing in prose: {phrase!r}")
    takeaway_body = actual_s.split("# Top Takeaways", 1)[-1] if "# Top Takeaways" in actual_s else actual_s
    for stop in ("## Helpful Links", "\n***", "\n# "):
        if stop in takeaway_body:
            takeaway_body = takeaway_body.split(stop, 1)[0]
    hi = takeaway_body.split("## Lowlights")[0].split("## Highlights")[-1] if "## Highlights" in takeaway_body else ""
    lo = takeaway_body.split("## Lowlights")[-1] if "## Lowlights" in takeaway_body else ""
    hi_count = hi.count("\n- ")
    lo_count = lo.count("\n- ")
    if hi_count > 3:
        issues.append(f"Too many Highlights bullets: {hi_count}")
    if lo_count > 3:
        issues.append(f"Too many Lowlights bullets: {lo_count}")
    good = not issues
    return {"good_copy": good, "issues": issues, "eval_text": "\n".join(f"- {x}" for x in issues) if issues else "OK"}


def _run_test(user_message: str, optimizer: TopTakeawaysVerboseOptimizer | None = None) -> str:
    if optimizer is None:
        optimizer = TopTakeawaysVerboseOptimizer()
    print("=" * 80)
    print("LLM USER MESSAGE (top takeaways task):")
    print("=" * 80)
    print(user_message)
    print("=" * 80 + "\n")
    result = optimizer.generate_markdown(user_message)
    print("=" * 80)
    print("LLM OUTPUT:")
    print("=" * 80)
    print(result)
    print("=" * 80 + "\n")
    return result


def get_test_case(name_or_index: str | int) -> dict[str, Any] | None:
    if isinstance(name_or_index, int):
        if 0 <= name_or_index < len(TEST_CASES):
            return TEST_CASES[name_or_index]
        return None
    for tc in TEST_CASES:
        if tc["name"] == name_or_index:
            return tc
    return None


def run_test(
    name_or_index: str | int,
    optimizer: TopTakeawaysVerboseOptimizer | None = None,
    *,
    run_sandbox: bool = True,
) -> str | None:
    tc = get_test_case(name_or_index) if not isinstance(name_or_index, dict) else name_or_index
    if tc is None:
        print(f"Test case {name_or_index!r} not found.")
        return None
    print(f"\n{'=' * 80}\nRunning test: {tc['name']}\n{'=' * 80}\n")
    result = _run_test(tc["input"], optimizer)
    if run_sandbox and result and tc.get("output"):
        print("=" * 80)
        print("IDEAL OUTPUT:")
        print("=" * 80)
        print(tc["output"].strip())
        print("=" * 80)
        print("SANDBOX EXECUTION:")
        print("=" * 80)
        sand = mechanical_sandbox_check(ideal=tc["output"], actual=result)
        print(sand["eval_text"])
        print(f"{'✅' if sand['good_copy'] else '❌'} good_copy={sand['good_copy']}")
        print("=" * 80 + "\n")
    return result


def main(*, test: str | None) -> None:
    if test is None:
        print("Usage: python3 active_experiments/top_takeaways_verbose_optimizer.py --test <index|name|all>")
        print("Tests:")
        for i, tc in enumerate(TEST_CASES):
            print(f"  {i}: {tc['name']}")
        return
    optimizer = TopTakeawaysVerboseOptimizer(thinking_budget=TOP_TAKEAWAYS_THINKING_BUDGET)
    if test.strip().lower() == "all":
        for i, tc in enumerate(TEST_CASES):
            run_test(i, optimizer)
            if i < len(TEST_CASES) - 1:
                print("\n" + "-" * 80 + "\n")
        return
    key: str | int = int(test) if test.isdigit() else test
    run_test(key, optimizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Top takeaways verbose optimizer (Gemini one-shot)")
    parser.add_argument("--test", type=str, default=None, help='Index, test name, or "all"')
    args = parser.parse_args()
    main(test=args.test)
