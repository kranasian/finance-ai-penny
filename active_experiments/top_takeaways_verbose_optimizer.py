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

You will be given multiple **top takeaway context** blocks as plain markdown, separated by blank lines. Each block contains:
1. **Body** — **either** full prior rationalize markdown (``# {Category} Insight`` with ``Explain:``, then ``# {Category} Rationalization`` with ``## Figures`` / ``## Drivers``) **or** a **single non-rationalized insight string** (no Insight/Rationalization headings — e.g. uncategorized or large-transaction alerts).
2. ``## Helpful Links to Information`` — bullet list of Markdown links ``- [display name](/cashflow/...)`` (one or more per block). Copy ``href`` paths verbatim from this section.

Your goal is to synthesize a **substantive rollup** across all contexts: **highlights** (positive patterns, wins, healthy habits) and **lowlights** (concerns, risks, drag, overspending, gaps). Each bullet should read as a **mini-briefing**—rich enough that a reader grasps the **what**, **how much**, **compared to what**, and **why** (drivers) without opening the drill-down, while staying within the bullet budget.

**What to use for Highlights / Lowlights:** For rationalize blocks, base bullets on **Figures and Drivers** plus the ``Explain:`` line under ``# {Category} Insight``. **Go deep:** weave in **multiple** concrete details from Figures (period labels, dollar amounts, prior-window comparisons) and Drivers (named payees, payroll or bonus lines, miscategorization examples, volume or timing mechanisms). Prefer **several connected sentences** per bullet when the source supports it—do **not** compress rich rationalize into a thin headline if the Figures/Drivers give more. Copy numbers and dates **exactly** as written. For plain insight blocks: the **only** ground truth is that insight text — there are no Figures/Drivers; do **not** invent amounts or merchants not stated there. For uncategorized / large-transaction insights, write in the same **observational** register as **vs-forecast** rationalize **Drivers** (declarative, third person: what the spend reflects, what is driven by, what posted when) — **no** invented action items or imperatives (e.g. no “confirm”, “verify”, “should”, “until you act”, “tidy”, “worth checking”).

Rules:
- Ground every bullet in the provided body for that context. For rationalize blocks, ground claims in ``## Figures`` and ``## Drivers`` as well as ``Explain:``. Do not invent transactions, ids, or amounts.
- **Depth (rationalize blocks):** Within each ``- `` bullet, use **as much relevant Figures/Drivers material as fits**—typically **two to four sentences** when the context is meaty. Include **at least two** concrete facts (e.g. two amounts, or an amount plus a named driver, or a comparison across periods plus the mechanism). If you only restate ``Explain:`` without Figures/Drivers specifics, the bullet is **too shallow**.
- **Tap link anchor:** For rationalize blocks, choose the phrase you wrap in ``[...](path)`` **only** from ``# {Category} Insight`` / ``Explain:`` (ignore ``## Figures``, ``## Drivers``, and slugs like ``meals_groceries`` for this choice):
  1. If Explain **names a parent / umbrella category** in that same line, use **that parent’s wording** (verbatim casing from Explain) as the **primary** linked label.
  2. **Otherwise** use the **first leaf** category name in Explain (first such name in reading order).
  3. For plain insight blocks, choose a **short natural phrase from the insight text itself** as the link anchor (e.g. merchant name like “Property Group LLC” or “Burger King”, the word **Uncategorized**, or **Clothing** when it names a likely category — pick the clearest single anchor that fits the sentence; use **that** visible text with a path from that block’s **Helpful Links** list).
  4. At least one Markdown link per bullet that summarizes a context must use the chosen anchor and **exactly** one ``/cashflow/...`` path from that block’s **Helpful Links** section as ``href`` (add a second link only if another context is also summarized in the same bullet).
- **Tap label vs Markdown anchor (hard rule):** When a helpful link is ``[Name](/cashflow/...)``, the ``[...]`` anchor you use in the rollup for **that** path must name the **same** category as **Name**—do not attach that URL to a different category label (e.g. do not write **Groceries** or **Dining** on a link whose helpful-link line says **Food**). If Explain’s wording would disagree with **Name**, align the **link anchor** with **Name** and carry Explain’s nuance in the surrounding sentence without mislabeling the drill-down.
- **Tap links:** Use **only** paths listed under **Helpful Links to Information** for each block. **Link syntax:** ``[text](/path)`` with **no whitespace** between ``(`` and ``/`` — never ``[text]( /path)``. If multiple contexts appear in one bullet, include one correct link per context. **Never** paste one context’s ``/cashflow/...`` URL next to another context’s category name.
- **Anchor vs path:** The ``href`` must be **exactly** a path from that context block’s **Helpful Links** list. The visible ``[anchor]`` must name the **same** Penny category level that URL opens (parent rollup vs leaf): align with the **display name** in the helpful-link bullet when it names a category; do not write **Food** on a path that opens **Groceries** or **Meals** alone, or **Groceries** on a path meant for a different rollup. Do not paste another context’s path into a sentence tied to a different Explain theme.
- **Link in sentence:** The linked category must be **woven into the sentence** as normal grammar (subject, object, or “in …” phrase) — not bolted on after a period. **Bad:** ``…following bonus payouts in March. [Salary](/cashflow/…)`` **Good:** ``…following March bonus payouts; [Salary](/cashflow/…) matches your usual early-month rhythm`` or ``Early-month [Salary](/cashflow/…) matches April…``. Do not end a clause with ``.`` and then append only a bare link.
- **Tools:** You may call finance tools when helpful—especially to **verify** totals, reconcile conflicting numbers across contexts, or confirm whether a claim in the markdown still matches current data. Prefer tools when verification would materially increase confidence; otherwise rely on the pasted rationalize text.
- Merge semantically duplicate ideas into one bullet; avoid repeating the same point under different wording. When merging, keep the **combined** bullet **fully detailed** (do not drop figures or drivers just to shorten).
- **Bullet budget:** At most **3** ``- `` bullets under ``## Highlights`` and at most **3** under ``## Lowlights`` (six total max). If contexts are very thin, use fewer; never exceed 3 per section. **Prefer longer, denser bullets** over many short ones—use the budget for **depth**, not for splitting one story into vague fragments.
- Your **written reply** must not include raw ``llm_calls`` text or pasted tool-call payloads.
- **Final reply:** After any tool calls, your **only** user-visible text follows the markdown format below — **no** standalone preamble or wrap-up outside it. The first line of your last message must be exactly ``# Top Takeaways``.

Final message format (MUST match exactly — nothing else in the last message):

# Top Takeaways

## Highlights

- ...

## Lowlights

- ...

Notes:
- Use `- ` bullets only (no numbered lists in these sections). Inline Markdown links ``[text](/path)`` inside bullets are required for each context summarized; anchor text follows **Tap link anchor** rules (Explain-based for rationalize blocks, insight-text-based for plain insight blocks), subject to **Tap label vs Markdown anchor** when helpful links include a display name; ``href`` is a ``/cashflow/...`` path from that block’s **Helpful Links** section with **no space** after ``(``. The linked phrase must read as part of the sentence (**Link in sentence** rule), not a trailing tag after the final period. **Anchor vs path** (same subsection in Rules): label and ``href`` must describe the same category level, and you must not apply one context’s URL to another context’s category name.
- Match the rationalize voice: **direct, readable, and thorough**—high information density per bullet, not telegraphic stubs. For ``uncat_txn`` / ``large_txn`` insights, match **vs-forecast Drivers** observational tone, not advisory copy.
- Start each bullet with a **bold** short label when helpful (e.g. **Savings:** …).
- Do not add introductory or closing sentences outside the ``# Top Takeaways`` structure.
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

Explain: Salary is significantly down this month at $6369. (2026-05-01 to 2026-05-10)

# Salary Rationalization

## Figures

*   **May 1–10, 2026:** $6,369.24
*   **April 1–10, 2026:** $6,369.24
*   **March 1–10, 2026:** $17,369.24

## Drivers

The "significant drop" in salary compared to early March is due to a large, one-time bonus payout received in the first half of March 2026. Specifically, you received $8,800 in total "Genentech US Bonus" payments on March 11, alongside higher "CA State Payroll" amounts ($8,600 total) compared to your standard bi-weekly payroll cycle. Your income for the first 10 days of May is consistent with your regular pay cycle, mirroring the amount received during the same period in April.

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

- **Salary rhythm:** Early-month [Salary](/cashflow/36/monthly/2026-05) through May 10 matches April at $6,369.24 while March’s early-month lift came from a one-time Genentech bonus and higher CA State Payroll, not a pay cut.
- **Food framing:** [Food](/cashflow/1/monthly/2026-05) looks “down” partly because only 10 days of May are counted vs full April/March totals; Applebee’s dining charges may be sitting under groceries.

## Lowlights

- **Uncategorized spike:** Last week [Uncategorized](/cashflow/-1/weekly/2026-05-03) jumped to about $6,382 on large Property Group LLC payments and possible duplicate recurring charges (Community Pool, Costco, BP).
- **Uncategorized trend:** Month-to-date [Uncategorized](/cashflow/-1/monthly/2026-05) is slightly down vs March/April while big Property Group LLC repeats still dominate with fewer small misc transactions so far in May.
""",
    },
    {
        "name": "two_contexts_salary_and_groceries_with_tap_links",
        "input": """
# Salary Insight

Explain: Salary is significantly down this month at $6369. (2026-05-01 to 2026-05-10)

# Salary Rationalization

## Figures

*   **May 1–10, 2026:** $6,369.24
*   **April 1–10, 2026:** $6,369.24
*   **March 1–10, 2026:** $17,369.24

## Drivers

The "significant drop" in salary compared to early March is due to a large, one-time bonus payout received in the first half of March 2026. Specifically, you received $8,800 in total "Genentech US Bonus" payments on March 11, alongside higher "CA State Payroll" amounts ($8,600 total) compared to your standard bi-weekly payroll cycle. Your income for the first 10 days of May is consistent with your regular pay cycle, mirroring the amount received during the same period in April.

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
""",
        "output": """
# Top Takeaways

## Highlights

- **Income stability:** May 1–10 [Salary](/cashflow/36/monthly/2026-05) aligns with April’s same window at $6,369.24; March’s higher early-month income reflected bonus and payroll timing.
- **Groceries vs dining:** [Food](/cashflow/1/monthly/2026-05) looks lower partly from comparing 10 days of May to full months; Applebee’s may be miscategorized under groceries.

## Lowlights

- **Applebee’s pattern:** Dining at Applebee’s recorded under groceries inflates grocery totals until those charges are recategorized.
""",
    },
    {
        "name": "uncat_txn_large_txn_plus_four_vs_forecast",
        "input": """
Uncategorized outflow transaction for Property Group LLC for $2,850 with a likely category of Clothing.

## Helpful Links to Information

- [Property Group LLC](/cashflow/transaction/123)

Large outflow transaction with Burger King for $40 last 05/09.

## Helpful Links to Information

- [Burger King](/cashflow/transaction/456)

# Travel & Vacations Insight

Explain: Leisure is significantly up this month at $412. (2026-05-01 to 2026-05-10)

# Travel & Vacations Rationalization

## Figures

*   **Leisure (May 1–10, 2026):** $412.18
*   **Leisure (Apr 1–30, 2026):** $251.00

## Drivers

Higher leisure spend vs forecast is driven by weekend entertainment and one concert ticket purchase.

## Helpful Links to Information

- [Travel & Vacations](/cashflow/7/monthly/2026-05)

# Pets Insight

Explain: Transport is significantly down this month at $68. (2026-05-01 to 2026-05-10)

# Pets Rationalization

## Figures

*   **Transport (May 1–10, 2026):** $68.40
*   **Transport (Apr 1–30, 2026):** $310.25

## Drivers

Lower transport vs forecast reflects fewer commutes and no fuel fill-ups yet in early May.

## Helpful Links to Information

- [Pets](/cashflow/8/monthly/2026-05)

# Bills Insight

Explain: Bills is slightly up this month at $1,240. (2026-05-01 to 2026-05-10)

# Bills Rationalization

## Figures

*   **Bills (May 1–10, 2026):** $1,240.55
*   **Bills (Apr 1–30, 2026):** $1,180.00

## Drivers

Slight uptick vs forecast from annual software renewal posting in the first week of May.

## Helpful Links to Information

- [Bills](/cashflow/9/monthly/2026-05)
- [Connectivity](/cashflow/10/monthly/2026-05)
- [Insurance](/cashflow/11/monthly/2026-05)
- [Taxes](/cashflow/12/monthly/2026-05)
- [Service Fees](/cashflow/13/monthly/2026-05)

# Shopping Insight

Explain: Shopping is significantly up this month at $295. (2026-05-01 to 2026-05-10)

# Shopping Rationalization

## Figures

*   **Shopping (May 1–10, 2026):** $295.00
*   **Shopping (Apr 1–30, 2026):** $142.30

## Drivers

Shopping vs forecast rose after two online orders (electronics accessories and home goods) early in the month.

## Helpful Links to Information

- [Shopping](/cashflow/44/monthly/2026-05)
- [Pets](/cashflow/8/monthly/2026-05)
- [Clothing](/cashflow/22/monthly/2026-05)
- [Gadgets](/cashflow/23/monthly/2026-05)
- [Kids](/cashflow/24/monthly/2026-05)
""",
        "output": """
# Top Takeaways

## Highlights

- **Uncategorized outflow:** The insight flags [Property Group LLC](/cashflow/transaction/123) as an uncategorized **$2,850** outflow with **Clothing** as the likely category.
- **Leisure vs plan:** [Leisure](/cashflow/7/monthly/2026-05) is up vs forecast on weekend entertainment and a concert ticket early in May.
- **Shopping lift:** [Shopping](/cashflow/44/monthly/2026-05) is up vs forecast after two online orders for accessories and home goods.

## Lowlights

- **Uncategorized mix:** Uncategorized totals still include [Property Group LLC](/cashflow/transaction/123) at **$2,850** while that charge posts without a category.
- **Large dining outflow:** [Burger King](/cashflow/transaction/456) shows a **$40** single-ticket outflow on 05/09.
- **Commutes and bills:** [Transport](/cashflow/8/monthly/2026-05) is down vs forecast with fewer commutes and no fuel fill-ups yet, while [Bills](/cashflow/9/monthly/2026-05) ticked slightly up on an annual software renewal in early May.
""",
    },
]

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


def run_test(name_or_index: str | int, optimizer: TopTakeawaysVerboseOptimizer | None = None) -> str | None:
    tc = get_test_case(name_or_index) if not isinstance(name_or_index, dict) else name_or_index
    if tc is None:
        print(f"Test case {name_or_index!r} not found.")
        return None
    print(f"\n{'=' * 80}\nRunning test: {tc['name']}\n{'=' * 80}\n")
    return _run_test(tc["input"], optimizer)


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
