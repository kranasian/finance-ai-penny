"""
Rationalize-by-category prompt optimizer: simulate a real Hermes agent run.

This script runs a real Hermes batch via the Hermes CLI entrypoint
(`finance-ai-hermes/handlers/run_rationalize.py`) and passes a system prompt override.
This avoids local `hermes-agent` dependency issues in the finance-ai-penny venv.

Run from `finance-ai-penny` repo root:

  python active_experiments/rationalize_by_category_optimizer.py --user-id 3 --test all
  python active_experiments/rationalize_by_category_optimizer.py --user-id 3 --test batch_1
  python active_experiments/rationalize_by_category_optimizer.py --user-id 3 --count 3 --offset 0
  python active_experiments/rationalize_by_category_optimizer.py --user-id 3 --count 1 --offset 0 --quiet
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PENNY_REPO_ROOT = os.path.dirname(_THIS_DIR)
_HERMES_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(_PENNY_REPO_ROOT), "finance-ai-hermes"))
_HERMES_CLI = os.path.join(_HERMES_REPO_ROOT, "handlers", "run_rationalize.py")
_HERMES_VENV_PY = os.path.join(_HERMES_REPO_ROOT, ".venv", "bin", "python3")


def _resolve_hermes_python(explicit_path: str | None) -> str:
  if explicit_path:
    return explicit_path
  if os.path.exists(_HERMES_VENV_PY):
    return _HERMES_VENV_PY
  return sys.executable


SYSTEM_PROMPT = (
  "You are **Penny**. You only state dollar facts from **tools**—never invent amounts, merchants, or dates.\n\n"
  "**Insight context (no forecast / “expected” SQL here):** The user message is usually ``Explain: …`` from an insight. "
  "That insight’s “expected” side reflects **past average spending (or income patterns) at this point in the week or month**—"
  "not something you re-query as forecast rows in this agent. **Do not** say you loaded forecast tables.\n\n"
  "**Your job:** explain *why* the insight happened with tool-backed figures, and propose next steps that Penny can actually do "
  "(budgets/goals, recategorization, categorization rules).\n\n"
  "**Before you write (investigation algorithm):**\n"
  "1) **Verify totals efficiently**: use ``lookup_user_aggregate_spending`` with **monthly** granularity for **this month** and the **prior 2 months**. "
  "Prefer **one call per month** by passing a category list (e.g. ``['shelter_home','shelter_utilities']``) rather than separate calls. "
  "Use a mid-month ``date_in_range`` (the 15th) to avoid month-end edge cases.\n"
  "2) **Resolve category naming**: if user uses display words (e.g. “Home”, “Utilities”, “Shelter”), map to canonical categories. "
  "If the mapping is unclear or aggregates look wrong/zero, run ``retrieve_user_spending_transactions_by_sql`` "
  "(``SELECT DISTINCT category ... LIMIT 50``) once and retry with canonical categories (e.g. ``shelter_home``, ``shelter_utilities``).\n"
  "3) **Isolate the driver**: compute the delta vs last month and identify the **top 1–2 changing categories**.\n"
  "4) **Explain the driver with line items** (required when delta is noticeable or the insight claims a spike/drop): "
  "run transaction SQL scoped to the *driver category* for this month and last month, and cite the top 1–3 merchants/charges "
  "(or state “no transactions found” if true). Look for missing/shifted recurring bills and timing.\n\n"
  "**Final answer — structure (must match headings):**\n"
  "Use these exact headings: ``## Figures``, then ``## Drivers``, then ``## Next steps``.\n\n"
  "**Figures** (no preamble before heading): 3–6 bullets. Each bullet must include **$** and a **period** "
  "(e.g. Apr 1–30, 2026) and should include this month vs prior two months. Include a **reconciled total** if relevant.\n\n"
  "**Drivers**: 2–6 sentences. Tie the delta to **specific categories and (when needed) specific merchants/charges**. "
  "If the user’s statement is inconsistent with tools, say so plainly and show corrected tool-backed totals. "
  "When you cite a merchant/charge as a driver, prefer the format: ``Merchant: $this_month vs $last_month``.\n\n"
  "**Next steps** (Penny actions, not generic advice): 2–4 numbered lines (``1. ...``). Must include at least **one** of:\n"
  "- creating/updating a **budget** for the category (with a suggested $ target backed by recent months)\n"
  "- creating a **goal** (e.g. utilities cap / maintenance sinking fund)\n"
  "- a **recategorization** suggestion (if a transaction looks miscategorized) with a concrete example merchant\n"
  "- a **categorization rule** suggestion (merchant → category) when a pattern is clear\n"
  "If you can justify it from the data, include one concrete rule line like: ``Rule: Dominion Energy → shelter_utilities`` (no extra tool calls).\n\n"
  "**CRITICAL:**\n"
  "- Every number you cite must come from tools.\n"
  "- Don’t split totals across subcategories unless you queried them.\n"
  "- If tools disagree with the narrative, give 1–2 plausible reasons (mapping/window/pending) and proceed with tool-backed totals."
)


TEST_CASES: list[dict[str, Any]] = [
  {
    "name": "batch_1",
    "batch": 1,
    "count": 3,
    "offset": 0,
    "min_urgency": 2.0,
    "lookback_hours": 24,
  },
  {
    "name": "batch_2",
    "batch": 1,
    "count": 3,
    "offset": 3,
    "min_urgency": 2.0,
    "lookback_hours": 24,
  },
]


def run_test(
  tc: dict[str, Any],
  *,
  user_id: int,
  system_prompt_override: str | None = None,
  hermes_python: str | None = None,
  quiet: bool = False,
) -> dict[str, Any]:
  print(f"\n# Test: {tc.get('name')}\n")
  if not os.path.exists(_HERMES_CLI):
    raise SystemExit(f"Hermes CLI not found: {_HERMES_CLI}")

  prompt_text = system_prompt_override if system_prompt_override is not None else SYSTEM_PROMPT
  count = int(tc.get("count", 3) or 3)
  offset = int(tc.get("offset", 0) or 0)
  min_urgency = float(tc.get("min_urgency", 2.0) or 2.0)
  lookback_hours = int(tc.get("lookback_hours", 24) or 24)

  py = _resolve_hermes_python(hermes_python)
  cmd = [
    py,
    _HERMES_CLI,
    "--user-id",
    str(int(user_id)),
    "--count",
    str(count),
    "--offset",
    str(offset),
    "--min-urgency",
    str(min_urgency),
    "--lookback-hours",
    str(lookback_hours),
    "--prompt-template",
    prompt_text,
    "--no-persist",
    "--print-json",
    "--include-messages",
  ]
  print("## Command\n")
  print(" ".join([c if " " not in c else repr(c) for c in cmd]))
  print()

  proc = subprocess.run(
    cmd,
    cwd=_HERMES_REPO_ROOT,
    capture_output=True,
    text=True,
  )

  parsed = None
  stdout_str = proc.stdout.strip()
  if stdout_str:
    try:
      parsed = json.loads(stdout_str)
    except Exception:
      parsed = None

  print("## Hermes CLI stdout\n")
  if parsed is not None:
    print(json.dumps(parsed, indent=2, ensure_ascii=False))
  else:
    print(stdout_str)
  if proc.stderr.strip():
    if not quiet:
      print("\n## Hermes CLI stderr\n")
      print(proc.stderr.strip())

  return {"returncode": int(proc.returncode), "stdout": proc.stdout, "stderr": proc.stderr, "json": parsed}


def run_all_tests_batch(*, batch_num: int = 1, user_id: int, system_prompt_override: str | None = None) -> list[tuple[str, dict[str, Any]]]:
  cases = [tc for tc in TEST_CASES if tc.get("batch") == batch_num]
  results: list[tuple[str, dict[str, Any]]] = []
  for tc in cases:
    results.append((tc["name"], run_test(tc, user_id=user_id, system_prompt_override=system_prompt_override)))
  return results


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--test", type=str, default=None, help="Test name or index or 'all'.")
  parser.add_argument("--batch", type=int, default=None, help="Run a numbered batch (e.g. 1).")
  parser.add_argument("--user-id", type=int, required=True)
  parser.add_argument(
    "--hermes-python",
    type=str,
    default=None,
    help="Optional path to Hermes venv python. Defaults to finance-ai-hermes/.venv/bin/python3 if present.",
  )
  parser.add_argument(
    "--count",
    type=int,
    default=None,
    help="Ad-hoc run: number of insights to process (requires --offset too).",
  )
  parser.add_argument(
    "--offset",
    type=int,
    default=None,
    help="Ad-hoc run: offset into insights query (requires --count too).",
  )
  parser.add_argument("--min-urgency", type=float, default=2.0, help="Ad-hoc run: minimum urgency (default 2.0).")
  parser.add_argument("--lookback-hours", type=int, default=24, help="Ad-hoc run: lookback hours (default 24).")
  parser.add_argument(
    "--quiet",
    action="store_true",
    help="Suppress Hermes CLI stderr output (keeps JSON output readable).",
  )
  parser.add_argument(
    "--use-exact-hermes-prompt",
    action="store_true",
    help="Use the exact Hermes prompt from this file (default).",
  )
  parser.add_argument(
    "--system-prompt-file",
    type=str,
    default=None,
    help="Optional path to a file containing a system prompt override (for iteration).",
  )
  args = parser.parse_args()

  system_prompt_override = None
  if args.system_prompt_file:
    with open(args.system_prompt_file, "r", encoding="utf-8") as f:
      system_prompt_override = f.read()

  # Ad-hoc single run (useful for "one insight only")
  if (args.count is not None) or (args.offset is not None):
    if args.count is None or args.offset is None:
      raise SystemExit("For ad-hoc runs, pass both --count and --offset.")
    tc = {
      "name": "adhoc",
      "batch": 0,
      "count": int(args.count),
      "offset": int(args.offset),
      "min_urgency": float(args.min_urgency),
      "lookback_hours": int(args.lookback_hours),
    }
    run_test(
      tc,
      user_id=int(args.user_id),
      system_prompt_override=system_prompt_override,
      hermes_python=args.hermes_python,
      quiet=bool(args.quiet),
    )
    return

  if args.batch is not None:
    cases = [tc for tc in TEST_CASES if tc.get("batch") == int(args.batch)]
    for tc in cases:
      run_test(
        tc,
        user_id=int(args.user_id),
        system_prompt_override=system_prompt_override,
        hermes_python=args.hermes_python,
        quiet=bool(args.quiet),
      )
    return

  if not args.test:
    parser.print_help()
    return

  if args.test == "all":
    for tc in TEST_CASES:
      run_test(
        tc,
        user_id=int(args.user_id),
        system_prompt_override=system_prompt_override,
        hermes_python=args.hermes_python,
        quiet=bool(args.quiet),
      )
    return

  # Numeric index
  try:
    idx = int(args.test)
    if 0 <= idx < len(TEST_CASES):
      run_test(
        TEST_CASES[idx],
        user_id=int(args.user_id),
        system_prompt_override=system_prompt_override,
        hermes_python=args.hermes_python,
        quiet=bool(args.quiet),
      )
      return
  except ValueError:
    pass

  tc = next((t for t in TEST_CASES if t.get("name") == args.test), None)
  if not tc:
    raise SystemExit(f"Unknown test: {args.test!r}")
  run_test(
    tc,
    user_id=int(args.user_id),
    system_prompt_override=system_prompt_override,
    hermes_python=args.hermes_python,
    quiet=bool(args.quiet),
  )


if __name__ == "__main__":
  main()

