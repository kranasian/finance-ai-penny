"""
Rationalize-next-steps optimizer: simulate Hermes /rationalize_next_steps.

This runs inside the finance-ai-hermes virtualenv (if present) to avoid local dependency issues,
inserts a fixture row into `ai_agent_outcomes`, then POSTs to `/rationalize_next_steps`.

Run from `finance-ai-penny` repo root:

  python active_experiments/rationalize_next_steps_optimizer.py --user-id 3 --test all
  python active_experiments/rationalize_next_steps_optimizer.py --user-id 3 --test propose_create_goal
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
_HERMES_VENV_PY = os.path.join(_HERMES_REPO_ROOT, ".venv", "bin", "python3")

SYSTEM_PROMPT = (
  "You are **Penny**.\n\n"
  "You will be given an **agent_outcome** (markdown) produced by a prior rationalize run. "
  "Your goal is to propose the most helpful *next* actions Penny should take, using ONLY the proposal tools available.\n\n"
  "Rules:\n"
  "- Prefer calling **1–3 tools total**.\n"
  "- Each tool call must include a clear, user-facing **rationale**.\n"
  "- Do not invent transaction ids.\n"
  "- If the next step is to recategorize a transaction and the input does not include transaction ids, you MUST first "
  "use `retrieve_user_spending_transactions_by_sql` to find the matching transaction_id(s) (by merchant name + date + amount), "
  "then call `propose_recategorize_transactions` with those ids.\n"
  "- These are **proposals** only. Do **not** claim you executed changes.\n"
  "- After tool calls, return a concise human summary under the heading `## Proposed next steps`.\n"
  "- Prefer imperative voice (e.g. \"Create…\", \"Set…\", \"Recategorize…\"; avoid \"has been proposed\" / \"a rule was proposed\").\n"
)

# Optional overrides for Hermes agent settings (model/limits). Adjust per experiment.
AGENT_SETTINGS: dict[str, Any] = {}


def _resolve_hermes_python(explicit_path: str | None) -> str:
  if explicit_path:
    return explicit_path
  if os.path.exists(_HERMES_VENV_PY):
    return _HERMES_VENV_PY
  return sys.executable


def _hermes_inline_runner() -> str:
  # Runs in the Hermes repo cwd. Reads JSON from stdin:
  # { "user_id": int, "agent_outcome_md": str, "system_prompt": str|None, "agent_settings": dict|None, "mock_spending_sql_result": dict|None }
  # Prints JSON to stdout:
  # { "agent_outcome_id": int, "response": {...endpoint json...} }
  return r"""
import json
import sys
import traceback
import os

from flask_app import app

try:
  payload_in = json.loads(sys.stdin.read() or "{}")
  uid = int(payload_in["user_id"])
  agent_outcome_md = str(payload_in["agent_outcome_md"])
  system_prompt = payload_in.get("system_prompt")
  agent_settings = payload_in.get("agent_settings")
  mock_spending_sql_result = payload_in.get("mock_spending_sql_result")

  if mock_spending_sql_result is not None:
    # Optimizer-only mock: patch the finance tool handler before the agent/plugin system loads it.
    plugins_dir = os.path.join(os.getcwd(), "plugins")
    if plugins_dir not in sys.path:
      sys.path.insert(0, plugins_dir)
    import finance_tools.retrieve_user_spending_transactions_by_sql as _tool_mod

    def _mock_handler(args: dict, **kwargs) -> str:
      return json.dumps(
        {
          "success": True,
          "user_id": uid,
          "count": len((mock_spending_sql_result or {}).get("data") or []),
          "columns": (mock_spending_sql_result or {}).get("columns") or [],
          "data": (mock_spending_sql_result or {}).get("data") or [],
          "mocked": True,
        },
        default=str,
      )

    _tool_mod.handler = _mock_handler

  client = app.test_client()
  resp = client.post(
    "/rationalize_next_steps",
    json={
      "user_id": uid,
      "agent_outcome": agent_outcome_md,
      "system_prompt": system_prompt,
      "agent_settings": agent_settings,
    },
  )
  try:
    out_json = resp.get_json(force=True)
  except Exception:
    out_json = {"raw": resp.get_data(as_text=True)}

  print(json.dumps({"response": out_json}, ensure_ascii=False))
except Exception as e:
  print(json.dumps({"error": str(e), "trace": traceback.format_exc()}, ensure_ascii=False))
""".strip()


TEST_CASES: list[dict[str, Any]] = [
  {
    "name": "propose_create_goal",
    "agent_outcome_md": """# Rationalize What

Explain: Clothing is significantly down this month at $36. (2026-04-01 to 2026-04-29)

# Rationalize Response

## Figures

*   **Clothing (Apr 1–30, 2026):** $35.60
*   **Clothing (Mar 1–31, 2026):** $74.94
*   **Clothing (Feb 1–28, 2026):** $78.33

## Drivers

The decrease in clothing spending is driven by a reduction in both the frequency and total value of purchases. In March, your spending was concentrated in a single $74.94 transaction at H&M. In April, your only clothing-related transaction was $35.60 at Adidas. This represents a $39.34 decrease compared to last month.

## Next steps

1.  **Set a Clothing Budget:** Based on your average spending over the last three months (~$63), consider setting a monthly clothing budget of $60 to keep spending steady.
2.  **Review Categorization:** If you frequently shop for items that might be miscategorized as clothing, verify your recent transactions; if they look correct, keep the current categorization rule for Adidas and H&M.
""",
    "expects_tool": "propose_create_goal",
  },
  {
    "name": "propose_create_categorization_rule",
    "agent_outcome_md": """# Rationalize What

Explain: Groceries is significantly down this month at $120. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures
- Groceries: $120 this month (Apr 1–30, 2026), which seems too low given your purchase pattern.

## Drivers
Several Costco transactions appear miscategorized.

## Next steps
Create a categorization rule so Costco is always groceries.
""",
    "expects_tool": "propose_create_categorization_rule",
  },
  {
    "name": "multiple_tools_goal_and_rule",
    # Built from the same real-world pattern you shared (Costco miscategorized → needs rule),
    # but also includes a clear budget/goal next step so the next-steps agent should call 2 tools.
    "agent_outcome_md": """# Rationalize What

Explain: Groceries is significantly down this month at $120. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures
- Groceries: $120 this month (Apr 1–30, 2026), which seems too low given your purchase pattern.

## Drivers
Several Costco transactions appear miscategorized, which is making your Groceries totals misleading.

## Next steps
1. Create a categorization rule so Costco is always Groceries going forward (and apply it to the past).
2. Set a monthly Groceries budget so you can track progress once categorization is correct (e.g. $400/month).
""",
    "expects_tools": ["propose_create_categorization_rule", "propose_create_goal"],
  },
  {
    "name": "propose_recategorize_transactions",
    "agent_outcome_md": """# Rationalize What

Explain: Shopping is significantly up this month at $920. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures
- Shopping: $920 this month (Apr 1–30, 2026).

## Drivers
The spike is driven by a single large line item currently categorized under Shopping: **“Chase Credit Card Payment”** for **$920** on **2026-04-18**. This looks like an internal transfer / card payment, so counting it as Shopping inflates discretionary spend for the month.

## Next steps
Recategorize the “Chase Credit Card Payment” transaction on 2026-04-18 to Transfers.
""",
    "expects_tool": "propose_recategorize_transactions",
    "mock_spending_sql_result": {
      "columns": ["transaction_id", "date", "name", "amount", "account_id", "category"],
      "data": [
        {
          "transaction_id": 98765,
          "date": "2026-04-18",
          "name": "Chase Credit Card Payment",
          "amount": 920.00,
          "account_id": 12,
          "category": "shopping_clothing",
        }
      ],
    },
  },
]


def run_test(
  tc: dict[str, Any],
  *,
  user_id: int,
  hermes_python: str | None = None,
  quiet: bool = False,
) -> dict[str, Any]:
  test_name = tc.get("name") or "unnamed_test"
  divider = "=" * 88
  print(f"\n{divider}")
  print(f"TEST START: {test_name}")
  print(f"{divider}\n")
  if not os.path.exists(_HERMES_REPO_ROOT):
    raise SystemExit(f"Hermes repo not found: {_HERMES_REPO_ROOT}")

  py = _resolve_hermes_python(hermes_python)
  cmd = [py, "-c", _hermes_inline_runner()]

  stdin_obj = {"user_id": int(user_id), "agent_outcome_md": tc["agent_outcome_md"]}
  # Pass optional prompt/settings to Hermes so experiments can override behavior.
  stdin_obj["system_prompt"] = SYSTEM_PROMPT
  stdin_obj["agent_settings"] = AGENT_SETTINGS
  if tc.get("mock_spending_sql_result") is not None:
    stdin_obj["mock_spending_sql_result"] = tc["mock_spending_sql_result"]
  proc = subprocess.run(
    cmd,
    cwd=_HERMES_REPO_ROOT,
    input=json.dumps(stdin_obj),
    capture_output=True,
    text=True,
  )

  stdout_str = (proc.stdout or "").strip()
  parsed = None
  if stdout_str:
    # Hermes stdout may include extra log lines (e.g. DB connection debug) before the final JSON.
    # Parse the last JSON-looking line/object.
    candidate_lines = [ln.strip() for ln in stdout_str.splitlines() if ln.strip()]
    json_line = None
    for ln in reversed(candidate_lines):
      if ln.startswith("{") and ln.endswith("}"):
        json_line = ln
        break
    if json_line is None and candidate_lines:
      # Fallback: attempt to parse the full stdout (older runs may emit only JSON).
      json_line = stdout_str
    try:
      parsed = json.loads(json_line)
    except Exception:
      parsed = None

  print("## Hermes runner output (formatted)\n")
  resp = (parsed or {}).get("response") if isinstance(parsed, dict) else None
  if isinstance(resp, dict):
    agent_outcome = (resp.get("agent_outcome") or "").strip()
    proposals_out = resp.get("proposals") or []
    tool_calls_out = resp.get("tool_calls") or []
    final_text = (resp.get("response") or "").strip()
    uid_out = resp.get("user_id")

    print("### Agent outcome (input)\n")
    if agent_outcome:
      print(agent_outcome)
    else:
      print("(missing agent_outcome)")

    print("\n### Proposals (tool calls)\n")
    if isinstance(proposals_out, list) and proposals_out:
      for i, p in enumerate(proposals_out, start=1):
        if not isinstance(p, dict):
          continue
        tool_name = p.get("tool_name") or p.get("name") or "(unknown_tool)"
        args = p.get("arguments")
        print(f"{i}. {tool_name}")
        if args is not None:
          try:
            print(json.dumps(args, indent=2, ensure_ascii=False))
          except Exception:
            print(str(args))
        else:
          print("(no arguments)")
        print()
    else:
      print("(no proposals)")

    print("### Tool calls (all)\n")
    if isinstance(tool_calls_out, list) and tool_calls_out:
      for i, p in enumerate(tool_calls_out, start=1):
        if not isinstance(p, dict):
          continue
        tool_name = p.get("tool_name") or "(unknown_tool)"
        args = p.get("arguments")
        print(f"{i}. {tool_name}")
        if args is not None:
          try:
            print(json.dumps(args, indent=2, ensure_ascii=False))
          except Exception:
            print(str(args))
        else:
          print("(no arguments)")
        print()
    else:
      print("(no tool calls)")

    print("### Final LLM output (text)\n")
    print(final_text or "(missing response text)")
    if uid_out is not None:
      print(f"\n(user_id: {uid_out})")
  else:
    # Fallback to raw stdout when JSON parsing failed (or runner errored).
    if stdout_str:
      print(stdout_str)
    elif proc.stderr and proc.stderr.strip():
      # If stdout is empty, show stderr to make failures diagnosable.
      print(proc.stderr.strip())

  # Intentionally suppress Hermes runner stderr (noisy dependency logs).

  # Lightweight check: did the expected tool(s) appear in *any* tool calls?
  expected = tc.get("expects_tools") or tc.get("expects_tool")
  expected_list = expected if isinstance(expected, list) else ([expected] if expected else [])
  found = False
  found_tools: set[str] = set()
  try:
    resp_obj = (parsed or {}).get("response") or {}
    calls = resp_obj.get("tool_calls") or resp_obj.get("proposals") or []
    for p in calls:
      name = p.get("tool_name") or p.get("name")
      if name is None and isinstance(p.get("function"), dict):
        name = p["function"].get("name")
      if isinstance(name, str):
        found_tools.add(name)
    found = all((t in found_tools) for t in expected_list) if expected_list else False
  except Exception:
    found = False

  print("\n## Expected tool trigger\n")
  print(f"- expected_tools: {expected_list}")
  print(f"- observed_tools: {sorted(found_tools)}")
  print(f"- observed_all_expected: {found}")

  print(f"\n{divider}")
  print(f"TEST END: {test_name}")
  print(f"{divider}\n")

  return {
    "returncode": int(proc.returncode),
    "stdout": proc.stdout,
    "stderr": proc.stderr,
    "json": parsed,
    "expected_tools": expected_list,
    "observed_tools": sorted(found_tools),
    "observed_all_expected": found,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--test", type=str, default=None, help="Test name or index or 'all'.")
  parser.add_argument("--user-id", type=int, required=True)
  parser.add_argument(
    "--hermes-python",
    type=str,
    default=None,
    help="Optional path to Hermes venv python. Defaults to finance-ai-hermes/.venv/bin/python3 if present.",
  )
  parser.add_argument("--quiet", action="store_true")
  args = parser.parse_args()

  if not args.test:
    parser.print_help()
    return

  if args.test == "all":
    for tc in TEST_CASES:
      run_test(tc, user_id=int(args.user_id), hermes_python=args.hermes_python, quiet=bool(args.quiet))
    return

  # Numeric index
  try:
    idx = int(args.test)
    if 0 <= idx < len(TEST_CASES):
      run_test(TEST_CASES[idx], user_id=int(args.user_id), hermes_python=args.hermes_python, quiet=bool(args.quiet))
      return
  except ValueError:
    pass

  tc = next((t for t in TEST_CASES if t.get("name") == args.test), None)
  if not tc:
    raise SystemExit(f"Unknown test: {args.test!r}")
  run_test(tc, user_id=int(args.user_id), hermes_python=args.hermes_python, quiet=bool(args.quiet))


if __name__ == "__main__":
  main()

