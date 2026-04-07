"""
WhatCanHelp orchestration: same loop shape as ``rationalize_change_engine`` (penny1): injected optimizer
``generate_response``, **no** DB template lookup in this repo.

**Iteration 1:** Optimizer’s fixed ``system_prompt`` + structured user turn → sandbox
``execute_plan`` with lookup tools and ``refine_strategy``.

**Iteration 2+:** ``refine_strategy(aggregated)`` passes concatenated tool strings; the host places that text in
``# Latest Outcome`` on the next turn (optional ``# Previous Outcomes`` for numbered prior strategizer outcomes).
``_call_llm`` always passes ``system_prompt_override=None``.
Further ``refine_strategy`` generations are capped; retries append ``_FOLLOWUP_HOST_REMINDER``; the last
generation uses a ``refine_strategy`` stub that returns ``(False, …)``.

Structured turn: ``# Latest Outcome`` (prefilled data and/or tool aggregates; token ``None`` when not loaded yet),
then optionally ``# Previous Outcomes`` (numbered lines). Pass ``None`` or ``""`` for ``latest_outcome`` when empty.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from penny.tool_funcs import what_can_help_lookups as default_lookups

_MAX_FOLLOWUP_STRATEGIZER_GENERATIONS = 3

_GEN0_REFINE_HINT = (
  "\n\n**HOST:** `# Latest Outcome` above is **aggregated tool output** from the prior `execute_plan`. "
  "On this turn, `execute_plan` must **only** `return True, \"…\"` (concise strategy summary). "
  "Do **not** call lookup tools or `refine_strategy` again.\n"
)

_FOLLOWUP_HOST_REMINDER = """
---
**HOST (required on this turn):** `# Latest Outcome` contains data to finalize your strategy. Your ```python``` `execute_plan` must **only** `return True, "…"` with urgency, goals, and tactics. **Do not** call lookup tools or `refine_strategy`.
"""

PreviousOutcomesArg = Optional[Union[Dict[Union[int, str], str], List[str], str]]


def format_previous_outcomes_block(previous_outcomes: PreviousOutcomesArg) -> str:
  """Format prior WhatCanHelp / host outcomes like ``rationalize_change`` (numbered lines)."""
  if previous_outcomes is None:
    return ""
  if isinstance(previous_outcomes, str):
    return f"1. **Outcome #1**: {previous_outcomes}"
  if isinstance(previous_outcomes, dict):
    numbered_rows = []
    for key in sorted(previous_outcomes, key=lambda x: int(x) if str(x).isdigit() else str(x)):
      value = previous_outcomes[key]
      if isinstance(value, str) and value.strip():
        label_num = int(key) if str(key).isdigit() else key
        numbered_rows.append(f"{label_num}. **Outcome #{label_num}**: {value.strip()}")
    if numbered_rows:
      return "\n".join(numbered_rows)
    return ""
  clean_outcomes = [item.strip() for item in previous_outcomes if isinstance(item, str) and item.strip()]
  if not clean_outcomes:
    return ""
  return "\n".join(
    f"{idx}. **Outcome #{idx}**: {outcome}" for idx, outcome in enumerate(clean_outcomes, start=1)
  )


def latest_outcome_payload(latest_outcome: str | None) -> str:
  """Normalized latest-outcome fragment for the user turn (``None`` → empty)."""
  if latest_outcome is None:
    return ""
  return latest_outcome.strip()


_LATEST_OUTCOME_PREFIX = "# Latest Outcome\n\n"


def _wch_latest_outcome_section(structured_turn: str) -> str:
  """Body under ``# Latest Outcome`` through (but not including) ``# Previous Outcomes``."""
  s = structured_turn.strip()
  if s.startswith(_LATEST_OUTCOME_PREFIX):
    s = s[len(_LATEST_OUTCOME_PREFIX) :]
  elif "\n# Latest Outcome\n\n" in s:
    s = s.split("\n# Latest Outcome\n\n", 1)[1]
  else:
    return ""
  if "\n# Previous Outcomes\n\n" in s:
    s = s.split("\n# Previous Outcomes\n\n", 1)[0]
  return s.strip()


def wch_latest_outcome_is_empty(structured_turn: str) -> bool:
  """True iff the ``# Latest Outcome`` body is exactly the token ``None`` (no data / tools yet)."""
  return _wch_latest_outcome_section(structured_turn) == "None"


def wch_lookup_is_empty(structured_turn: str) -> bool:
  """Deprecated alias for :func:`wch_latest_outcome_is_empty`."""
  return wch_latest_outcome_is_empty(structured_turn)


def format_lookup_user_turn(
  latest_outcome: str | None,
  *,
  previous_outcomes: PreviousOutcomesArg = None,
) -> str:
  """Structured message: ``# Latest Outcome`` (empty → token ``None``), optional ``# Previous Outcomes``."""
  payload = latest_outcome_payload(latest_outcome)
  latest_body = payload if payload else "None"
  parts = [
    _LATEST_OUTCOME_PREFIX,
    f"{latest_body}\n",
  ]
  prev_block = format_previous_outcomes_block(previous_outcomes)
  if prev_block.strip():
    parts.extend(["\n# Previous Outcomes\n\n", prev_block, "\n"])
  return "".join(parts)


def _followup_llm_body(
  aggregated_latest_outcome: str,
  *,
  followup_generation: int,
  previous_outcomes: PreviousOutcomesArg = None,
) -> tuple[str, str]:
  """Return ``(structured_turn_for_sandbox, full_prompt_for_llm)``."""
  agg = (aggregated_latest_outcome or "").strip()
  structured = format_lookup_user_turn(agg if agg else None, previous_outcomes=previous_outcomes)
  full = structured.rstrip()
  if followup_generation == 0:
    full = full + _GEN0_REFINE_HINT
  else:
    full = full + _FOLLOWUP_HOST_REMINDER
  return structured, full


def _refine_strategy_followup_terminal_stub(_aggregated: str) -> Tuple[bool, str]:
  return (
    False,
    "WhatCanHelp: after lookup aggregation, execute_plan must only return True with your summary; "
    "do not call refine_strategy again.",
  )


def _extract_code_from_response(text: str) -> str:
  code_start = text.find("```python")
  if code_start != -1:
    code_start += len("```python")
    code_end = text.find("```", code_start)
    if code_end != -1:
      return text[code_start:code_end].strip()
    return text[code_start:].strip()
  return text.strip()


def _process_plan_result(exec_result: Dict[str, Any], function_calls: List[Dict[str, Any]]) -> Tuple[bool, str]:
  if isinstance(exec_result, dict) and "_function_result" in exec_result:
    result = exec_result["_function_result"]
    if isinstance(result, tuple) and len(result) == 2:
      success, output = result
      if isinstance(output, dict):
        try:
          output = json.dumps(output, indent=2)
        except (TypeError, ValueError):
          output = str(output)
      return success, output
  combined_output_info = ""
  if function_calls:
    for call_info in function_calls:
      function_name = call_info["function_name"]
      arguments = call_info["arguments"]
      combined_output_info += f"{function_name}:\n"
      internal_args = {"kwargs", "l", "dl", "debug_arr", "db_mconn", "db_sconn", "templates_df", "user_id", "client_"}
      for arg_name, arg_value in arguments.items():
        if arg_name not in internal_args:
          combined_output_info += f"  {arg_name}: {arg_value}\n"
      combined_output_info += "\n"
  if combined_output_info:
    return True, combined_output_info.strip()
  return False, "Could not execute plan or retrieve result"


def _execute_plan_code(code_str: str, exec_globals: Dict[str, Any], function_name: str = "execute_plan") -> Dict[str, Any]:
  g = dict(exec_globals)
  compiled = compile(code_str, f"<{function_name}>", "exec")
  exec(compiled, g)
  if function_name in g and callable(g[function_name]):
    g["_function_result"] = g[function_name]()
  return g


def _default_tool_bindings() -> Dict[str, Any]:
  return {
    "lookup_accounts": default_lookups.lookup_accounts,
    "lookup_spending_forecasts": default_lookups.lookup_spending_forecasts,
    "lookup_income_forecasts": default_lookups.lookup_income_forecasts,
    "lookup_transactions": default_lookups.lookup_transactions,
    "lookup_spending_transactions": default_lookups.lookup_spending_transactions,
    "lookup_income_transactions": default_lookups.lookup_income_transactions,
    "lookup_monthly_spending_by_category": default_lookups.lookup_monthly_spending_by_category,
    "lookup_future_spending_by_category": default_lookups.lookup_future_spending_by_category,
    "lookup_avg_monthly_spending": default_lookups.lookup_avg_monthly_spending,
  }


class WhatCanHelpEngine:
  """Runs the WhatCanHelp strategizer loop (``# Latest Outcome`` / ``# Previous Outcomes`` turns + refine_strategy)."""

  def __init__(
    self,
    optimizer: Any,
    *,
    tool_bindings: Optional[Dict[str, Any]] = None,
    l: Any = None,
    dl: Optional[Dict[str, Any]] = None,
    print_thought_summary: bool = False,
  ):
    self.optimizer = optimizer
    self.tool_bindings = tool_bindings if tool_bindings is not None else _default_tool_bindings()
    self.l = l
    self.dl = dl or {}
    self.print_thought_summary = print_thought_summary

  def _log_what_can_help(self, phase_label: str, title: str, body: str, debug_arr: Optional[List[str]] = None) -> None:
    if self.l is None:
      return
    msg = f"WhatCanHelp [{phase_label}] {title}:\n{body}"
    log_fn = getattr(self.l, "info", None)
    if callable(log_fn):
      log_fn(msg)

  def _lw(self, msg: str) -> None:
    if self.l is None:
      return
    log_fn = getattr(self.l, "warning", None)
    if callable(log_fn):
      log_fn(msg)

  def _call_llm(self, *, prompt_override: str | None = None, **generate_kwargs: Any) -> str:
    """Always ``system_prompt_override=None`` — initial and follow-up turns share the optimizer’s system instruction."""
    return self.optimizer.generate_response(
      print_thought_summary=self.print_thought_summary,
      prompt_override=prompt_override,
      system_prompt_override=None,
      **generate_kwargs,
    )

  def run(
    self,
    *,
    latest_outcome: str | None = None,
    previous_outcomes: PreviousOutcomesArg = None,
    debug_arr: Optional[List[str]] = None,
  ) -> Tuple[bool, str]:
    """First host call: ``latest_outcome`` unset → token ``None`` under ``# Latest Outcome``."""
    body = format_lookup_user_turn(latest_outcome, previous_outcomes=previous_outcomes)
    setattr(self.optimizer, "_last_lookup_user_turn", body)
    llm_out = self._call_llm(
      task_description="",
      insight="",
      top_transactions_recent_period="",
      top_transactions_previous_period="",
      recent_insight_date_range="—",
      previous_insight_date_range="—",
      previous_outcomes=None,
      prompt_override=body,
    )
    generated = llm_out if isinstance(llm_out, str) else str(llm_out)
    code = _extract_code_from_response(generated)
    if not (code or "").strip() or "execute_plan" not in code:
      msg = "WhatCanHelpStrategizer: missing execute_plan in model output."
      self._log_what_can_help("initial", "extracted code", (code or "").strip() or "(empty)", debug_arr)
      self._log_what_can_help("initial", "execution result", "skipped — no valid execute_plan", debug_arr)
      self._lw(msg)
      return False, msg

    refine_cb = self._make_refinement_callback(previous_outcomes=previous_outcomes)
    exec_globals: Dict[str, Any] = dict(self.tool_bindings)
    exec_globals.update(
      {
        "WCH_USER_TURN": body,
        "wch_latest_outcome_is_empty": wch_latest_outcome_is_empty,
        "wch_lookup_is_empty": wch_lookup_is_empty,
        "refine_strategy": refine_cb,
      }
    )
    try:
      exec_result = _execute_plan_code(code, exec_globals, "execute_plan")
    except Exception as e:
      self._log_what_can_help("initial", "execution result", f"sandbox raised: {e!s}", debug_arr)
      self._lw(f"WhatCanHelpStrategizer sandbox: {e!s}")
      return False, f"WhatCanHelpStrategizer sandbox: {e!s}"

    success, out = _process_plan_result(exec_result, [])
    out_str = out if isinstance(out, str) else str(out)
    return success, out_str

  def run_refinement_turn(
    self,
    aggregated_latest_outcome: str,
    *,
    previous_outcomes: PreviousOutcomesArg = None,
    debug_arr: Optional[List[str]] = None,
  ) -> Tuple[bool, str]:
    """Host convenience: ``refine_strategy`` follow-up; pass same ``previous_outcomes`` as ``run`` when applicable."""
    return self._execute_strategizer_followup(
      aggregated_latest_outcome,
      followup_generation=0,
      previous_outcomes=previous_outcomes,
      debug_arr=debug_arr,
    )

  def sandbox_refinement_callback(
    self,
    *,
    previous_outcomes: PreviousOutcomesArg = None,
  ) -> Callable[[str], Tuple[bool, str]]:
    """``refine_strategy(aggregated)`` for tests/sandbox; threads ``previous_outcomes`` into follow-ups."""
    return self._make_refinement_callback(previous_outcomes=previous_outcomes)

  def _execute_strategizer_followup(
    self,
    aggregated_latest_outcome: str,
    *,
    followup_generation: int = 0,
    previous_outcomes: PreviousOutcomesArg = None,
    debug_arr: Optional[List[str]] = None,
  ) -> Tuple[bool, str]:
    phase = "follow-up" if followup_generation == 0 else f"follow-up-retry-{followup_generation}"
    structured_turn, prompt_for_llm = _followup_llm_body(
      aggregated_latest_outcome,
      followup_generation=followup_generation,
      previous_outcomes=previous_outcomes,
    )
    setattr(self.optimizer, "_last_lookup_user_turn", prompt_for_llm)
    try:
      llm_out = self._call_llm(
        task_description="",
        insight="",
        top_transactions_recent_period="",
        top_transactions_previous_period="",
        recent_insight_date_range="—",
        previous_insight_date_range="—",
        previous_outcomes=None,
        prompt_override=prompt_for_llm,
      )
    except Exception as e:
      self._log_what_can_help(phase, "LLM call error", f"{e!s}", debug_arr)
      self._log_what_can_help(phase, "execution result", f"skipped — LLM call failed: {e!s}", debug_arr)
      self._lw(f"WhatCanHelpStrategizer follow-up: {e!s}")
      return False, f"WhatCanHelpStrategizer follow-up: {e!s}"
    generated = llm_out if isinstance(llm_out, str) else str(llm_out)
    code = _extract_code_from_response(generated)
    if not (code or "").strip() or "execute_plan" not in code:
      msg = "WhatCanHelpStrategizer follow-up: missing execute_plan in model output."
      self._log_what_can_help(phase, "extracted code", (code or "").strip() or "(empty)", debug_arr)
      self._log_what_can_help(phase, "execution result", "skipped — no valid execute_plan", debug_arr)
      self._lw(msg)
      return False, msg

    if followup_generation + 1 < _MAX_FOLLOWUP_STRATEGIZER_GENERATIONS:
      refine_cb = self._make_followup_refinement_recurse_callback(
        followup_generation=followup_generation + 1,
        previous_outcomes=previous_outcomes,
      )
    else:
      refine_cb = _refine_strategy_followup_terminal_stub

    exec_globals: Dict[str, Any] = dict(self.tool_bindings)
    exec_globals.update(
      {
        "WCH_USER_TURN": structured_turn,
        "wch_latest_outcome_is_empty": wch_latest_outcome_is_empty,
        "wch_lookup_is_empty": wch_lookup_is_empty,
        "refine_strategy": refine_cb,
      }
    )
    try:
      exec_result = _execute_plan_code(code, exec_globals, "execute_plan")
    except Exception as e:
      self._log_what_can_help(phase, "execution result", f"sandbox raised: {e!s}", debug_arr)
      self._lw(f"WhatCanHelpStrategizer follow-up sandbox: {e!s}")
      return False, f"WhatCanHelpStrategizer follow-up sandbox: {e!s}"
    success, out = _process_plan_result(exec_result, [])
    out_str = out if isinstance(out, str) else str(out)
    return success, out_str

  def _make_followup_refinement_recurse_callback(
    self,
    *,
    followup_generation: int,
    previous_outcomes: PreviousOutcomesArg,
  ):
    phase = "follow-up-retry-" + str(followup_generation) if followup_generation > 0 else "follow-up"

    def _refine_strategy(aggregated_lookup: str) -> Tuple[bool, str]:
      self._log_what_can_help(
        phase,
        "nested refine_strategy()",
        f"running strategizer generation {followup_generation}.",
        None,
      )
      return self._execute_strategizer_followup(
        aggregated_lookup,
        followup_generation=followup_generation,
        previous_outcomes=previous_outcomes,
      )

    return _refine_strategy

  def _make_refinement_callback(
    self,
    *,
    previous_outcomes: PreviousOutcomesArg,
  ):
    def _refine_strategy(aggregated_lookup: str) -> Tuple[bool, str]:
      return self._execute_strategizer_followup(
        aggregated_lookup,
        followup_generation=0,
        previous_outcomes=previous_outcomes,
      )

    return _refine_strategy
