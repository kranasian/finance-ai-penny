"""
RationalizeChange orchestration aligned with penny2 ``rationalize_change_engine``.

Uses an injected **StrategizerOptimizer**-like object (``generate_response``) instead of
``call_llm_with_run_config`` / DB templates. ``lookup_transactions`` is the host stub from
``penny.tool_funcs.lookup_transactions`` unless overridden when building the sandbox namespace.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from penny.tool_funcs.lookup_transactions import lookup_transactions as default_lookup_transactions

# Supplemental LLM generations 0 .. N-1; generation N uses terminal rationalize stub (no more nested LLM).
_MAX_FOLLOWUP_STRATEGIZER_GENERATIONS = 3

_GEN0_SUPPLEMENTAL_HINT = (
  "\n\n**HOST:** `# Supplemental lookup` is already attached—finalize with "
  "`execute_plan` that **only** `return True, \"…\"` (≤3 sentences). Do **not** call `rationalize` again.\n"
)

_FOLLOWUP_HOST_REMINDER = """
---
**HOST (required on this turn):** `# Supplemental lookup` is already in the message above. Your ```python``` `execute_plan` must **only** `return True, "…"` (≤3 sentences, explain the change vs forecast using Insight + both Top Transactions sections + supplemental lookup). **Do not** call `lookup_transactions`, `rationalize`, or any tool—only `from datetime import date` if needed (you should not need it here).
"""


def _user_message_with_supplemental_lookup(user_message: str, lookup_info: str) -> str:
  return (
    f"{user_message}\n\n"
    f"# Supplemental lookup (from lookup_transactions)\n\n"
    f"{lookup_info}\n"
  )


def _followup_llm_body(user_message: str, lookup_info: str, *, followup_generation: int) -> str:
  base = _user_message_with_supplemental_lookup(user_message, lookup_info)
  if followup_generation == 0:
    return base + _GEN0_SUPPLEMENTAL_HINT
  return base + _FOLLOWUP_HOST_REMINDER


def _rationalize_followup_terminal_stub(_user_message: str, _lookup_info: str) -> Tuple[bool, str]:
  return (
    False,
    "RationalizeChange: after supplemental lookup, execute_plan must only return True with your summary "
    "(≤3 sentences); do not call rationalize again.",
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


def _bind_lookup_transactions() -> Callable[..., str]:
  return default_lookup_transactions


class RationalizeChangeEngine:
  """Runs the RationalizeChange strategizer loop (initial + optional post-lookup follow-up)."""

  def __init__(
    self,
    optimizer: Any,
    *,
    l: Any = None,
    dl: Optional[Dict[str, Any]] = None,
    print_thought_summary: bool = False,
  ):
    self.optimizer = optimizer
    self.l = l
    self.dl = dl or {}
    self.print_thought_summary = print_thought_summary

  def _li_rationalize(self, phase_label: str, title: str, body: str, debug_arr: Optional[List[str]] = None) -> None:
    if self.l is None:
      return
    msg = f"RationalizeChange [{phase_label}] {title}:\n{body}"
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
    return self.optimizer.generate_response(
      print_thought_summary=self.print_thought_summary,
      prompt_override=prompt_override,
      system_prompt_override=None,
      **generate_kwargs,
    )

  def run(
    self,
    *,
    task_description: str,
    insight: str,
    top_transactions_recent_period: str,
    top_transactions_previous_period: str,
    recent_insight_date_range: str,
    previous_insight_date_range: str,
    previous_outcomes: Optional[Union[Dict[Union[int, str], str], List[str], str]] = None,
    debug_arr: Optional[List[str]] = None,
  ) -> Tuple[bool, str]:
    """Format the user turn via the optimizer’s normal fields, run strategizer + sandbox."""
    llm_out = self._call_llm(
      task_description=task_description,
      insight=insight,
      top_transactions_recent_period=top_transactions_recent_period,
      top_transactions_previous_period=top_transactions_previous_period,
      recent_insight_date_range=recent_insight_date_range,
      previous_insight_date_range=previous_insight_date_range,
      previous_outcomes=previous_outcomes,
      prompt_override=None,
    )
    generated = llm_out if isinstance(llm_out, str) else str(llm_out)
    code = _extract_code_from_response(generated)
    if not (code or "").strip() or "execute_plan" not in code:
      msg = "RationalizeChangeStrategizer: missing execute_plan in model output."
      self._li_rationalize("initial", "extracted code", (code or "").strip() or "(empty)", debug_arr)
      self._li_rationalize("initial", "execution result", "skipped — no valid execute_plan", debug_arr)
      self._lw(msg)
      return False, msg

    user_message = getattr(self.optimizer, "_last_formatted_user_message", "") or ""

    bound_lookup = _bind_lookup_transactions()
    rationalize_cb = self._make_rationalize_callback()
    exec_globals: Dict[str, Any] = {
      "USER_MESSAGE": user_message,
      "lookup_transactions": bound_lookup,
      "rationalize": rationalize_cb,
    }
    try:
      exec_result = _execute_plan_code(code, exec_globals, "execute_plan")
    except Exception as e:
      self._li_rationalize("initial", "execution result", f"sandbox raised: {e!s}", debug_arr)
      self._lw(f"RationalizeChangeStrategizer sandbox: {e!s}")
      return False, f"RationalizeChangeStrategizer sandbox: {e!s}"

    success, out = _process_plan_result(exec_result, [])
    out_str = out if isinstance(out, str) else str(out)
    return success, out_str

  def run_followup_turn(self, user_message: str, lookup_info: str, *, debug_arr: Optional[List[str]] = None) -> Tuple[bool, str]:
    """After ``lookup_transactions`` on the host: same strategizer template + supplemental body (no RationalizeMerge)."""
    return self._execute_strategizer_followup(user_message, lookup_info, followup_generation=0, debug_arr=debug_arr)

  def sandbox_rationalize_callback(self) -> Callable[[str, str], Tuple[bool, str]]:
    """``rationalize`` injected into the initial sandbox (nested follow-up generations)."""
    return self._make_rationalize_callback()

  def _execute_strategizer_followup(
    self,
    user_message: str,
    lookup_info: str,
    *,
    followup_generation: int = 0,
    debug_arr: Optional[List[str]] = None,
  ) -> Tuple[bool, str]:
    phase = "follow-up" if followup_generation == 0 else f"follow-up-retry-{followup_generation}"
    body = _followup_llm_body(user_message, lookup_info, followup_generation=followup_generation)
    try:
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
    except Exception as e:
      self._li_rationalize(phase, "LLM call error", f"{e!s}", debug_arr)
      self._li_rationalize(phase, "execution result", f"skipped — LLM call failed: {e!s}", debug_arr)
      self._lw(f"RationalizeChangeStrategizer follow-up: {e!s}")
      return False, f"RationalizeChangeStrategizer follow-up: {e!s}"
    generated = llm_out if isinstance(llm_out, str) else str(llm_out)
    code = _extract_code_from_response(generated)
    if not (code or "").strip() or "execute_plan" not in code:
      msg = "RationalizeChangeStrategizer follow-up: missing execute_plan in model output."
      self._li_rationalize(phase, "extracted code", (code or "").strip() or "(empty)", debug_arr)
      self._li_rationalize(phase, "execution result", "skipped — no valid execute_plan", debug_arr)
      self._lw(msg)
      return False, msg

    bound_lookup = _bind_lookup_transactions()
    if followup_generation + 1 < _MAX_FOLLOWUP_STRATEGIZER_GENERATIONS:
      rationalize_cb = self._make_followup_rationalize_recurse_callback(
        followup_generation=followup_generation + 1,
      )
    else:
      rationalize_cb = _rationalize_followup_terminal_stub
    exec_globals: Dict[str, Any] = {
      "USER_MESSAGE": user_message,
      "LOOKUP_INFO": lookup_info,
      "lookup_transactions": bound_lookup,
      "rationalize": rationalize_cb,
    }
    try:
      exec_result = _execute_plan_code(code, exec_globals, "execute_plan")
    except Exception as e:
      self._li_rationalize(phase, "execution result", f"sandbox raised: {e!s}", debug_arr)
      self._lw(f"RationalizeChangeStrategizer follow-up sandbox: {e!s}")
      return False, f"RationalizeChangeStrategizer follow-up sandbox: {e!s}"
    success, out = _process_plan_result(exec_result, [])
    out_str = out if isinstance(out, str) else str(out)
    return success, out_str

  def _make_followup_rationalize_recurse_callback(self, *, followup_generation: int):
    phase = "follow-up-retry-" + str(followup_generation) if followup_generation > 0 else "follow-up"

    def _rationalize(user_msg: str, lookup_payload: str) -> Tuple[bool, str]:
      self._li_rationalize(
        phase,
        "nested rationalize()",
        f"running strategizer generation {followup_generation}.",
        None,
      )
      return self._execute_strategizer_followup(
        user_msg,
        lookup_payload,
        followup_generation=followup_generation,
      )

    return _rationalize

  def _make_rationalize_callback(self):
    def _rationalize(input_info: str, lookup_info: str) -> Tuple[bool, str]:
      return self._execute_strategizer_followup(
        input_info,
        lookup_info,
        followup_generation=0,
      )

    return _rationalize
