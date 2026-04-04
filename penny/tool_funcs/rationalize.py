import importlib.util
import os
from pathlib import Path

_experiment_module = None


def _experiment_rationalize_module():
  """Load `experiments/rationalize_change_strategizer_optimizer_v2.py` by path (folder is not a package)."""
  global _experiment_module
  if _experiment_module is not None:
    return _experiment_module
  root = Path(__file__).resolve().parents[2]
  path = root / "experiments" / "rationalize_change_strategizer_optimizer_v2.py"
  name = "rationalize_change_strategizer_optimizer_v2"
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load rationalize experiment module from {path}")
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  _experiment_module = mod
  return mod


def rationalize(input_info: str, lookup_info: str) -> tuple[bool, str]:
  """Combine the full user turn with `lookup_transactions` output into the final user-facing text.

  When ``GEMINI_API_KEY`` is set, runs the post-lookup strategizer turn from
  ``experiments/rationalize_change_strategizer_optimizer_v2.py`` (``generate_rationalization_text``).
  Otherwise returns lookup text only (offline / tests).
  """
  if not os.getenv("GEMINI_API_KEY"):
    _ = input_info
    body = lookup_info.strip() if lookup_info.strip() else "(no lookup data)"
    return True, body

  try:
    mod = _experiment_rationalize_module()
    success, text = mod.generate_rationalization_text(input_info, lookup_info)
    return success, text.strip()
  except Exception as e:
    return False, str(e)
