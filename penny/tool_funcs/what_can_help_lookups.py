"""Stub lookup callables for WhatCanHelp strategizer (host replaces with DB-backed impl)."""

from __future__ import annotations

from datetime import date

from penny.tool_funcs.lookup_transactions import OFFICIAL_CATEGORY_VALUES, lookup_transactions


def lookup_accounts() -> str:
  """Return a human-readable summary of linked accounts (stub)."""
  return (
    "(lookup_accounts stub)\n"
    "No linked accounts in offline stub. Host should return institution, mask, type, balance snapshot."
  )


def lookup_spending_forecasts(*, horizon_months: int = 3) -> str:
  """Return planned or forecast spending totals by period (stub)."""
  return (
    f"(lookup_spending_forecasts stub: horizon_months={horizon_months})\n"
    "No forecast rows in offline stub."
  )


def lookup_income_forecasts(*, horizon_months: int = 3) -> str:
  """Return planned or forecast income by period (stub)."""
  return (
    f"(lookup_income_forecasts stub: horizon_months={horizon_months})\n"
    "No income forecast rows in offline stub."
  )


def lookup_spending_transactions(
  start: date,
  end: date,
  *,
  name_contains: str = "",
  in_category: list[str] | None = None,
  max_visible: int = 10,
) -> str:
  """Spending-only slice for a window (stub)."""
  if in_category:
    bad = [c for c in in_category if c not in OFFICIAL_CATEGORY_VALUES]
    if bad:
      raise ValueError(f"in_category contains non-official slugs: {bad}")
  return (
    f"(lookup_spending_transactions stub: {start.isoformat()}..{end.isoformat()} "
    f"name_contains={name_contains!r} in_category={in_category!r} max_visible={max_visible})\n"
    "No rows in offline stub."
  )


def lookup_income_transactions(
  start: date,
  end: date,
  *,
  name_contains: str = "",
  max_visible: int = 10,
) -> str:
  """Income-only slice for a window (stub)."""
  return (
    f"(lookup_income_transactions stub: {start.isoformat()}..{end.isoformat()} "
    f"name_contains={name_contains!r} max_visible={max_visible})\n"
    "No rows in offline stub."
  )


def lookup_monthly_spending_by_category(*, months_back: int = 6) -> str:
  """Aggregate actual spending by category per calendar month (stub)."""
  return (
    f"(lookup_monthly_spending_by_category stub: months_back={months_back})\n"
    "No aggregates in offline stub."
  )


def lookup_future_spending_by_category(*, months_ahead: int = 3) -> str:
  """Forecast or scheduled spending by category (stub)."""
  return (
    f"(lookup_future_spending_by_category stub: months_ahead={months_ahead})\n"
    "No future category rows in offline stub."
  )


def lookup_avg_monthly_spending(category: str) -> str:
  """Rolling average monthly spend for one official category slug (stub)."""
  if category not in OFFICIAL_CATEGORY_VALUES:
    raise ValueError(
      f"category {category!r} not in OFFICIAL_CATEGORY_VALUES; use an official slug."
    )
  return (
    f"(lookup_avg_monthly_spending stub: category={category!r})\n"
    "No average in offline stub."
  )


__all__ = [
  "lookup_accounts",
  "lookup_avg_monthly_spending",
  "lookup_future_spending_by_category",
  "lookup_income_forecasts",
  "lookup_income_transactions",
  "lookup_monthly_spending_by_category",
  "lookup_spending_forecasts",
  "lookup_spending_transactions",
  "lookup_transactions",
  "OFFICIAL_CATEGORY_VALUES",
]
