from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import calendar
import pandas as pd


def get_today_date() -> datetime:
    """Returns today's date."""
    return datetime.now()


def get_date(year: int, month: int, day: int) -> datetime:
    """
    Returns a datetime object for the specified date.
    
    Args:
        year: The year (e.g., 2025)
        month: The month (1-12)
        day: The day of the month (1-31)
    
    Returns:
        A datetime object for the specified date at 00:00:00
    """
    return datetime(year, month, day)


def get_start_of_month(date: datetime) -> datetime:
    """Returns the first day of the month for a given date."""
    date_obj = pd.to_datetime(date)
    start = date_obj.replace(day=1)
    return datetime.combine(start.date(), datetime.min.time())


def get_end_of_month(date: datetime) -> datetime:
    """Returns the last day of the month for a given date."""
    date_obj = pd.to_datetime(date)
    end = date_obj.replace(day=1) + pd.offsets.MonthEnd()
    return datetime.combine(end.date(), datetime.min.time())


def get_start_of_year(date: datetime) -> datetime:
    """Returns January 1st for the year of the given date."""
    date_obj = pd.to_datetime(date)
    return datetime(date_obj.year, 1, 1)


def get_end_of_year(date: datetime) -> datetime:
    """Returns December 31st for the year of the given date."""
    date_obj = pd.to_datetime(date)
    return datetime(date_obj.year, 12, 31)


def get_start_of_week(date: datetime) -> datetime:
    """Returns the Sunday of the week for the given date."""
    date_obj = pd.to_datetime(date)
    start = (date_obj + pd.DateOffset(days=1) - pd.offsets.Week(weekday=6))
    return datetime.combine(start.date(), datetime.min.time())


def get_end_of_week(date: datetime) -> datetime:
    """Returns the Saturday of the week for the given date."""
    date_obj = pd.to_datetime(date)
    end = (date_obj + pd.offsets.Week(weekday=6) - pd.DateOffset(days=1))
    return datetime.combine(end.date(), datetime.min.time())


def get_after_periods(date: datetime, count: int, granularity: str) -> datetime:
    """
    Adds periods and returns date.
    
    Args:
        date: The starting date
        count: Number of periods to add (can be negative to subtract)
        granularity: One of "daily", "weekly", "monthly", "yearly"
    
    Returns:
        The date after adding the specified periods
    """
    date_obj = pd.to_datetime(date)
    g = (granularity or "").lower()
    
    if g == "daily" or g == "day":
        result = date_obj + pd.DateOffset(days=count)
    elif g == "weekly" or g == "week":
        result = date_obj + pd.DateOffset(weeks=count)
    elif g == "monthly" or g == "month":
        result = date_obj + pd.DateOffset(months=count)
    elif g == "yearly" or g == "year":
        result = date_obj + pd.DateOffset(years=count)
    else:
        # Fallback: return original date when granularity not recognized
        result = date_obj
    
    ts = pd.to_datetime(result).normalize()
    return datetime.combine(ts.date(), datetime.min.time())


def get_date_string(date: datetime) -> str:
    """Return date in "YYYY-MM-DD" format."""
    return date.strftime("%Y-%m-%d")

