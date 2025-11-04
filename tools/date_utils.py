from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import calendar


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
    """Returns the start of the month for a given date."""
    return date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def get_end_of_month(date: datetime) -> datetime:
    """Returns the end of the month for a given date."""
    # Get the last day of the month
    last_day = calendar.monthrange(date.year, date.month)[1]
    return date.replace(day=last_day, hour=23, minute=59, second=59, microsecond=999999)


def get_start_of_year(date: datetime) -> datetime:
    """Returns the start of the year for a given date."""
    return date.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)


def get_end_of_year(date: datetime) -> datetime:
    """Returns the end of the year for a given date."""
    return date.replace(month=12, day=31, hour=23, minute=59, second=59, microsecond=999999)


def get_start_of_week(date: datetime) -> datetime:
    """Returns the start of the week (Monday) for a given date."""
    # Get Monday of the week (weekday() returns 0 for Monday, 6 for Sunday)
    days_since_monday = date.weekday()
    start_of_week = date - timedelta(days=days_since_monday)
    return start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)


def get_end_of_week(date: datetime) -> datetime:
    """Returns the end of the week (Sunday) for a given date."""
    # Get Sunday of the week (weekday() returns 0 for Monday, 6 for Sunday)
    days_until_sunday = 6 - date.weekday()
    end_of_week = date + timedelta(days=days_until_sunday)
    return end_of_week.replace(hour=23, minute=59, second=59, microsecond=999999)


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
    granularity = granularity.lower()
    
    if granularity == "daily":
        return date + timedelta(days=count)
    elif granularity == "weekly":
        return date + timedelta(weeks=count)
    elif granularity == "monthly":
        return date + relativedelta(months=count)
    elif granularity == "yearly":
        return date + relativedelta(years=count)
    else:
        raise ValueError(f"Invalid granularity: {granularity}. Must be one of: daily, weekly, monthly, yearly")

