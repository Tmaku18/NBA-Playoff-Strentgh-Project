"""As-of date semantics: filter to rows with game_date < as_of_date (strict t-1)."""
import pandas as pd


def filter_as_of(df: pd.DataFrame, as_of_date: str, date_col: str = "GAME_DATE") -> pd.DataFrame:
    """Return rows where date_col < as_of_date. as_of_date can be 'YYYY-MM-DD' or datetime-like."""
    d = pd.to_datetime(as_of_date).date() if isinstance(as_of_date, str) else as_of_date
    if hasattr(df[date_col].iloc[0], "date"):
        return df[df[date_col].dt.date < d].copy()
    return df[pd.to_datetime(df[date_col]).dt.date < d].copy()


def assert_no_future_leakage(df: pd.DataFrame, as_of_date: str, date_col: str = "GAME_DATE") -> None:
    """Raise if any row has date_col >= as_of_date."""
    d = pd.to_datetime(as_of_date).date() if isinstance(as_of_date, str) else as_of_date
    dates = pd.to_datetime(df[date_col]).dt.date
    bad = (dates >= d).sum()
    if bad > 0:
        raise ValueError(f"Future leakage: {bad} rows with {date_col} >= {as_of_date}")
