import pandas as pd

WEEKDAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

def order_days(days: list[str]) -> list[str]:
    observed = (
        pd.Series(days)
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not observed:
        return []

    canonical = [d for d in WEEKDAY_ORDER if d in observed]
    if canonical:
        return canonical

    abbrev_map = {}
    for v in observed:
        abbrev = v[:3].title()
        abbrev_map.setdefault(abbrev, v)

    abbrev_order = [d for d in WEEKDAY_ORDER if d in abbrev_map]
    if abbrev_order:
        return [abbrev_map[a] for a in abbrev_order]

    return observed