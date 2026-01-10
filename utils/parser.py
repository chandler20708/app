import io
from typing import Any, Dict, Tuple

import gurobipy as gp
import polars as pl

from models import SchedulingProblem

def _load_polars_frame(uploaded_file) -> pl.DataFrame:
    """
    Read CSV content into a Polars DataFrame from a file-like object, bytes, or path.
    """
    if hasattr(uploaded_file, "read"):
        raw = uploaded_file.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        return pl.read_csv(io.BytesIO(raw))
    if isinstance(uploaded_file, (bytes, bytearray)):
        return pl.read_csv(io.BytesIO(uploaded_file))
    # Fallback: assume path-like
    return pl.read_csv(uploaded_file)


def parse_csv_to_problem(uploaded_file) -> tuple[SchedulingProblem, str]:
    """
    Expect CSV columns (required): student, day, availability, wage_rate
    Optional flags per-row: bachelor, master, programming, troubleshooting (Y/N or 1/0)
    Returns (SchedulingProblem, key_str_for_cache)
    """
    df = _load_polars_frame(uploaded_file)
    if df.is_empty():
        raise ValueError("Uploaded CSV contains no rows.")

    # normalize column names for easier matching
    rename_map = {col: col.strip().lower() for col in df.columns}
    df = df.rename(rename_map)

    def trueish(val) -> bool:
        if val is None:
            return False
        v = str(val).strip().lower()
        return v in ("y", "yes", "1", "true", "t")

    # ensure required columns exist
    required = {"student", "day", "availability", "wage_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    # fill optional columns with defaults if absent
    for opt_col in ["bachelor", "master", "programming", "troubleshooting"]:
        if opt_col not in df.columns:
            df = df.with_columns(pl.lit(None).alias(opt_col))

    students = set()
    days = set()
    avail = {}
    wage_rates = {}
    bachelor = set()
    master = set()
    programming = set()
    troubleshooting = set()

    for row in df.iter_rows(named=True):
        s = row.get("student")
        d = row.get("day")
        if not s or not d:
            continue
        students.add(s)
        days.add(d)

        # availability
        try:
            a = float(row.get("availability") or 0)
        except (TypeError, ValueError):
            a = 0.0
        avail[(d, s)] = a

        # wage
        try:
            w = float(row.get("wage_rate") or 0)
            wage_rates[s] = w
        except (TypeError, ValueError):
            wage_rates[s] = wage_rates.get(s, 0.0)

        # optional flags
        if trueish(row.get("bachelor")):
            bachelor.add(s)
        if trueish(row.get("master")):
            master.add(s)
        if trueish(row.get("programming")):
            programming.add(s)
        if trueish(row.get("troubleshooting")):
            troubleshooting.add(s)

    if not students or not days:
        raise ValueError("Uploaded CSV must contain at least one student and one day with availability.")

    # Convert to canonical ordered lists
    students = sorted(students)
    days = sorted(days)

    # Convert avail dict to gurobipy.tupledict
    gp_avail = gp.tupledict({(d, s): avail.get((d, s), 0.0) for d in days for s in students})

    # Ensure wage_rates has entries for all students
    for s in students:
        wage_rates.setdefault(s, 0.0)

    problem = SchedulingProblem(
        students=students,
        days=days,
        availability=gp_avail,
        wage_rates=wage_rates,
        bachelor_students=set(bachelor),
        master_students=set(master),
        programming_ops=list(programming),
        troubleshooting_ops=list(troubleshooting),
        daily_coverage_req=14.0,
    )

    # Build a simple string key for caching
    key = f"{','.join(students)}|{','.join(days)}|{len(avail)}|{sum(wage_rates.values())}"
    return problem, key

def validate_csv_file(uploaded_file, daily_coverage_req: float = 14.0) -> Dict[str, Any]:
    # call parse_csv_to_problem (already implemented) and run checks
    uploaded_file.seek(0)
    try:
        problem, _ = parse_csv_to_problem(uploaded_file)
    except Exception as e:
        return {"ok": False, "errors": [str(e)], "warnings": [], "problem": None}

    errors, warnings = [], []
    # check numeric ranges
    neg_avails = [(d, s, problem.availability[d, s]) for d in problem.days for s in problem.students if problem.availability[d, s] < 0]
    if neg_avails:
        errors.append("Negative availability values found.")
    neg_wages = [s for s, w in problem.wage_rates.items() if w < 0]
    if neg_wages:
        errors.append("Negative wage rates found.")

    # quick feasibility: total availability per day
    for d in problem.days:
        total = sum(problem.availability[d, s] for s in problem.students)
        if total < daily_coverage_req:
            warnings.append(f"Day {d}: total availability {total} < required coverage {daily_coverage_req}")

    return {"ok": not errors, "errors": errors, "warnings": warnings, "problem": problem}
