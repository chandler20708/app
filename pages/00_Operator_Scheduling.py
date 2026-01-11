import io
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

import gurobipy as gp
import pandas as pd
import streamlit as st

from models import (
    AllPairsFairness,
    AllPairsAvailabilityFairness,
    MinimizeCostPolicy,
    NoFairness,
    MinimizeUnfairnessPolicy,
    SchedulerService,
    SchedulingProblem,
)
from utils import (
    add_title,
    validate_csv_file,
    log_action,
    seed_everything,
    build_workload_chart,
    _ordered_days,
)

# --- Constants ---

DEFAULT_CSV_PATH = "data/default_params.csv"
COLOR_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
DAY_COLORS = [
    "#0b84a5",
    "#f6c85f",
    "#6f4e7c",
    "#9dd866",
    "#d65f5f",
    "#4cc3d9",
    "#ffa600",
]

DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@dataclass
class ProblemContext:
    problem: Optional[SchedulingProblem]
    warnings: List[str]
    source: str


@dataclass
class ProblemMeta:
    students: List[str]
    days: List[str]
    wage_rates: Dict[str, float]
    programming_ops: List[str]
    troubleshooting_ops: List[str]


@dataclass
class SolutionView:
    id: str
    status: str
    total_cost: float
    weekly_hours: Dict[str, float]
    cost_breakdown: pd.DataFrame
    schedule: pd.DataFrame
    coverage_by_day: Dict[str, float]
    color_map: Dict[str, str]
    day_color_map: Dict[str, str]
    day_order: List[str]
    baseline_cost: Optional[float]
    objective_label: str
    cost_limit: Optional[float]
    problem: ProblemMeta


# --- Caching Helpers ---


@st.cache_resource
def get_scheduler_service() -> SchedulerService:
    return SchedulerService()


@st.cache_data
def _load_default_csv_bytes() -> bytes:
    with open(DEFAULT_CSV_PATH, "rb") as f:
        return f.read()


@st.cache_data(show_spinner=False)
def _baseline_cost_cached(
    students: tuple,
    days: tuple,
    availability_items: tuple,
    wage_items: tuple,
    bachelor_students: tuple,
    master_students: tuple,
) -> float:
    """
    Baseline: task-1 cost-only solve using the uploaded CSV values (no skills/fairness tweaks).
    """
    gp_avail = gp.tupledict({(d, s): v for (d, s, v) in availability_items})
    wage_rates = dict(wage_items)
    problem = SchedulingProblem(
        students=list(students),
        days=list(days),
        availability=gp_avail,
        wage_rates=wage_rates,
        bachelor_students=set(bachelor_students),
        master_students=set(master_students),
        programming_ops=list(),
        troubleshooting_ops=list(),
        daily_coverage_req=14.0,
    )
    service = get_scheduler_service()
    sol = service.run(problem, NoFairness(), MinimizeCostPolicy())
    return sol.total_cost


# --- Problem Loading & Validation ---


def _problem_from_bytes(data: bytes, source: str) -> ProblemContext:
    file_like = io.BytesIO(data)
    result = validate_csv_file(file_like)
    if not result["ok"]:
        st.error(result["errors"])
        return ProblemContext(None, result.get("warnings", []), source)
    if result["warnings"]:
        for warn in result["warnings"]:
            st.warning(warn)
    return ProblemContext(result["problem"], result.get("warnings", []), source)


def resolve_problem(uploaded_bytes: bytes | None) -> ProblemContext:
    if uploaded_bytes is not None:
        return _problem_from_bytes(uploaded_bytes, "Uploaded CSV")
    try:
        return _problem_from_bytes(_load_default_csv_bytes(), "Default dataset")
    except FileNotFoundError:
        st.error("Default CSV not found. Please upload a dataset to continue.")
        return ProblemContext(None, None, [], "Missing default dataset")


def baseline_cost(problem: Optional[SchedulingProblem]) -> Optional[float]:
    if not problem:
        return None
    try:
        availability_items = tuple(
            (d, s, float(problem.availability[d, s]))
            for d in problem.days
            for s in problem.students
        )
        wage_items = tuple(problem.wage_rates.items())
        return _baseline_cost_cached(
            tuple(problem.students),
            tuple(problem.days),
            availability_items,
            wage_items,
            tuple(sorted(problem.bachelor_students)),
            tuple(sorted(problem.master_students)),
        )
    except Exception:
        st.info("Baseline cost unavailable (solver failed on baseline scenario).")
        return None


# --- Sidebar & Controls ---


def _last_solution_status() -> Optional[str]:
    last_solution = st.session_state.get("last_solution")
    if isinstance(last_solution, SolutionView):
        return last_solution.status
    if isinstance(last_solution, dict):
        return last_solution.get("status")
    return None


def render_sidebar_and_problem() -> tuple[ProblemContext, Dict[str, object], Optional[float]]:
    with st.sidebar:
        st.markdown("### Operator Scheduling ⚙️")
        uploaded_file = st.file_uploader(
            "Upload students availability/wage CSV",
            type=["csv"],
            help="CSV with columns: student, day, availability, wage_rate, optional: bachelor, master, programming, troubleshooting",
            key="upload_csv",
        )
        template_bytes = None
        try:
            template_bytes = _load_default_csv_bytes()
        except FileNotFoundError:
            pass
        if template_bytes:
            st.download_button(
                "Download CSV template",
                data=template_bytes,
                file_name="students_schedule_template.csv",
                mime="text/csv",
                width='stretch',
            )

    if uploaded_file is not None:
        uploaded_file.seek(0)
        st.session_state["uploaded_csv_bytes"] = uploaded_file.read()
    problem_ctx = resolve_problem(st.session_state.get("uploaded_csv_bytes"))
    baseline = baseline_cost(problem_ctx.problem)

    with st.sidebar:
        fairness_mode = st.selectbox(
            "Objective mode",
            ("Min cost", "Fairness (cost-bounded)", "Fairness only (ignore cost)"),
            index=0,
        )
        consider_skills = st.toggle("Consider skills constraints", value=False)

        allowable_pct = None
        if fairness_mode == "Fairness (cost-bounded)":
            allowable_pct = st.slider(
                "Allowable cost increase (%)",
                min_value=0.0,
                max_value=30.0,
                value=5.0,
                step=0.1,
            )

        st.caption(problem_ctx.source or "No dataset detected")
        if baseline is not None:
            st.metric("Baseline min cost", f"£{baseline:,.2f}")
        else:
            st.caption("Baseline min cost unavailable.")

        status = _last_solution_status()
        if status:
            st.caption(f"Optimisation Status: {status}")

        run_clicked = st.button("Run optimisation", width="stretch")
        st.markdown("---")
        use_availability_view = st.toggle(
            "Use availability-normalised fairness view",
            value=False,
            key="fairness_view_toggle",
        )

    controls = {
        "allowable_pct": allowable_pct,
        "fairness_mode": fairness_mode,
        "use_availability": use_availability_view,
        "consider_skills": consider_skills,
        "run_clicked": run_clicked,
    }

    return problem_ctx, controls, baseline


# --- Solver Utilities ---


def _apply_skill_preferences(problem: SchedulingProblem, consider_skills: bool) -> SchedulingProblem:
    if not consider_skills:
        return SchedulingProblem(
            students=problem.students,
            days=problem.days,
            availability=problem.availability,
            wage_rates=problem.wage_rates,
            bachelor_students=problem.bachelor_students,
            master_students=problem.master_students,
            programming_ops=[],
            troubleshooting_ops=[],
            daily_coverage_req=problem.daily_coverage_req,
        )

    if problem.programming_ops and problem.troubleshooting_ops:
        return problem

    students = list(problem.students)
    programming_ops = problem.programming_ops or students[:2]
    troubleshooting_ops = problem.troubleshooting_ops or (
        students[-2:] if len(students) >= 2 else students
    )

    return SchedulingProblem(
        students=problem.students,
        days=problem.days,
        availability=problem.availability,
        wage_rates=problem.wage_rates,
        bachelor_students=problem.bachelor_students,
        master_students=problem.master_students,
        programming_ops=programming_ops,
        troubleshooting_ops=troubleshooting_ops,
        daily_coverage_req=problem.daily_coverage_req,
    )


def _choose_policies(
    fairness_mode: str,
    allowable_pct: Optional[float],
    baseline: Optional[float],
    fairness_strategy,
):
    if fairness_mode == "Min cost":
        fairness = NoFairness()
        objective = MinimizeCostPolicy()
        cost_limit = None
        objective_label = "Minimise cost"
    elif fairness_mode == "Fairness (cost-bounded)":
        fairness = fairness_strategy
        cost_limit = (
            baseline * (1 + allowable_pct / 100.0)
            if baseline is not None and allowable_pct is not None
            else None
        )
        objective = MinimizeUnfairnessPolicy(cost_limit=cost_limit)
        objective_label = "Fairness with cost bound"
    else:  # Fairness only
        fairness = fairness_strategy
        cost_limit = None
        objective = MinimizeUnfairnessPolicy(cost_limit=None)
        objective_label = "Fairness only (ignore cost)"
    return fairness, objective, objective_label, cost_limit


def _assign_colors(students: List[str]) -> Dict[str, str]:
    return {
        student: COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        for idx, student in enumerate(students)
    }


def _assign_day_colors(days: List[str]) -> Dict[str, str]:
    ordered = _ordered_days(days)
    return {day: DAY_COLORS[idx % len(DAY_COLORS)] for idx, day in enumerate(ordered)}


def build_solution_view(
    sol,
    baseline: Optional[float],
    objective_label: str,
    cost_limit: Optional[float],
) -> SolutionView:
    cost_rows = []
    schedule_rows = []
    coverage_by_day: Dict[str, float] = {d: 0.0 for d in sol.problem.days}
    skill_users = set(sol.problem.programming_ops) | set(sol.problem.troubleshooting_ops)

    for student in sol.problem.students:
        hours = float(sol.weekly_hours.get(student, 0.0))
        wage = float(sol.problem.wage_rates.get(student, 0.0))
        cost_rows.append(
            {
                "student": student,
                "hours": round(hours, 2),
                "wage_rate": wage,
                "cost": round(hours * wage, 2),
            }
        )

    for day in sol.problem.days:
        for student in sol.problem.students:
            hours = float(sol.H_vars[day, student].X)
            coverage_by_day[day] += hours
            avail = float(sol.problem.availability[day, student])
            schedule_rows.append(
                {
                    "day": day,
                    "student": student,
                    "hours": round(hours, 2),
                    "availability": avail,
                    "skill_user": student in skill_users,
                }
            )

    return SolutionView(
        id=uuid.uuid4().hex,
        status="OPTIMAL",
        total_cost=float(sol.total_cost),
        weekly_hours={k: float(v) for k, v in sol.weekly_hours.items()},
        cost_breakdown=pd.DataFrame(cost_rows),
        schedule=pd.DataFrame(schedule_rows),
        coverage_by_day=coverage_by_day,
        color_map=_assign_colors(list(sol.problem.students)),
        day_color_map=_assign_day_colors(list(sol.problem.days)),
        day_order=_ordered_days(DAY_ORDER),
        baseline_cost=baseline,
        objective_label=objective_label,
        cost_limit=cost_limit,
        problem=ProblemMeta(
            students=list(sol.problem.students),
            days=list(sol.problem.days),
            wage_rates=dict(sol.problem.wage_rates),
            programming_ops=list(sol.problem.programming_ops),
            troubleshooting_ops=list(sol.problem.troubleshooting_ops),
        ),
    )


def run_scheduler(
    problem: SchedulingProblem,
    controls: Dict[str, object],
    baseline: Optional[float],
    fairness_strategy=None,
) -> SolutionView:
    if fairness_strategy is None:
        fairness_strategy = AllPairsFairness()
    adjusted_problem = _apply_skill_preferences(problem, controls["consider_skills"])
    fairness, objective, objective_label, cost_limit = _choose_policies(
        controls["fairness_mode"],
        controls["allowable_pct"],
        baseline,
        fairness_strategy,
    )
    service = get_scheduler_service()
    sol = service.run(adjusted_problem, fairness, objective)
    return build_solution_view(sol, baseline, objective_label, cost_limit)


def _looks_like_solution_view(obj) -> bool:
    required = (
        "total_cost",
        "weekly_hours",
        "cost_breakdown",
        "schedule",
        "day_color_map",
        "day_order",
        "problem",
    )
    return all(hasattr(obj, name) for name in required)


def _coerce_solution_view(last_solution) -> Optional[SolutionView]:
    if isinstance(last_solution, SolutionView):
        return last_solution
    if _looks_like_solution_view(last_solution):
        return last_solution
    if not isinstance(last_solution, dict) or "total_cost" not in last_solution:
        return None

    # Allow legacy dict payloads in session_state to keep existing sessions working.
    problem = last_solution.get("problem", {})
    return SolutionView(
        id=last_solution.get("id", uuid.uuid4().hex),
        status=last_solution.get("status", ""),
        total_cost=float(last_solution.get("total_cost", 0.0)),
        weekly_hours={
            k: float(v) for k, v in last_solution.get("weekly_hours", {}).items()
        },
        cost_breakdown=pd.DataFrame(last_solution.get("cost_breakdown", [])),
        schedule=pd.DataFrame(last_solution.get("schedule", [])),
        coverage_by_day=last_solution.get("coverage_by_day", {}),
        color_map=last_solution.get("color_map", {}),
        day_color_map=last_solution.get("day_color_map", {}),
        day_order=last_solution.get("day_order", []),
        baseline_cost=last_solution.get("baseline_cost"),
        objective_label=last_solution.get("objective_label", ""),
        cost_limit=last_solution.get("cost_limit"),
        problem=ProblemMeta(
            students=problem.get("students", []),
            days=problem.get("days", []),
            wage_rates=problem.get("wage_rates", {}),
            programming_ops=problem.get("programming_ops", []),
            troubleshooting_ops=problem.get("troubleshooting_ops", []),
        ),
    )


def maybe_run_optimisation(
    problem: SchedulingProblem, controls: Dict[str, object], baseline: Optional[float]
) -> None:
    auto_run = not st.session_state.get("last_solution")
    if not controls["run_clicked"] and not auto_run:
        return

    try:
        solution_view = run_scheduler(
            problem,
            controls,
            baseline,
            fairness_strategy=AllPairsFairness(),
        )
        if controls["fairness_mode"] == "Min cost":
            availability_view = solution_view
        else:
            availability_view = run_scheduler(
                problem,
                controls,
                baseline,
                fairness_strategy=AllPairsAvailabilityFairness(),
            )
        st.session_state.last_solution = solution_view
        st.session_state.last_solution_availability = availability_view
        st.success("Optimisation finished: OPTIMAL")
        log_action(
            page="Operator Scheduling",
            action="run model",
            parameters={
                "fairness_mode": controls["fairness_mode"],
                "allowable_pct": controls["allowable_pct"],
                "consider_skills": controls["consider_skills"],
                "baseline_cost": baseline,
                "status": "OPTIMAL",
            },
        )
    except Exception as e:
        st.session_state.last_solution = {"status": f"ERROR: {e}"}
        st.session_state.last_solution_availability = {"status": f"ERROR: {e}"}
        st.error(f"Optimisation failed: {e}")
        log_action(
            page="Operator Scheduling",
            action="run model error",
            parameters={
                "error": str(e),
                "fairness_mode": controls["fairness_mode"],
                "allowable_pct": controls["allowable_pct"],
                "consider_skills": controls["consider_skills"],
            },
        )


# --- Reporting Components ---


def render_header():
    add_title("Operator Scheduling")
    st.markdown(
        """
        <style>
          .equation-label {font-size:16px; font-weight:600; text-align:center; margin:8px 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_cost_and_table(solution_view: SolutionView):
    total_cost = solution_view.total_cost
    baseline = solution_view.baseline_cost
    cost_limit = solution_view.cost_limit

    st.markdown(
        """
        <div style="text-align:center; margin-top: -6px;">
            <div style="font-size:30px; font-weight:700;">Total Cost (£)</div>
            <div style="font-size:44px; font-weight:800; margin:6px 0;">{value}</div>
        </div>
        """.format(value=f"£{total_cost:,.2f}"),
        unsafe_allow_html=True,
    )
    if baseline is not None and baseline != 0:
        delta_pct = ((total_cost - baseline) / baseline) * 100
        st.caption(
            f"vs baseline £{baseline:,.2f} ({delta_pct:+.1f}%)",
            text_alignment="center",
        )
    if cost_limit is not None:
        st.caption(f"Cost limit: £{cost_limit:,.2f}")

    schedule_df = solution_view.schedule
    if schedule_df.empty:
        st.info("No schedule table available.")
        return
    days = _ordered_days(solution_view.problem.days)
    pivot = schedule_df.pivot_table(
        index="student", columns="day", values="hours", fill_value=0, aggfunc="sum"
    )
    pivot = pivot.reindex(columns=days, fill_value=0)
    weekly_hours = pd.Series(solution_view.weekly_hours, name="Total")
    wage_rates = pd.Series(solution_view.problem.wage_rates)
    table = pivot.join(weekly_hours).join(wage_rates.rename("Wage Rate"))
    table["Weekly Cost"] = table["Total"] * table["Wage Rate"]
    total_row = pd.DataFrame([table.sum(numeric_only=True)])
    total_row.index = ["TOTAL"]
    total_row["Wage Rate"] = ""
    table = pd.concat([table, total_row], axis=0)
    money_fmt = lambda v: f"£{v:,.0f}" if pd.notna(v) and v != "" else v
    for col in ["Wage Rate", "Weekly Cost"]:
        table[col] = table[col].apply(money_fmt)
    st.markdown("#### Detailed Allocation Table")
    st.dataframe(
        table.reset_index().rename(columns={"index": "Student"}),
        width='stretch',
    )


def render_schedule(view: SolutionView, view_label: str):
    schedule_df = view.schedule
    cost_df = view.cost_breakdown.sort_values("cost", ascending=False)
    if schedule_df.empty or "availability" not in schedule_df.columns:
        st.info("No utilisation data available.")
        return
    st.markdown(f"### {view_label}")
    chart = build_workload_chart(
        schedule_df, cost_df, view.day_color_map, view.day_order
    )
    st.altair_chart(chart, width="stretch")


def render_footer(solution_view: SolutionView, problem_ctx: ProblemContext):
    st.markdown("---")
    objective_label = (
        solution_view.objective_label
        if solution_view.objective_label is not None
        else "N/A"
    )
    st.caption(
        f"Objective: {objective_label} "
        f"| Data: {problem_ctx.source} "
        f"| Students: {len(solution_view.problem.students)}"
    )


def render_results(
    solution_view: SolutionView,
    availability_view: SolutionView,
    use_availability: bool,
    problem_ctx: ProblemContext,
):
    title_section = st.container()
    kpi_section = st.container()
    schedule_section = st.container()
    footer_section = st.container()

    with title_section:
        render_header()
    with kpi_section:
        active_view = availability_view if use_availability else solution_view
        view_label = (
            "Hour / Availability Utilisation"
            if use_availability
            else "Hour Distribution"
        )
        render_cost_and_table(active_view)
    with schedule_section:
        render_schedule(active_view, view_label)
    with footer_section:
        render_footer(solution_view, problem_ctx)


# --- Page Entry Point ---


def init_session():
    seed_everything()
    if "last_solution" not in st.session_state:
        st.session_state.last_solution = {}


def main():
    init_session()

    problem_ctx, controls, baseline = render_sidebar_and_problem()
    if not problem_ctx.problem:
        st.stop()

    maybe_run_optimisation(problem_ctx.problem, controls, baseline)

    solution_view = _coerce_solution_view(st.session_state.get("last_solution"))
    if not solution_view:
        st.info("Upload data and run optimisation to see results.")
        st.stop()

    availability_view = _coerce_solution_view(
        st.session_state.get("last_solution_availability")
    )
    if not availability_view:
        availability_view = solution_view

    st.session_state.last_solution = solution_view
    st.session_state.last_solution_availability = availability_view
    render_results(
        solution_view,
        availability_view,
        controls["use_availability"],
        problem_ctx,
    )


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
