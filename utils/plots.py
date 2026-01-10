import altair as alt
import pandas as pd
import streamlit as st


def build_workload_chart(
    schedule_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    day_color_map: dict,
    day_order: list[str],
) -> alt.LayerChart:

    # ---- Canonical ordering ----
    student_order = cost_df["student"].tolist()

    # ---- Canonical day colour scale (safe) ----
    day_domain = [d for d in day_order if d in day_color_map]
    day_range = [day_color_map[d] for d in day_domain]

    # ---- Aggregate per-student totals ----
    avail_df = (
        schedule_df
        .groupby("student", as_index=False)
        .agg(
            hours=("hours", "sum"),
            availability=("availability", "sum"),
        )
    )

    # ---- Base stacked bars (VERTICAL) ----
    workload_chart = (
        alt.Chart(schedule_df)
        .mark_bar(width=60, cornerRadius=20)
        .encode(
            x=alt.X("student:N", sort=student_order, title=None),
            y=alt.Y("hours:Q", stack="zero", title="Hours"),
            color=alt.Color(
                "day:N",
                scale=alt.Scale(domain=day_domain, range=day_range),
                legend=alt.Legend(title="Day of Week"),
            ),
            tooltip=["student", "day", "hours", "availability"],
        )
    )

    # ---- Availability reference (horizontal tick) ----
    util_line = (
        alt.Chart(avail_df)
        .mark_tick(
            orient="horizontal",
            thickness=2,
            size=36,
            color="black",
            opacity=0.8,
        )
        .encode(
            x=alt.X("student:N", sort=student_order),
            y=alt.Y("availability:Q"),
        )
    )

    # ---- Utilisation % label ----
    util_text = (
        alt.Chart(avail_df)
        .mark_text(
            dy=-8,
            fontSize=11,
            color="black",
        )
        .encode(
            x=alt.X("student:N", sort=student_order),
            y=alt.Y("availability:Q"),
            text=alt.Text("util:Q", format=".0%"),
        )
        .transform_calculate(
            util="datum.hours / datum.availability"
        )
    )

    # ---- Total allocated hours label ----
    total_text = (
        alt.Chart(avail_df)
        .mark_text(
            dy=-14,
            fontWeight="bold",
            fontSize=16,
            color="black",
        )
        .encode(
            x=alt.X("student:N", sort=student_order),
            y=alt.Y("hours:Q"),
            text=alt.Text("hours:Q", format=".1f"),
        )
    )

    # ---- Final composed chart ----
    return (
        workload_chart
            + util_line
            + util_text
            + total_text
        ).properties(
            height=600, width=1200
        )



def build_utilisation_chart(schedule_df: pd.DataFrame, cost_df: pd.DataFrame) -> alt.Chart:
    util_df = (
        schedule_df.groupby("student")
        .agg(hours=("hours", "sum"), availability=("availability", "sum"))
        .reset_index()
    )
    util_df["utilisation"] = (util_df["hours"] / util_df["availability"]).fillna(0).clip(0, 1)
    return (
        alt.Chart(util_df)
        .mark_bar(cornerRadius=10, color="#4c78a8")
        .encode(
            x=alt.X("utilisation:Q", title="Utilisation (hours / availability)", axis=alt.Axis(format="%")),
            y=alt.Y("student:N", sort=cost_df["student"].tolist(), title=None),
            tooltip=["student", "hours", "availability", "utilisation"],
        )
        .properties(height=260, title="Utilisation")
    )
