# app.py
"""
Streamlit app for the SEIRD ABM.
Run with:
    streamlit run app.py
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import altair as alt

from model import SEIRDModel
from utils.visualization import (
    prepare_epi_curve_df,
    prepare_alert_df,
    latest_grid_positions,
)
from config.parameters import TOTAL_TICKS, POPULATION_SIZE


st.set_page_config(
    page_title="SEIRD ABM Simulator",
    layout="wide",
)

st.title("SEIRD Agent-Based Model Simulator")
st.caption("Bayesian agent-based model of infectious disease transmission in an urban center")

if "playing" not in st.session_state:
    st.session_state.playing = False

if "model" not in st.session_state:
    st.session_state.model = None

if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()

if "initialized" not in st.session_state:
    st.session_state.initialized = False


with st.sidebar:
    st.header("Simulation Controls")

    random_seed = st.number_input(
        "Random seed",
        min_value=1,
        max_value=999999,
        value=42,
        step=1,
    )

    population_size = st.number_input(
        "Population size",
        min_value=100,
        max_value=10000,
        value=POPULATION_SIZE,
        step=100,
    )

    run_ticks = st.slider(
        "Ticks to run",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
    )

    st.markdown("---")

    init_clicked = st.button("Initialize model", use_container_width=True)
    step_clicked = st.button("Run 1 tick", use_container_width=True)
    run_clicked = st.button(f"Run {run_ticks} ticks", use_container_width=True)
    run_full_clicked = st.button("Run until end", use_container_width=True)
    reset_clicked = st.button("Reset", use_container_width=True)


if reset_clicked:
    st.session_state.model = None
    st.session_state.results_df = pd.DataFrame()
    st.session_state.initialized = False
    st.rerun()

if init_clicked:
    with st.spinner("Initializing model..."):
        st.session_state.model = SEIRDModel(
            population_size=int(population_size),
            random_seed=int(random_seed),
        )
        st.session_state.results_df = st.session_state.model.get_results_df()
        st.session_state.initialized = True

if st.session_state.model is not None:
    if step_clicked:
        st.session_state.model.step()
        st.session_state.results_df = st.session_state.model.get_results_df()

    if run_clicked:
        st.session_state.model.run(run_ticks)
        st.session_state.results_df = st.session_state.model.get_results_df()

    if run_full_clicked:
        st.session_state.model.run_until_end()
        st.session_state.results_df = st.session_state.model.get_results_df()


if st.session_state.model is None:
    st.info("Initialize the model from the sidebar to start the simulation.")
else:
    df = st.session_state.results_df
    latest = df.iloc[-1]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Tick", int(latest["tick"]))
    c2.metric("Day", float(latest["day"]))
    c3.metric("Susceptible", int(latest["susceptible"]))
    c4.metric("Exposed", int(latest["exposed"]))
    c5.metric("Infectious", int(latest["total_infectious"]))
    c6.metric("Dead", int(latest["dead"]))

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Epidemic curves", "Policy", "Grid snapshot", "Raw data"]
    )

    with tab1:
        st.subheader("SEIRD epidemic curves")

        curve_df = prepare_epi_curve_df(df)

        chart = (
            alt.Chart(curve_df)
            .mark_line(interpolate="step-after")
            .encode(
                x=alt.X("tick:Q", title="Tick"),
                y=alt.Y("count:Q", title="Agents"),
                color=alt.Color("state:N", title="State"),
                tooltip=["tick", "day", "state", "count"],
            )
            .properties(height=420)
        )

        st.altair_chart(chart, use_container_width=True)

        incidence_df = df[["tick", "day", "new_exposed", "new_dead", "confirmed_cases"]].copy()
        incidence_long = incidence_df.melt(
            id_vars=["tick", "day"],
            value_vars=["new_exposed", "new_dead", "confirmed_cases"],
            var_name="indicator",
            value_name="value",
        )

        incidence_chart = (
            alt.Chart(incidence_long)
            .mark_line()
            .encode(
                x=alt.X("tick:Q", title="Tick"),
                y=alt.Y("value:Q", title="Count"),
                color=alt.Color("indicator:N", title="Indicator"),
                tooltip=["tick", "day", "indicator", "value"],
            )
            .properties(height=300)
        )

        st.altair_chart(incidence_chart, use_container_width=True)

    with tab2:
        st.subheader("Health Ministry policy state")

        alert_df = prepare_alert_df(df)

        alert_df["alert_level"] = pd.to_numeric(alert_df["alert_level"], errors="coerce")
        alert_df = alert_df.dropna(subset=["alert_level"])

        alert_chart = (
            alt.Chart(alert_df)
            .mark_line(interpolate="step-after")
            .encode(
                x=alt.X("tick:Q", title="Tick"),
                y=alt.Y("alert_level:Q", title="Alert level"),
                tooltip=["tick", "day", "alert_state", "seven_day_prev"],
            )
            .properties(height=300)
        )

        st.altair_chart(alert_chart, use_container_width=True)

        prev_df = df[["tick", "day", "seven_day_prev", "alert_state"]].copy()
        prev_df["seven_day_prev"] = pd.to_numeric(prev_df["seven_day_prev"], errors="coerce")
        prev_df = prev_df.dropna(subset=["seven_day_prev"])

        prev_chart = (
            alt.Chart(prev_df)
            .mark_line(color="firebrick")
            .encode(
                x=alt.X("tick:Q", title="Tick"),
                y=alt.Y("seven_day_prev:Q", title="7-day prevalence"),
                tooltip=["tick", "day", "seven_day_prev", "alert_state"],
            )
            .properties(height=300)
        )

        st.altair_chart(prev_chart, use_container_width=True)

    with tab3:
        st.subheader("Live spatial animation")

        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])

        with col_ctrl1:
            if st.button("▶ Play / ⏸ Pause", use_container_width=True):
                st.session_state.playing = not st.session_state.get("playing", False)

        with col_ctrl2:
            tick_delay = st.slider("Speed (sec/tick)", 0.1, 2.0, 0.5, 0.1)

        st.caption(f"Status: {'▶ Playing' if st.session_state.get('playing') else '⏸ Paused'}")

        chart_placeholder = st.empty()
        info_placeholder = st.empty()

        def render_grid():
            pos_df = latest_grid_positions(st.session_state.model.individuals)
            sample_size = min(2000, len(pos_df))
            pos_df = pos_df.sample(sample_size, random_state=42)

            scatter = (
                alt.Chart(pos_df)
                .mark_circle(size=20, opacity=0.65)
                .encode(
                    x=alt.X("x:Q", title="X", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("y:Q", title="Y", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("state:N", title="State"),
                    tooltip=["agent_id", "x", "y", "state", "at_work", "residence"],
                )
                .properties(height=600)
            )
            chart_placeholder.altair_chart(scatter, use_container_width=True)
            info_placeholder.caption(
                f"Tick {st.session_state.model.tick} | Day {st.session_state.model.current_day} | Showing {sample_size} agents"
            )

        # Always render current state
        render_grid()

        # Auto-play loop
        if st.session_state.get("playing", False):
            import time
            while st.session_state.get("playing", False):
                if st.session_state.model.tick >= TOTAL_TICKS:
                    st.session_state.playing = False
                    st.info("Simulation ended.")
                    break

                st.session_state.model.step()
                st.session_state.results_df = st.session_state.model.get_results_df()
                render_grid()
                time.sleep(tick_delay)

    with tab4:
        st.subheader("Raw simulation output")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV results",
            data=csv,
            file_name="simulation_results.csv",
            mime="text/csv",
        )
