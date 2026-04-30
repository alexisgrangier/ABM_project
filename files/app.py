# app.py
"""
Streamlit app for the SEIRD ABM.
Run with:
    streamlit run app.py
"""

from __future__ import annotations
import time
import streamlit as st
import pandas as pd
import numpy as np
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

# ── Session state ─────────────────────────────────────────────────────────────

for key, default in [
    ("playing",       False),
    ("model",         None),
    ("results_df",    pd.DataFrame()),
    ("initialized",   False),
    ("posterior_df",  None),
    ("ppc_envelope",  None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Simulation Controls")

    random_seed = st.number_input(
        "Random seed", min_value=1, max_value=999999, value=42, step=1,
    )
    population_size = st.number_input(
        "Population size", min_value=100, max_value=10000,
        value=POPULATION_SIZE, step=100,
    )
    run_ticks = st.slider("Ticks to run", min_value=1, max_value=100, value=10, step=1)

    st.markdown("---")

    init_clicked     = st.button("Initialize model",      use_container_width=True)
    step_clicked     = st.button("Run 1 tick",            use_container_width=True)
    run_clicked      = st.button(f"Run {run_ticks} ticks", use_container_width=True)
    run_full_clicked = st.button("Run until end",         use_container_width=True)
    reset_clicked    = st.button("Reset",                 use_container_width=True)

    st.markdown("---")
    st.header("Bayesian calibration")

    with st.expander("Load posterior particles", expanded=False):
        st.caption(
            "Upload the abc_smc_posterior.csv produced by the calibration page, "
            "or run a quick ABC below to generate one."
        )

        uploaded = st.file_uploader("Posterior CSV", type="csv", key="posterior_upload")
        if uploaded is not None:
            try:
                df_up = pd.read_csv(uploaded)
                required = {
                    "base_transmission_rate", "asymptomatic_fraction",
                    "incubation_period_ticks", "infectious_period_ticks", "weight",
                }
                if required.issubset(df_up.columns):
                    st.session_state.posterior_df  = df_up
                    st.session_state.ppc_envelope  = None
                    st.success(f"Loaded {len(df_up)} particles ✓")
                else:
                    st.error(f"CSV must contain columns: {required}")
            except Exception as e:
                st.error(f"Could not parse file: {e}")

        st.markdown("---")
        st.caption("**Or** run a minimal ABC-SMC right now (fast, low resolution).")

        quick_particles   = st.slider("Particles",   50,  200, 50,  step=25, key="qp")
        quick_ticks       = st.slider("Sim ticks",  100,  730, 200, step=50, key="qt")

        run_quick_abc = st.button("▶ Run quick ABC-SMC", use_container_width=True)

    if st.session_state.posterior_df is not None:
        n_p = len(st.session_state.posterior_df)
        st.success(f"Posterior loaded — {n_p} particles")

        n_ppc = st.slider(
            "PPC trajectories to draw", min_value=10, max_value=min(200, n_p),
            value=min(20, n_p), step=10, key="n_ppc",
        )
        compute_ppc = st.button("Compute uncertainty bands", use_container_width=True)
    else:
        st.info("No posterior loaded. Upload a CSV or run quick ABC above.")
        compute_ppc = False
        n_ppc = 20


# ── Model actions ─────────────────────────────────────────────────────────────

if reset_clicked:
    st.session_state.model         = None
    st.session_state.results_df    = pd.DataFrame()
    st.session_state.initialized   = False
    st.session_state.ppc_envelope  = None
    st.rerun()

if init_clicked:
    with st.spinner("Initializing model..."):
        st.session_state.model = SEIRDModel(
            population_size=int(population_size),
            random_seed=int(random_seed),
        )
        st.session_state.results_df = st.session_state.model.get_results_df()
        st.session_state.initialized = True
        st.session_state.ppc_envelope = None

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


# ── Quick ABC-SMC ─────────────────────────────────────────────────────────────

if run_quick_abc:
    if st.session_state.model is None:
        st.warning("Initialize the model first so we have target statistics to calibrate to.")
    else:
        from calibration.abc_smc import (
            ABCSMCCalibrator,
            compute_summary_statistics,
        )

        ref_stats = compute_summary_statistics(
            st.session_state.results_df, int(population_size)
        )

        with st.spinner("Running ABC-SMC ..."):
            cal = ABCSMCCalibrator(
                observed_stats  = ref_stats,
                n_particles     = int(quick_particles),
                n_populations   = 1,
                run_ticks       = int(quick_ticks),
                population_size = int(population_size),
                verbose         = True,
            )
            posterior = cal.run()
            post_df   = posterior.as_dataframe()

        st.session_state.posterior_df = post_df
        st.session_state.ppc_envelope = None
        st.success(f"ABC-SMC done — {len(post_df)} particles loaded ✓")
        st.rerun()


# ── Posterior predictive computation ─────────────────────────────────────────

def _run_ppc(
    posterior_df: pd.DataFrame,
    n_draws: int,
    pop_size: int,
    n_ticks: int,
) -> dict[str, pd.DataFrame]:
    import config.parameters as params
    import submodels.transmission as tx

    weights = posterior_df["weight"].values
    weights = weights / weights.sum()

    rng = np.random.default_rng(0)
    indices = rng.choice(len(posterior_df), size=n_draws, replace=True, p=weights)
    sampled = posterior_df.iloc[indices].reset_index(drop=True)

    track_cols = ["total_infectious", "susceptible", "exposed", "dead"]
    all_runs: dict[str, list] = {c: [] for c in track_cols}
    tick_index = None

    orig = {
        "BASE_TRANSMISSION_RATE":  params.BASE_TRANSMISSION_RATE,
        "ASYMPTOMATIC_FRACTION":   params.ASYMPTOMATIC_FRACTION,
        "INCUBATION_PERIOD_TICKS": params.INCUBATION_PERIOD_TICKS,
        "INFECTIOUS_PERIOD_TICKS": params.INFECTIOUS_PERIOD_TICKS,
    }

    try:
        for i, row in sampled.iterrows():
            params.BASE_TRANSMISSION_RATE  = float(row["base_transmission_rate"])
            params.ASYMPTOMATIC_FRACTION   = float(row["asymptomatic_fraction"])
            params.INCUBATION_PERIOD_TICKS = int(round(row["incubation_period_ticks"]))
            params.INFECTIOUS_PERIOD_TICKS = int(round(row["infectious_period_ticks"]))
            tx.spatial_transmission.__defaults__ = (params.BASE_TRANSMISSION_RATE,)
            tx.commute_transmission.__defaults__ = (params.BASE_TRANSMISSION_RATE,)

            seed  = int(rng.integers(0, 2**31))
            model = SEIRDModel(population_size=pop_size, random_seed=seed)
            model.run(n_ticks)
            run_df = model.get_results_df()

            if tick_index is None:
                tick_index = run_df["tick"].values

            for col in track_cols:
                if col in run_df.columns:
                    series = run_df.set_index("tick")[col].reindex(tick_index).ffill().values
                    all_runs[col].append(series)

    finally:
        params.BASE_TRANSMISSION_RATE  = orig["BASE_TRANSMISSION_RATE"]
        params.ASYMPTOMATIC_FRACTION   = orig["ASYMPTOMATIC_FRACTION"]
        params.INCUBATION_PERIOD_TICKS = orig["INCUBATION_PERIOD_TICKS"]
        params.INFECTIOUS_PERIOD_TICKS = orig["INFECTIOUS_PERIOD_TICKS"]
        tx.spatial_transmission.__defaults__ = (orig["BASE_TRANSMISSION_RATE"],)
        tx.commute_transmission.__defaults__ = (orig["BASE_TRANSMISSION_RATE"],)

    envelopes: dict[str, pd.DataFrame] = {}
    for col in track_cols:
        mat = np.array(all_runs[col])
        if mat.size == 0:
            continue
        envelopes[col] = pd.DataFrame({
            "tick":   tick_index,
            "median": np.median(mat, axis=0),
            "q025":   np.percentile(mat, 2.5,  axis=0),
            "q975":   np.percentile(mat, 97.5, axis=0),
            "q25":    np.percentile(mat, 25,   axis=0),
            "q75":    np.percentile(mat, 75,   axis=0),
        })

    return envelopes


if compute_ppc and st.session_state.posterior_df is not None:
    n_sim_ticks = (
        st.session_state.model.tick
        if st.session_state.model is not None
        else TOTAL_TICKS
    )
    with st.spinner("Computing posterior predictive bands ..."):
        result = _run_ppc(
            posterior_df = st.session_state.posterior_df,
            n_draws      = int(n_ppc),
            pop_size     = int(population_size),
            n_ticks      = n_sim_ticks,
        )
    st.session_state.ppc_envelope = result


# ── Main display ──────────────────────────────────────────────────────────────

if st.session_state.model is None:
    st.info("Initialize the model from the sidebar to start the simulation.")
else:
    df     = st.session_state.results_df
    latest = df.iloc[-1]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Tick",         int(latest["tick"]))
    c2.metric("Day",          float(latest["day"]))
    c3.metric("Susceptible",  int(latest["susceptible"]))
    c4.metric("Exposed",      int(latest["exposed"]))
    c5.metric("Infectious",   int(latest["total_infectious"]))
    c6.metric("Dead",         int(latest["dead"]))

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

        st.markdown("---")
        st.subheader("Posterior predictive uncertainty")

        envelope = st.session_state.ppc_envelope

        if envelope is None and st.session_state.posterior_df is None:
            st.info(
                "Load a posterior in the sidebar to see Bayesian uncertainty bands."
            )
        elif envelope is None:
            st.info(
                "Posterior loaded ✓ — press **Compute uncertainty bands** in the "
                "sidebar to generate the predictive envelope."
            )
        else:
            state_cfg = {
                "total_infectious": {"label": "Infectious",  "color": "#e45756"},
                "exposed":          {"label": "Exposed",     "color": "#f58518"},
                "susceptible":      {"label": "Susceptible", "color": "#4c78a8"},
                "dead":             {"label": "Dead",        "color": "#54a24b"},
            }

            for state_key, cfg in state_cfg.items():
                if state_key not in envelope:
                    continue

                env   = envelope[state_key].copy()
                color = cfg["color"]
                label = cfg["label"]

                band = (
                    alt.Chart(env)
                    .mark_area(opacity=0.15, color=color)
                    .encode(
                        x=alt.X("tick:Q", title="Tick"),
                        y=alt.Y("q025:Q", title="Agents"),
                        y2=alt.Y2("q975:Q"),
                    )
                )
                iqr = (
                    alt.Chart(env)
                    .mark_area(opacity=0.25, color=color)
                    .encode(
                        x="tick:Q",
                        y=alt.Y("q25:Q"),
                        y2=alt.Y2("q75:Q"),
                    )
                )
                median_line = (
                    alt.Chart(env)
                    .mark_line(color=color, strokeDash=[4, 2], strokeWidth=1.5)
                    .encode(
                        x="tick:Q",
                        y=alt.Y("median:Q"),
                        tooltip=[
                            alt.Tooltip("tick:Q"),
                            alt.Tooltip("median:Q", title="Posterior median"),
                            alt.Tooltip("q025:Q",   title="2.5%"),
                            alt.Tooltip("q975:Q",   title="97.5%"),
                        ],
                    )
                )

                if state_key in df.columns:
                    obs_line = (
                        alt.Chart(
                            df[["tick", state_key]].rename(columns={state_key: "value"})
                        )
                        .mark_line(color=color, strokeWidth=2.5)
                        .encode(
                            x="tick:Q",
                            y=alt.Y("value:Q"),
                            tooltip=["tick:Q", "value:Q"],
                        )
                    )
                    combined = (band + iqr + median_line + obs_line).properties(
                        height=240,
                        title=f"{label}: single run (solid) vs posterior predictive (bands)",
                    )
                else:
                    combined = (band + iqr + median_line).properties(
                        height=240,
                        title=f"{label}: posterior predictive envelope",
                    )

                st.altair_chart(combined, use_container_width=True)

            st.caption(
                "**Bands**: outer = 95% credible interval, inner = IQR (25–75%).  "
                "**Dashed line**: posterior median.  "
                "**Solid line**: current single run."
            )

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
        info_placeholder  = st.empty()

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
                f"Tick {st.session_state.model.tick} | "
                f"Day {st.session_state.model.current_day} | "
                f"Showing {sample_size} agents"
            )

        render_grid()

        if st.session_state.get("playing", False):
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