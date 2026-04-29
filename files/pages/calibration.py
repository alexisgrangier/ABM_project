# pages/calibration.py
"""
Streamlit page — ABC-SMC Bayesian calibration.

Run from the project root:
    streamlit run app.py
Then navigate to the "Calibration" page in the sidebar.
"""

from __future__ import annotations

import threading
import time
import queue

import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

from calibration.abc_smc import (
    ABCSMCCalibrator,
    Prior,
    compute_summary_statistics,
    normalised_euclidean_distance,
)
from model import SEIRDModel
from config.parameters import POPULATION_SIZE, TOTAL_TICKS


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="ABC-SMC Calibration", layout="wide")
st.title("🔬 Bayesian Calibration — ABC-SMC")
st.caption(
    "Calibrate the SEIRD ABM to target epidemic statistics using "
    "Approximate Bayesian Computation – Sequential Monte Carlo."
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

for key, default in [
    ("abc_running",    False),
    ("abc_result",     None),
    ("abc_progress",   []),   # list of (gen, accepted, n_sims) tuples
    ("abc_calibrator", None),
    ("pilot_stats",    None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — calibration settings
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Calibration settings")

    st.subheader("Target statistics")
    st.caption(
        "Enter 'observed' values you want the model to match. "
        "You can generate them from a reference run below."
    )

    obs_peak_inf    = st.number_input("Peak infectious count",     min_value=1,   value=150,  step=5)
    obs_peak_tick   = st.number_input("Peak timing (tick)",        min_value=1,   value=180,  step=10)
    obs_total_dead  = st.number_input("Total dead",                min_value=0,   value=30,   step=1)
    obs_attack_rate = st.slider(      "Attack rate (0–1)",         min_value=0.0, max_value=1.0, value=0.30, step=0.01)

    observed_stats = {
        "peak_infectious":  float(obs_peak_inf),
        "peak_timing_tick": float(obs_peak_tick),
        "total_dead":       float(obs_total_dead),
        "attack_rate":      float(obs_attack_rate),
    }

    st.divider()
    st.subheader("SMC settings")

    n_particles   = st.slider("Particles per generation", 50,  500, 100, step=25)
    n_populations = st.slider("Number of generations",    2,   8,   4,   step=1)
    pop_size      = st.number_input("Population size (agents)", 100, 2000, 500, step=100)
    run_ticks     = st.slider("Simulation ticks",         100, TOTAL_TICKS, 730, step=10)

    st.divider()

    run_btn   = st.button("▶ Run ABC-SMC calibration", use_container_width=True, type="primary")
    pilot_btn = st.button("🔍 Generate reference stats from default model",
                          use_container_width=True)
    clear_btn = st.button("🗑 Clear results", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Generate reference statistics
# ─────────────────────────────────────────────────────────────────────────────

if pilot_btn:
    with st.spinner("Running reference simulation …"):
        ref_model = SEIRDModel(population_size=int(pop_size), random_seed=42)
        ref_model.run(int(run_ticks))
        ref_df    = ref_model.get_results_df()
        ref_stats = compute_summary_statistics(ref_df, int(pop_size))
        st.session_state.pilot_stats = ref_stats

    st.success("Reference statistics generated — use these as targets!")
    st.json(ref_stats)

if st.session_state.pilot_stats:
    st.info(
        "💡 Reference stats available above. "
        "Manually enter these values in the sidebar targets to calibrate to them."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Clear
# ─────────────────────────────────────────────────────────────────────────────

if clear_btn:
    st.session_state.abc_result     = None
    st.session_state.abc_progress   = []
    st.session_state.abc_calibrator = None
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Run calibration (blocking — Streamlit does not support true background threads
# on the free tier; we run synchronously with live progress bars)
# ─────────────────────────────────────────────────────────────────────────────

if run_btn:
    st.session_state.abc_progress   = []
    st.session_state.abc_result     = None
    st.session_state.abc_running    = True

    progress_placeholder = st.empty()
    status_placeholder   = st.empty()

    progress_q: queue.Queue = queue.Queue()

    def _progress_cb(generation: int, n_accepted: int, n_sims: int) -> None:
        progress_q.put((generation, n_accepted, n_sims))

    calibrator = ABCSMCCalibrator(
        observed_stats  = observed_stats,
        n_particles     = int(n_particles),
        n_populations   = int(n_populations),
        run_ticks       = int(run_ticks),
        population_size = int(pop_size),
        on_progress     = _progress_cb,
        verbose         = True,
    )
    st.session_state.abc_calibrator = calibrator

    # Because Streamlit reruns on every widget interaction, we run the
    # calibration synchronously here (no background thread needed).
    with st.spinner("Running ABC-SMC … this may take several minutes."):
        posterior = calibrator.run()

    st.session_state.abc_result  = posterior
    st.session_state.abc_running = False
    st.success("✅ ABC-SMC calibration complete!")
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Display results
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.abc_result is not None:
    calibrator = st.session_state.abc_calibrator
    posterior  = st.session_state.abc_result

    st.markdown("---")
    st.header("Calibration results")

    # ── Convergence ──────────────────────────────────────────────────────
    st.subheader("SMC convergence")

    pops = calibrator.populations
    conv_df = pd.DataFrame({
        "Generation":   [p.generation         for p in pops],
        "Tolerance ε":  [p.epsilon             for p in pops],
        "Simulations":  [p.n_simulations_run   for p in pops],
        "ESS":          [p.effective_sample_size() for p in pops],
    })
    st.dataframe(conv_df, use_container_width=True)

    eps_chart = (
        alt.Chart(conv_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Generation:O", title="SMC generation"),
            y=alt.Y("Tolerance ε:Q", title="ε (lower = tighter)"),
            tooltip=["Generation", "Tolerance ε", "Simulations", "ESS"],
        )
        .properties(height=220, title="Tolerance ε per generation")
    )
    st.altair_chart(eps_chart, use_container_width=True)

    # ── Posterior summary table ───────────────────────────────────────────
    st.subheader("Posterior parameter estimates")

    summary = calibrator.posterior_summary(posterior)
    st.dataframe(summary.style.format("{:.5f}"), use_container_width=True)

    map_est = calibrator.map_estimate(posterior)
    st.markdown("**MAP estimate (highest-weight particle):**")
    st.json({k: round(v, 6) for k, v in map_est.items()})

    # ── Posterior marginals ───────────────────────────────────────────────
    st.subheader("Posterior marginal distributions")

    post_df = posterior.as_dataframe()

    param_labels = {
        "base_transmission_rate":  "β — transmission rate",
        "asymptomatic_fraction":   "σ_a — asymptomatic fraction",
        "incubation_period_ticks": "t_inc — incubation (ticks)",
        "infectious_period_ticks": "t_inf — infectious (ticks)",
    }

    cols = st.columns(2)
    for i, (param, label) in enumerate(param_labels.items()):
        ax_col = cols[i % 2]
        hist_df = post_df[["weight", param]].copy()
        hist_df.columns = ["weight", "value"]

        # Weighted histogram via binning
        vals  = hist_df["value"].values
        w     = hist_df["weight"].values
        n_bins = 30
        counts, edges = np.histogram(vals, bins=n_bins, weights=w * len(w), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        bin_df = pd.DataFrame({"value": centers, "density": counts})

        mean_val = float(np.average(vals, weights=w))

        bar = (
            alt.Chart(bin_df)
            .mark_bar(color="#4C72B0", opacity=0.75)
            .encode(
                x=alt.X("value:Q", title=label, bin=False),
                y=alt.Y("density:Q", title="Density"),
                tooltip=["value", "density"],
            )
        )
        mean_line = (
            alt.Chart(pd.DataFrame({"mean": [mean_val]}))
            .mark_rule(color="#DD4444", strokeWidth=2)
            .encode(x=alt.X("mean:Q"))
        )
        chart = (bar + mean_line).properties(height=220, title=f"Posterior: {label}")
        ax_col.altair_chart(chart, use_container_width=True)

    # ── Prior vs posterior overlay ─────────────────────────────────────────
    st.subheader("Bayesian update: prior vs posterior")

    prior    = calibrator.prior
    rng_plot = np.random.default_rng(0)
    n_prior  = 2000
    prior_df = pd.DataFrame([prior.sample(rng_plot) for _ in range(n_prior)])

    cols2 = st.columns(2)
    for i, (param, label) in enumerate(param_labels.items()):
        ax_col   = cols2[i % 2]
        pv       = post_df["weight"].values
        post_vals = post_df[param].values
        prior_vals = prior_df[param].values

        lo   = min(prior_vals.min(), post_vals.min())
        hi   = max(prior_vals.max(), post_vals.max())
        bins = np.linspace(lo, hi, 30)

        pc, pe   = np.histogram(prior_vals, bins=bins, density=True)
        ptc, pte = np.histogram(post_vals, bins=bins, weights=pv * len(pv), density=True)
        centers  = (bins[:-1] + bins[1:]) / 2

        cdf_prior = pd.DataFrame({"value": centers, "density": pc,  "dist": "Prior"})
        cdf_post  = pd.DataFrame({"value": centers, "density": ptc, "dist": "Posterior"})
        overlay   = pd.concat([cdf_prior, cdf_post])

        chart = (
            alt.Chart(overlay)
            .mark_bar(opacity=0.55)
            .encode(
                x=alt.X("value:Q", title=label, bin=False),
                y=alt.Y("density:Q"),
                color=alt.Color("dist:N", scale=alt.Scale(
                    domain=["Prior", "Posterior"],
                    range=["#AAAAAA", "#4C72B0"],
                )),
                tooltip=["value", "density", "dist"],
            )
            .properties(height=220, title=f"Prior vs Posterior: {label}")
        )
        ax_col.altair_chart(chart, use_container_width=True)

    # ── Posterior predictive check ──────────────────────────────────────────
    st.subheader("Posterior predictive check")
    st.caption(
        "Run a forward simulation using the MAP-estimated parameters and compare "
        "the resulting summary statistics to the targets."
    )

    if st.button("Run posterior predictive simulation", type="secondary"):
        with st.spinner("Simulating with MAP parameters …"):
            import config.parameters as params
            import submodels.transmission as tx

            orig_beta  = params.BASE_TRANSMISSION_RATE
            orig_asym  = params.ASYMPTOMATIC_FRACTION
            orig_tinc  = params.INCUBATION_PERIOD_TICKS
            orig_tinf  = params.INFECTIOUS_PERIOD_TICKS

            params.BASE_TRANSMISSION_RATE  = map_est["base_transmission_rate"]
            params.ASYMPTOMATIC_FRACTION   = map_est["asymptomatic_fraction"]
            params.INCUBATION_PERIOD_TICKS = int(round(map_est["incubation_period_ticks"]))
            params.INFECTIOUS_PERIOD_TICKS = int(round(map_est["infectious_period_ticks"]))
            tx.spatial_transmission.__defaults__ = (params.BASE_TRANSMISSION_RATE,)
            tx.commute_transmission.__defaults__ = (params.BASE_TRANSMISSION_RATE,)

            try:
                ppc_model = SEIRDModel(population_size=int(pop_size), random_seed=99)
                ppc_model.run(int(run_ticks))
                ppc_df  = ppc_model.get_results_df()
                ppc_stats = compute_summary_statistics(ppc_df, int(pop_size))
            finally:
                params.BASE_TRANSMISSION_RATE  = orig_beta
                params.ASYMPTOMATIC_FRACTION   = orig_asym
                params.INCUBATION_PERIOD_TICKS = orig_tinc
                params.INFECTIOUS_PERIOD_TICKS = orig_tinf
                tx.spatial_transmission.__defaults__ = (orig_beta,)
                tx.commute_transmission.__defaults__ = (orig_beta,)

        compare_df = pd.DataFrame({
            "statistic": list(observed_stats.keys()),
            "target":    list(observed_stats.values()),
            "simulated": [ppc_stats[k] for k in observed_stats],
        })
        compare_df["error %"] = (
            (compare_df["simulated"] - compare_df["target"]).abs()
            / compare_df["target"].replace(0, 1)
            * 100
        ).round(2)

        st.dataframe(compare_df, use_container_width=True)

        d_final = normalised_euclidean_distance(ppc_stats, observed_stats)
        st.metric("Distance d(simulated, target)", f"{d_final:.4f}",
                  delta=f"ε_final = {posterior.epsilon:.4f}",
                  delta_color="inverse")

    # ── Raw particle data ──────────────────────────────────────────────────
    with st.expander("Raw posterior particles"):
        st.dataframe(post_df, use_container_width=True)
        csv = post_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download posterior CSV",
            data=csv,
            file_name="abc_smc_posterior.csv",
            mime="text/csv",
        )

else:
    if not st.session_state.abc_running:
        st.info(
            "Configure calibration settings in the sidebar and press "
            "**▶ Run ABC-SMC calibration** to start."
        )
        st.markdown("""
### How this works

**Approximate Bayesian Computation (ABC)** is the standard method for
calibrating simulators that have no closed-form likelihood — exactly the
situation with agent-based models.

Instead of evaluating $p(x_{obs} | \\theta)$ directly, ABC:

1. Draws candidate parameters $\\theta^*$ from a prior $\\pi(\\theta)$
2. Simulates the model with $\\theta^*$ → synthetic output $x^*$
3. Accepts $\\theta^*$ if $d(x^*, x_{obs}) < \\varepsilon$

The **Sequential Monte Carlo** variant (Toni et al. 2009) runs a sequence
of populations $P_1, \\ldots, P_T$ with shrinking tolerances
$\\varepsilon_1 > \\cdots > \\varepsilon_T$, progressively concentrating
particles around the posterior.

**Parameters calibrated:**

| Parameter | Symbol | Prior range |
|---|---|---|
| Base transmission rate | β | [0.005, 0.25] |
| Asymptomatic fraction | σ_a | [0.10, 0.70] |
| Incubation period | t_inc | [2, 10] ticks |
| Infectious period | t_inf | [4, 28] ticks |

**Summary statistics:**
- Peak infectious count
- Peak timing (tick)
- Total dead
- Final attack rate
""")
