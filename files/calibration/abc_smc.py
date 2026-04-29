# calibration/abc_smc.py
"""
Approximate Bayesian Computation – Sequential Monte Carlo (ABC-SMC)
calibration for the SEIRD ABM.

Background
----------
ABMs have no closed-form likelihood, so standard MCMC is not applicable.
ABC replaces the likelihood with a distance-based acceptance rule:

    θ ~ π(θ)                   (prior)
    x ~ p(x | θ)               (simulate)
    accept if  d(x, x_obs) < ε (tolerance)

ABC-SMC (Toni et al. 2009) improves efficiency over basic rejection ABC
by running a sequence of T populations P_1, …, P_T with decreasing
tolerances ε_1 > ε_2 > … > ε_T.  Each population is a weighted particle
cloud that approximates the posterior p(θ | x_obs).

Parameters calibrated
---------------------
theta = {
    BASE_TRANSMISSION_RATE   (β),
    ASYMPTOMATIC_FRACTION    (σ_a),
    INCUBATION_PERIOD_TICKS  (t_inc),
    INFECTIOUS_PERIOD_TICKS  (t_inf),
}

Summary statistics used as observed data
-----------------------------------------
- peak_infectious   : maximum simultaneous infectious count
- peak_timing_tick  : tick at which peak occurs
- total_dead        : cumulative deaths by end of simulation
- attack_rate       : fraction of population ever infected

Usage
-----
    from calibration.abc_smc import ABCSMCCalibrator

    calibrator = ABCSMCCalibrator(
        observed_stats={"peak_infectious": 312, "peak_timing_tick": 180,
                        "total_dead": 47,       "attack_rate": 0.34},
        n_particles=200,
        n_populations=5,
    )
    posterior = calibrator.run()
    calibrator.plot_posterior(posterior)
"""

from __future__ import annotations

import copy
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

# ── Lazy imports of model (avoids circular deps at module load) ───────────────
def _import_model():
    from model import SEIRDModel  # noqa: PLC0415
    return SEIRDModel


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Prior:
    """
    Independent uniform priors over the four calibrated parameters.

    Bounds are deliberately wide to let the data speak.
    """
    beta_low:    float = 0.005
    beta_high:   float = 0.25

    asym_low:    float = 0.10
    asym_high:   float = 0.70

    t_inc_low:   float = 2.0       # ticks (= 1 day)
    t_inc_high:  float = 10.0      # ticks (= 5 days)

    t_inf_low:   float = 4.0       # ticks (= 2 days)
    t_inf_high:  float = 28.0      # ticks (= 14 days)

    param_names: list[str] = field(default_factory=lambda: [
        "base_transmission_rate",
        "asymptomatic_fraction",
        "incubation_period_ticks",
        "infectious_period_ticks",
    ])

    def sample(self, rng: np.random.Generator) -> dict[str, float]:
        return {
            "base_transmission_rate":  rng.uniform(self.beta_low,   self.beta_high),
            "asymptomatic_fraction":   rng.uniform(self.asym_low,   self.asym_high),
            "incubation_period_ticks": rng.uniform(self.t_inc_low,  self.t_inc_high),
            "infectious_period_ticks": rng.uniform(self.t_inf_low,  self.t_inf_high),
        }

    def log_density(self, theta: dict[str, float]) -> float:
        """Log of the uniform prior (0 inside bounds, -inf outside)."""
        checks = [
            self.beta_low   <= theta["base_transmission_rate"]  <= self.beta_high,
            self.asym_low   <= theta["asymptomatic_fraction"]   <= self.asym_high,
            self.t_inc_low  <= theta["incubation_period_ticks"] <= self.t_inc_high,
            self.t_inf_low  <= theta["infectious_period_ticks"] <= self.t_inf_high,
        ]
        return 0.0 if all(checks) else -np.inf


@dataclass
class Particle:
    theta:  dict[str, float]
    weight: float = 1.0


@dataclass
class Population:
    """One SMC population (list of weighted particles)."""
    particles:  list[Particle]
    epsilon:    float
    generation: int
    n_simulations_run: int = 0

    def normalise_weights(self) -> None:
        total = sum(p.weight for p in self.particles)
        for p in self.particles:
            p.weight /= total

    def effective_sample_size(self) -> float:
        w = np.array([p.weight for p in self.particles])
        return 1.0 / np.sum(w ** 2)

    def as_dataframe(self) -> pd.DataFrame:
        rows = [p.theta | {"weight": p.weight} for p in self.particles]
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_summary_statistics(df: pd.DataFrame, population_size: int) -> dict[str, float]:
    """
    Compute the four scalar summary statistics from a simulation results
    DataFrame.

    The DataFrame is produced by SEIRDModel.get_results_df().
    """
    if df.empty:
        return {
            "peak_infectious":  0.0,
            "peak_timing_tick": 0.0,
            "total_dead":       0.0,
            "attack_rate":      0.0,
        }

    infectious_col = "total_infectious"
    if infectious_col not in df.columns:
        # fall back if column name differs
        candidates = [c for c in df.columns if "infectious" in c.lower()]
        infectious_col = candidates[0] if candidates else None

    peak_infectious  = float(df[infectious_col].max()) if infectious_col else 0.0
    peak_timing_tick = float(df.loc[df[infectious_col].idxmax(), "tick"]) if infectious_col else 0.0

    total_dead = float(df["dead"].iloc[-1]) if "dead" in df.columns else 0.0

    # Attack rate: fraction ever infected = 1 - (final susceptible / N)
    if "susceptible" in df.columns:
        final_susceptible = float(df["susceptible"].iloc[-1])
        attack_rate = 1.0 - (final_susceptible / population_size)
    else:
        attack_rate = 0.0

    return {
        "peak_infectious":  peak_infectious,
        "peak_timing_tick": peak_timing_tick,
        "total_dead":       total_dead,
        "attack_rate":      attack_rate,
    }


def normalised_euclidean_distance(
    simulated: dict[str, float],
    observed:  dict[str, float],
    scales:    dict[str, float] | None = None,
) -> float:
    """
    Compute a normalised Euclidean distance between two summary stat dicts.

    Each dimension is divided by its scale (default: observed value, or 1
    if observed == 0) so that all four statistics contribute equally.
    """
    keys = list(observed.keys())
    if scales is None:
        scales = {k: (abs(observed[k]) if observed[k] != 0 else 1.0) for k in keys}

    sq_sum = sum(
        ((simulated.get(k, 0.0) - observed[k]) / scales[k]) ** 2
        for k in keys
    )
    return float(np.sqrt(sq_sum))


# ─────────────────────────────────────────────────────────────────────────────
# Perturbation kernel (Gaussian on bounded domain)
# ─────────────────────────────────────────────────────────────────────────────

def _perturb_particle(
    particle: Particle,
    bandwidths: dict[str, float],
    prior: Prior,
    rng: np.random.Generator,
    max_tries: int = 100,
) -> dict[str, float]:
    """
    Draw a new θ by perturbing an existing particle with Gaussian noise,
    rejecting proposals that fall outside the prior support.
    """
    for _ in range(max_tries):
        candidate = {
            k: v + rng.normal(0.0, bandwidths[k])
            for k, v in particle.theta.items()
        }
        if prior.log_density(candidate) > -np.inf:
            return candidate

    # If we can't find a valid perturbation, return the original (rare)
    warnings.warn("Could not perturb particle within prior bounds; returning original.")
    return dict(particle.theta)


def _compute_bandwidths(population: Population) -> dict[str, float]:
    """
    Silverman's rule-of-thumb bandwidth for each parameter dimension.
    bw = 2 * std(weighted particles)
    """
    df = population.as_dataframe()
    weights = df["weight"].values
    bandwidths: dict[str, float] = {}

    for name in population.particles[0].theta:
        vals = df[name].values
        weighted_mean = np.average(vals, weights=weights)
        weighted_var  = np.average((vals - weighted_mean) ** 2, weights=weights)
        bandwidths[name] = 2.0 * np.sqrt(weighted_var) + 1e-8

    return bandwidths


# ─────────────────────────────────────────────────────────────────────────────
# Weight computation (importance sampling correction)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_weight(
    theta_new: dict[str, float],
    prev_population: Population,
    bandwidths: dict[str, float],
    prior: Prior,
) -> float:
    """
    Importance weight for ABC-SMC (Toni et al. 2009, eq. 5).

    w_t(θ*) ∝ π(θ*) / Σ_j w_{t-1,j} K(θ* | θ_{t-1,j})

    where K is a Gaussian kernel.
    """
    if prior.log_density(theta_new) == -np.inf:
        return 0.0

    # Denominator: weighted sum of kernel evaluations
    denominator = 0.0
    for particle in prev_population.particles:
        log_kernel = sum(
            -0.5 * ((theta_new[k] - particle.theta[k]) / bandwidths[k]) ** 2
            - np.log(bandwidths[k])
            for k in theta_new
        )
        denominator += particle.weight * np.exp(log_kernel)

    if denominator <= 0:
        return 0.0

    # Prior is uniform so π(θ*) = constant (absorbed into normalisation)
    return 1.0 / denominator


# ─────────────────────────────────────────────────────────────────────────────
# Main calibrator class
# ─────────────────────────────────────────────────────────────────────────────

class ABCSMCCalibrator:
    """
    Calibrate the SEIRD ABM via Approximate Bayesian Computation –
    Sequential Monte Carlo.

    Parameters
    ----------
    observed_stats : dict
        Target summary statistics (keys: peak_infectious, peak_timing_tick,
        total_dead, attack_rate).
    n_particles : int
        Number of particles per population (default 200).
    n_populations : int
        Number of SMC generations (default 5).
    epsilon_schedule : list[float] | None
        Explicit tolerance schedule.  If None, ε_1 is set automatically
        from a pilot run and subsequent values decay geometrically by 0.6.
    run_ticks : int
        How many ticks to simulate per particle (default: full year = 730).
    population_size : int
        Agent count passed to SEIRDModel.
    prior : Prior | None
        Custom prior; defaults to Prior().
    on_progress : Callable | None
        Optional callback(generation, n_accepted, n_total) for UI updates.
    verbose : bool
    """

    def __init__(
        self,
        observed_stats: dict[str, float],
        n_particles: int = 200,
        n_populations: int = 5,
        epsilon_schedule: list[float] | None = None,
        run_ticks: int = 730,
        population_size: int = 500,
        prior: Prior | None = None,
        on_progress: Callable | None = None,
        verbose: bool = True,
    ) -> None:
        self.observed_stats  = observed_stats
        self.n_particles     = n_particles
        self.n_populations   = n_populations
        self.run_ticks       = run_ticks
        self.population_size = population_size
        self.prior           = prior or Prior()
        self.on_progress     = on_progress
        self.verbose         = verbose

        self._epsilon_schedule = epsilon_schedule
        self._rng = np.random.default_rng()

        self.populations: list[Population] = []

    # ── Simulation wrapper ────────────────────────────────────────────────

    def _simulate_and_summarise(
        self,
        theta: dict[str, float],
        seed: int,
    ) -> dict[str, float]:
        """
        Run a single forward simulation with parameters theta and return
        its summary statistics.
        """
        SEIRDModel = _import_model()

        # Patch module-level parameters inside submodels at runtime.
        # We do this by temporarily overriding the config values.
        import config.parameters as params  # noqa: PLC0415
        import submodels.transmission as tx  # noqa: PLC0415
        import submodels.disease_progression as dp  # noqa: PLC0415

        orig = {
            "BASE_TRANSMISSION_RATE":  params.BASE_TRANSMISSION_RATE,
            "ASYMPTOMATIC_FRACTION":   params.ASYMPTOMATIC_FRACTION,
            "INCUBATION_PERIOD_TICKS": params.INCUBATION_PERIOD_TICKS,
            "INFECTIOUS_PERIOD_TICKS": params.INFECTIOUS_PERIOD_TICKS,
        }

        try:
            # Override config
            params.BASE_TRANSMISSION_RATE  = theta["base_transmission_rate"]
            params.ASYMPTOMATIC_FRACTION   = theta["asymptomatic_fraction"]
            params.INCUBATION_PERIOD_TICKS = int(round(theta["incubation_period_ticks"]))
            params.INFECTIOUS_PERIOD_TICKS = int(round(theta["infectious_period_ticks"]))

            # Also patch the module-level defaults in submodels that
            # captured the value at import time via default arguments
            tx.spatial_transmission.__defaults__  = (params.BASE_TRANSMISSION_RATE,)
            tx.commute_transmission.__defaults__  = (params.BASE_TRANSMISSION_RATE,)

            model = SEIRDModel(
                population_size=self.population_size,
                random_seed=seed,
            )
            model.run(self.run_ticks)
            df = model.get_results_df()

        finally:
            # Always restore originals
            params.BASE_TRANSMISSION_RATE  = orig["BASE_TRANSMISSION_RATE"]
            params.ASYMPTOMATIC_FRACTION   = orig["ASYMPTOMATIC_FRACTION"]
            params.INCUBATION_PERIOD_TICKS = orig["INCUBATION_PERIOD_TICKS"]
            params.INFECTIOUS_PERIOD_TICKS = orig["INFECTIOUS_PERIOD_TICKS"]
            tx.spatial_transmission.__defaults__  = (orig["BASE_TRANSMISSION_RATE"],)
            tx.commute_transmission.__defaults__  = (orig["BASE_TRANSMISSION_RATE"],)

        return compute_summary_statistics(df, self.population_size)

    # ── Pilot run to set ε_1 ─────────────────────────────────────────────

    def _pilot_epsilon(self, n_pilot: int = 50) -> float:
        """
        Run n_pilot simulations from the prior and set ε_1 as the
        40th percentile of distances (accepts ~60% of pilots).
        """
        if self.verbose:
            print(f"Running {n_pilot} pilot simulations to calibrate ε_1 …")

        distances = []
        for i in range(n_pilot):
            theta = self.prior.sample(self._rng)
            seed  = int(self._rng.integers(0, 2**31))
            stats = self._simulate_and_summarise(theta, seed)
            d     = normalised_euclidean_distance(stats, self.observed_stats)
            distances.append(d)

        eps1 = float(np.percentile(distances, 40))
        if self.verbose:
            print(f"  ε_1 set to {eps1:.4f} (40th pct of pilot distances)")
        return eps1

    # ── Generation 1: rejection ABC ──────────────────────────────────────

    def _run_generation_1(self, epsilon: float) -> Population:
        accepted: list[Particle] = []
        n_sims = 0
        t0 = time.time()

        if self.verbose:
            print(f"\n[Gen 1]  ε = {epsilon:.4f}  target = {self.n_particles} particles")

        while len(accepted) < self.n_particles:
            theta = self.prior.sample(self._rng)
            seed  = int(self._rng.integers(0, 2**31))
            stats = self._simulate_and_summarise(theta, seed)
            n_sims += 1
            d = normalised_euclidean_distance(stats, self.observed_stats)

            if d < epsilon:
                accepted.append(Particle(theta=theta, weight=1.0 / self.n_particles))

            if self.on_progress:
                self.on_progress(1, len(accepted), n_sims)

            if self.verbose and n_sims % 50 == 0:
                rate = len(accepted) / n_sims * 100
                elapsed = time.time() - t0
                print(f"  sims={n_sims:5d}  accepted={len(accepted):4d}  "
                      f"rate={rate:.1f}%  elapsed={elapsed:.1f}s")

        pop = Population(
            particles=accepted,
            epsilon=epsilon,
            generation=1,
            n_simulations_run=n_sims,
        )
        pop.normalise_weights()
        if self.verbose:
            print(f"  ✓ Gen 1 complete: {n_sims} sims, "
                  f"ESS={pop.effective_sample_size():.1f}")
        return pop

    # ── Subsequent generations: importance-weighted perturbation ─────────

    def _run_generation_t(
        self,
        prev_population: Population,
        epsilon: float,
        generation: int,
    ) -> Population:
        bandwidths = _compute_bandwidths(prev_population)
        prev_weights = np.array([p.weight for p in prev_population.particles])

        accepted: list[Particle] = []
        n_sims = 0
        t0 = time.time()

        if self.verbose:
            print(f"\n[Gen {generation}]  ε = {epsilon:.4f}  "
                  f"target = {self.n_particles} particles")

        while len(accepted) < self.n_particles:
            # Sample a particle from the previous population
            idx = self._rng.choice(len(prev_population.particles), p=prev_weights)
            parent = prev_population.particles[idx]

            # Perturb
            theta = _perturb_particle(parent, bandwidths, self.prior, self._rng)
            seed  = int(self._rng.integers(0, 2**31))
            stats = self._simulate_and_summarise(theta, seed)
            n_sims += 1
            d = normalised_euclidean_distance(stats, self.observed_stats)

            if d < epsilon:
                w = _compute_weight(theta, prev_population, bandwidths, self.prior)
                accepted.append(Particle(theta=theta, weight=w))

            if self.on_progress:
                self.on_progress(generation, len(accepted), n_sims)

            if self.verbose and n_sims % 50 == 0:
                rate = len(accepted) / n_sims * 100
                elapsed = time.time() - t0
                print(f"  sims={n_sims:5d}  accepted={len(accepted):4d}  "
                      f"rate={rate:.1f}%  elapsed={elapsed:.1f}s")

        pop = Population(
            particles=accepted,
            epsilon=epsilon,
            generation=generation,
            n_simulations_run=n_sims,
        )
        pop.normalise_weights()
        if self.verbose:
            print(f"  ✓ Gen {generation} complete: {n_sims} sims, "
                  f"ESS={pop.effective_sample_size():.1f}")
        return pop

    # ── Main entry point ─────────────────────────────────────────────────

    def run(self) -> Population:
        """
        Execute the full ABC-SMC calibration and return the final
        posterior population.
        """
        # Build ε schedule
        if self._epsilon_schedule is not None:
            epsilons = self._epsilon_schedule
        else:
            eps1 = self._pilot_epsilon()
            decay = 0.60
            epsilons = [eps1 * (decay ** t) for t in range(self.n_populations)]

        if self.verbose:
            print(f"\nABC-SMC calibration starting")
            print(f"  n_particles   = {self.n_particles}")
            print(f"  n_populations = {self.n_populations}")
            print(f"  ε schedule    = {[f'{e:.4f}' for e in epsilons]}")

        total_t0 = time.time()

        # Generation 1
        pop = self._run_generation_1(epsilons[0])
        self.populations.append(pop)

        # Generations 2 … T
        for t in range(1, self.n_populations):
            pop = self._run_generation_t(pop, epsilons[t], generation=t + 1)
            self.populations.append(pop)

        total_elapsed = time.time() - total_t0
        if self.verbose:
            total_sims = sum(p.n_simulations_run for p in self.populations)
            print(f"\nABC-SMC complete: {total_sims} total simulations "
                  f"in {total_elapsed:.1f}s")

        return self.populations[-1]

    # ── Posterior summaries ───────────────────────────────────────────────

    def posterior_summary(self, population: Population | None = None) -> pd.DataFrame:
        """
        Return weighted posterior mean, std, and credible interval for
        each parameter.
        """
        pop = population or (self.populations[-1] if self.populations else None)
        if pop is None:
            raise RuntimeError("No populations available; run() first.")

        df = pop.as_dataframe()
        w  = df["weight"].values
        rows = []
        for param in self.prior.param_names:
            vals = df[param].values
            mean = np.average(vals, weights=w)
            std  = np.sqrt(np.average((vals - mean) ** 2, weights=w))
            # 95% credible interval via weighted quantiles
            sorted_idx = np.argsort(vals)
            cumw = np.cumsum(w[sorted_idx])
            cumw /= cumw[-1]
            lo = float(vals[sorted_idx][np.searchsorted(cumw, 0.025)])
            hi = float(vals[sorted_idx][np.searchsorted(cumw, 0.975)])
            rows.append({
                "parameter": param,
                "posterior_mean": round(mean, 6),
                "posterior_std":  round(std, 6),
                "ci_2.5%":        round(lo, 6),
                "ci_97.5%":       round(hi, 6),
            })

        return pd.DataFrame(rows).set_index("parameter")

    def map_estimate(self, population: Population | None = None) -> dict[str, float]:
        """Return the Maximum A Posteriori particle (highest weight)."""
        pop = population or self.populations[-1]
        best = max(pop.particles, key=lambda p: p.weight)
        return dict(best.theta)

    # ── Plotting ─────────────────────────────────────────────────────────

    def plot_posterior(
        self,
        population: Population | None = None,
        save_path: str | None = None,
    ) -> None:
        """
        Corner-style marginal posterior plots for each parameter.
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        pop = population or self.populations[-1]
        df  = pop.as_dataframe()
        w   = df["weight"].values
        params = self.prior.param_names

        fig, axes = plt.subplots(1, len(params), figsize=(4 * len(params), 4))
        if len(params) == 1:
            axes = [axes]

        labels = {
            "base_transmission_rate":  "β (transmission rate)",
            "asymptomatic_fraction":   "σ_a (asymptomatic fraction)",
            "incubation_period_ticks": "t_inc (ticks)",
            "infectious_period_ticks": "t_inf (ticks)",
        }

        for ax, param in zip(axes, params):
            vals = df[param].values
            # Weighted histogram
            ax.hist(vals, bins=30, weights=w * len(w), density=True,
                    color="#4C72B0", alpha=0.75, edgecolor="white")
            mean = np.average(vals, weights=w)
            ax.axvline(mean, color="#DD4444", linewidth=1.5, label=f"Mean: {mean:.4f}")
            ax.set_xlabel(labels.get(param, param), fontsize=9)
            ax.set_ylabel("Posterior density", fontsize=9)
            ax.legend(fontsize=8)

        # ε convergence across generations
        if len(self.populations) > 1:
            gen_nums = [p.generation for p in self.populations]
            epsilons = [p.epsilon    for p in self.populations]
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            ax2.plot(gen_nums, epsilons, "o-", color="#4C72B0")
            ax2.set_xlabel("SMC generation")
            ax2.set_ylabel("Tolerance ε")
            ax2.set_title("ABC-SMC tolerance schedule")
            fig2.tight_layout()
            if save_path:
                fig2.savefig(save_path.replace(".png", "_epsilon.png"), dpi=150)

        fig.suptitle(
            f"ABC-SMC Posterior  (Gen {pop.generation}, "
            f"ε = {pop.epsilon:.4f}, N = {len(pop.particles)} particles)",
            fontsize=11,
        )
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def plot_prior_vs_posterior(
        self,
        population: Population | None = None,
        n_prior_samples: int = 2000,
        save_path: str | None = None,
    ) -> None:
        """
        Overlay prior and posterior marginals to visualise Bayesian update.
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        pop    = population or self.populations[-1]
        df     = pop.as_dataframe()
        w      = df["weight"].values
        params = self.prior.param_names

        rng_plot = np.random.default_rng(0)
        prior_samples = pd.DataFrame([self.prior.sample(rng_plot) for _ in range(n_prior_samples)])

        fig, axes = plt.subplots(1, len(params), figsize=(4 * len(params), 4))
        if len(params) == 1:
            axes = [axes]

        for ax, param in zip(axes, params):
            post_vals  = df[param].values
            prior_vals = prior_samples[param].values
            lo = min(prior_vals.min(), post_vals.min())
            hi = max(prior_vals.max(), post_vals.max())
            bins = np.linspace(lo, hi, 30)
            ax.hist(prior_vals, bins=bins, density=True, alpha=0.40,
                    color="grey", label="Prior")
            ax.hist(post_vals, bins=bins, weights=w * len(w), density=True,
                    alpha=0.65, color="#4C72B0", label="Posterior")
            ax.set_xlabel(param, fontsize=9)
            ax.legend(fontsize=8)

        fig.suptitle("Prior vs Posterior", fontsize=11)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()
