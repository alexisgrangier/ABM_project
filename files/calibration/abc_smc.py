# calibration/abc_smc.py
"""
Approximate Bayesian Computation – Rejection ABC (single generation).

Simplified for reliability: no pilot run, no SMC generations, 
fixed epsilon based on percentile of first batch of simulations.

Parameters calibrated
---------------------
    BASE_TRANSMISSION_RATE   (β)
    ASYMPTOMATIC_FRACTION    (σ_a)
    INCUBATION_PERIOD_TICKS  (t_inc)
    INFECTIOUS_PERIOD_TICKS  (t_inf)

Summary statistics
------------------
    peak_infectious   : max simultaneous infectious count
    peak_timing_tick  : tick at which peak occurs
    total_dead        : cumulative deaths at end
    attack_rate       : fraction of population ever infected
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class Prior:
    beta_low:  float = 0.005
    beta_high: float = 0.10

    asym_low:  float = 0.10
    asym_high: float = 0.70

    t_inc_low:  float = 2.0
    t_inc_high: float = 10.0

    t_inf_low:  float = 4.0
    t_inf_high: float = 20.0

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

    def in_support(self, theta: dict[str, float]) -> bool:
        return all([
            self.beta_low   <= theta["base_transmission_rate"]  <= self.beta_high,
            self.asym_low   <= theta["asymptomatic_fraction"]   <= self.asym_high,
            self.t_inc_low  <= theta["incubation_period_ticks"] <= self.t_inc_high,
            self.t_inf_low  <= theta["infectious_period_ticks"] <= self.t_inf_high,
        ])


@dataclass
class Particle:
    theta:  dict[str, float]
    weight: float = 1.0


@dataclass
class Population:
    particles:         list[Particle]
    epsilon:           float
    generation:        int
    n_simulations_run: int = 0

    def normalise_weights(self) -> None:
        total = sum(p.weight for p in self.particles)
        for p in self.particles:
            p.weight /= total

    def effective_sample_size(self) -> float:
        w = np.array([p.weight for p in self.particles])
        w = w / w.sum()
        return float(1.0 / np.sum(w ** 2))

    def as_dataframe(self) -> pd.DataFrame:
        rows = [{**p.theta, "weight": p.weight} for p in self.particles]
        return pd.DataFrame(rows)


def compute_summary_statistics(
    df: pd.DataFrame,
    population_size: int,
) -> dict[str, float]:
    if df.empty:
        return {"peak_infectious": 0.0, "peak_timing_tick": 0.0,
                "total_dead": 0.0, "attack_rate": 0.0}

    col = "total_infectious"
    if col not in df.columns:
        candidates = [c for c in df.columns if "infectious" in c.lower()]
        col = candidates[0] if candidates else None

    peak_infectious  = float(df[col].max()) if col else 0.0
    peak_timing_tick = float(df.loc[df[col].idxmax(), "tick"]) if col else 0.0
    total_dead       = float(df["dead"].iloc[-1]) if "dead" in df.columns else 0.0

    if "susceptible" in df.columns:
        final_susc = float(df["susceptible"].iloc[-1])
        attack_rate = max(0.0, 1.0 - (final_susc / population_size))
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
) -> float:
    total = 0.0
    for k, obs_val in observed.items():
        scale = abs(obs_val) if obs_val != 0 else 1.0
        total += ((simulated.get(k, 0.0) - obs_val) / scale) ** 2
    return float(np.sqrt(total))


class ABCSMCCalibrator:
    """
    Single-generation rejection ABC calibrator.

    1. Runs n_calibration simulations from prior to set epsilon
       (target_percentile of observed distances).
    2. Runs rejection ABC until n_particles accepted.
    """

    def __init__(
        self,
        observed_stats:    dict[str, float],
        n_particles:       int      = 50,
        n_populations:     int      = 1,
        epsilon_schedule:  list     = None,
        run_ticks:         int      = 100,
        population_size:   int      = 500,
        prior:             Prior    = None,
        on_progress:       Callable = None,
        verbose:           bool     = True,
        target_percentile: float    = 30.0,
        n_calibration:     int      = 30,
    ) -> None:
        self.observed_stats    = observed_stats
        self.n_particles       = n_particles
        self.run_ticks         = run_ticks
        self.population_size   = population_size
        self.prior             = prior or Prior()
        self.on_progress       = on_progress
        self.verbose           = verbose
        self.target_percentile = target_percentile
        self.n_calibration     = n_calibration
        self._rng              = np.random.default_rng()
        self.populations: list[Population] = []

    def _simulate(self, theta: dict[str, float], seed: int) -> dict[str, float]:
        from model import SEIRDModel
        import config.parameters as params
        import submodels.transmission as tx

        orig = {
            "BASE_TRANSMISSION_RATE":  params.BASE_TRANSMISSION_RATE,
            "ASYMPTOMATIC_FRACTION":   params.ASYMPTOMATIC_FRACTION,
            "INCUBATION_PERIOD_TICKS": params.INCUBATION_PERIOD_TICKS,
            "INFECTIOUS_PERIOD_TICKS": params.INFECTIOUS_PERIOD_TICKS,
        }
        try:
            params.BASE_TRANSMISSION_RATE  = float(theta["base_transmission_rate"])
            params.ASYMPTOMATIC_FRACTION   = float(theta["asymptomatic_fraction"])
            params.INCUBATION_PERIOD_TICKS = int(round(theta["incubation_period_ticks"]))
            params.INFECTIOUS_PERIOD_TICKS = int(round(theta["infectious_period_ticks"]))
            tx.spatial_transmission.__defaults__ = (params.BASE_TRANSMISSION_RATE,)
            tx.commute_transmission.__defaults__ = (params.BASE_TRANSMISSION_RATE,)

            model = SEIRDModel(population_size=self.population_size, random_seed=seed)
            model.run(self.run_ticks)
            df = model.get_results_df()
        finally:
            params.BASE_TRANSMISSION_RATE  = orig["BASE_TRANSMISSION_RATE"]
            params.ASYMPTOMATIC_FRACTION   = orig["ASYMPTOMATIC_FRACTION"]
            params.INCUBATION_PERIOD_TICKS = orig["INCUBATION_PERIOD_TICKS"]
            params.INFECTIOUS_PERIOD_TICKS = orig["INFECTIOUS_PERIOD_TICKS"]
            tx.spatial_transmission.__defaults__ = (orig["BASE_TRANSMISSION_RATE"],)
            tx.commute_transmission.__defaults__ = (orig["BASE_TRANSMISSION_RATE"],)

        return compute_summary_statistics(df, self.population_size)

    def _calibrate_epsilon(self) -> float:
        if self.verbose:
            print(f"Calibrating ε from {self.n_calibration} simulations ...")
        distances = []
        for i in range(self.n_calibration):
            theta = self.prior.sample(self._rng)
            seed  = int(self._rng.integers(0, 2**31))
            stats = self._simulate(theta, seed)
            d     = normalised_euclidean_distance(stats, self.observed_stats)
            distances.append(d)
            if self.verbose:
                print(f"  calibration {i+1}/{self.n_calibration}  d={d:.3f}")
        epsilon = float(np.percentile(distances, self.target_percentile))
        if self.verbose:
            print(f"  → ε = {epsilon:.4f}\n")
        return epsilon

    def run(self) -> Population:
        epsilon  = self._calibrate_epsilon()
        accepted: list[Particle] = []
        n_sims   = 0
        t0       = time.time()

        if self.verbose:
            print(f"[ABC]  ε={epsilon:.4f}  target={self.n_particles} particles")

        while len(accepted) < self.n_particles:
            theta = self.prior.sample(self._rng)
            seed  = int(self._rng.integers(0, 2**31))
            stats = self._simulate(theta, seed)
            n_sims += 1
            d = normalised_euclidean_distance(stats, self.observed_stats)

            if d < epsilon:
                accepted.append(Particle(theta=theta, weight=1.0))

            if self.on_progress:
                self.on_progress(1, len(accepted), n_sims)

            if self.verbose and n_sims % 10 == 0:
                rate    = len(accepted) / n_sims * 100
                elapsed = time.time() - t0
                print(f"  sims={n_sims:4d}  accepted={len(accepted):3d}  "
                      f"rate={rate:.1f}%  elapsed={elapsed:.1f}s")

        pop = Population(
            particles=accepted,
            epsilon=epsilon,
            generation=1,
            n_simulations_run=n_sims,
        )
        pop.normalise_weights()
        self.populations = [pop]

        if self.verbose:
            print(f"\n✓ ABC complete: {n_sims} sims in {time.time()-t0:.1f}s  "
                  f"ESS={pop.effective_sample_size():.1f}")
        return pop

    def posterior_summary(self, population: Population = None) -> pd.DataFrame:
        pop = population or (self.populations[-1] if self.populations else None)
        if pop is None:
            raise RuntimeError("Run calibration first.")
        df = pop.as_dataframe()
        w  = df["weight"].values
        w  = w / w.sum()
        rows = []
        for param in self.prior.param_names:
            vals = df[param].values
            mean = float(np.average(vals, weights=w))
            std  = float(np.sqrt(np.average((vals - mean) ** 2, weights=w)))
            si   = np.argsort(vals)
            cw   = np.cumsum(w[si]); cw /= cw[-1]
            lo   = float(vals[si][np.searchsorted(cw, 0.025)])
            hi   = float(vals[si][np.searchsorted(cw, 0.975)])
            rows.append({
                "parameter":      param,
                "posterior_mean": round(mean, 6),
                "posterior_std":  round(std,  6),
                "ci_2.5%":        round(lo,   6),
                "ci_97.5%":       round(hi,   6),
            })
        return pd.DataFrame(rows).set_index("parameter")

    def map_estimate(self, population: Population = None) -> dict[str, float]:
        pop = population or self.populations[-1]
        return dict(max(pop.particles, key=lambda p: p.weight).theta)