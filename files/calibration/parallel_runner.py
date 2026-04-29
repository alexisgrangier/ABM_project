# calibration/parallel_runner.py
"""
Optional parallel execution wrapper for ABC-SMC.

Uses Python's multiprocessing.Pool to run multiple forward simulations
concurrently, which is the primary bottleneck in ABC calibration.

Usage
-----
    from calibration.parallel_runner import run_abc_parallel

    posterior = run_abc_parallel(
        observed_stats={"peak_infectious": 150, ...},
        n_particles=200,
        n_populations=5,
        n_workers=4,
    )

Note: Only use this from a __main__ guard or a CLI script,
not from Streamlit (Streamlit's process model conflicts with fork-based
multiprocessing on macOS/Windows — use the single-threaded ABCSMCCalibrator
in those contexts).
"""

from __future__ import annotations

import multiprocessing as mp
from functools import partial
from typing import Any

import numpy as np
import pandas as pd

from calibration.abc_smc import (
    ABCSMCCalibrator,
    Prior,
    Particle,
    Population,
    normalised_euclidean_distance,
    _perturb_particle,
    _compute_bandwidths,
    _compute_weight,
)


def _worker_simulate(args: tuple) -> tuple[dict, dict, float]:
    """
    Top-level function (picklable) for multiprocessing.

    Returns (theta, simulated_stats, distance).
    """
    theta, observed_stats, run_ticks, population_size, seed = args

    # Import inside worker to avoid serialisation issues
    from calibration.abc_smc import ABCSMCCalibrator  # noqa

    dummy = ABCSMCCalibrator(
        observed_stats=observed_stats,
        run_ticks=run_ticks,
        population_size=population_size,
    )
    stats = dummy._simulate_and_summarise(theta, seed)
    d = normalised_euclidean_distance(stats, observed_stats)
    return theta, stats, d


def run_abc_parallel(
    observed_stats: dict[str, float],
    n_particles: int = 200,
    n_populations: int = 5,
    run_ticks: int = 730,
    population_size: int = 500,
    n_workers: int | None = None,
    epsilon_schedule: list[float] | None = None,
    verbose: bool = True,
) -> Population:
    """
    Run ABC-SMC with a multiprocessing pool for parallelism.

    Parameters
    ----------
    n_workers : int | None
        Number of parallel workers.  Defaults to cpu_count() - 1.
    """
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    rng    = np.random.default_rng()
    prior  = Prior()

    # ── Epsilon schedule ─────────────────────────────────────────────────
    if epsilon_schedule is None:
        # Quick pilot (single-threaded) to calibrate ε_1
        dummy = ABCSMCCalibrator(
            observed_stats=observed_stats,
            run_ticks=run_ticks,
            population_size=population_size,
            verbose=verbose,
        )
        eps1 = dummy._pilot_epsilon(n_pilot=20)
        epsilon_schedule = [eps1 * (0.60 ** t) for t in range(n_populations)]

    if verbose:
        print(f"Parallel ABC-SMC  workers={n_workers}  "
              f"ε_schedule={[f'{e:.4f}' for e in epsilon_schedule]}")

    populations: list[Population] = []

    with mp.Pool(processes=n_workers) as pool:

        # ── Generation 1: rejection ABC ──────────────────────────────
        eps = epsilon_schedule[0]
        accepted: list[Particle] = []
        n_sims = 0
        batch = n_particles * 10   # over-sample to reduce round-trips

        if verbose:
            print(f"\n[Gen 1]  ε = {eps:.4f}")

        while len(accepted) < n_particles:
            # Sample a batch of candidates from the prior
            batch_args = [
                (prior.sample(rng), observed_stats, run_ticks, population_size,
                 int(rng.integers(0, 2**31)))
                for _ in range(batch)
            ]
            results = pool.map(_worker_simulate, batch_args)
            n_sims += batch

            for theta, _stats, d in results:
                if d < eps and len(accepted) < n_particles:
                    accepted.append(Particle(theta=theta, weight=1.0 / n_particles))

            if verbose:
                print(f"  sims={n_sims}  accepted={len(accepted)}")

        pop = Population(particles=accepted, epsilon=eps, generation=1,
                         n_simulations_run=n_sims)
        pop.normalise_weights()
        populations.append(pop)

        # ── Generations 2 … T ────────────────────────────────────────
        for t in range(1, n_populations):
            eps = epsilon_schedule[t]
            prev_pop   = populations[-1]
            bandwidths = _compute_bandwidths(prev_pop)
            prev_w     = np.array([p.weight for p in prev_pop.particles])

            accepted   = []
            n_sims     = 0

            if verbose:
                print(f"\n[Gen {t+1}]  ε = {eps:.4f}")

            while len(accepted) < n_particles:
                idxs = rng.choice(
                    len(prev_pop.particles), size=batch, p=prev_w
                )
                batch_args = []
                candidate_thetas = []
                for idx in idxs:
                    theta = _perturb_particle(
                        prev_pop.particles[idx], bandwidths, prior, rng
                    )
                    candidate_thetas.append(theta)
                    batch_args.append(
                        (theta, observed_stats, run_ticks, population_size,
                         int(rng.integers(0, 2**31)))
                    )

                results = pool.map(_worker_simulate, batch_args)
                n_sims += batch

                for (theta, _stats, d), _theta in zip(results, candidate_thetas):
                    if d < eps and len(accepted) < n_particles:
                        w = _compute_weight(theta, prev_pop, bandwidths, prior)
                        accepted.append(Particle(theta=theta, weight=w))

                if verbose:
                    print(f"  sims={n_sims}  accepted={len(accepted)}")

            pop = Population(particles=accepted, epsilon=eps, generation=t + 1,
                             n_simulations_run=n_sims)
            pop.normalise_weights()
            populations.append(pop)

    if verbose:
        total = sum(p.n_simulations_run for p in populations)
        print(f"\nParallel ABC-SMC complete: {total} total simulations")

    return populations[-1]


if __name__ == "__main__":
    # Example: calibrate to a reference run
    from model import SEIRDModel
    from calibration.abc_smc import compute_summary_statistics

    print("Generating reference statistics from default model …")
    ref = SEIRDModel(population_size=500, random_seed=42)
    ref.run(730)
    ref_stats = compute_summary_statistics(ref.get_results_df(), 500)
    print("Reference stats:", ref_stats)

    posterior = run_abc_parallel(
        observed_stats=ref_stats,
        n_particles=100,
        n_populations=4,
        run_ticks=730,
        population_size=500,
        n_workers=4,
    )

    from calibration.abc_smc import ABCSMCCalibrator
    dummy_cal = ABCSMCCalibrator(observed_stats=ref_stats)
    dummy_cal.populations = [posterior]
    print(dummy_cal.posterior_summary(posterior))
