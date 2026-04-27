# model.py
"""
Main simulation engine for the SEIRD ABM.

Implements the full tick-level simulation schedule from the ODD protocol:
- Morning tick: commute, work/home transmission, doctor visits
- Evening tick: return commute, home transmission, daily ministry report
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

from utils.initialization import (
    build_grid,
    build_agents,
    seed_epidemic,
    build_health_ministry,
    run_sanity_checks,
)
from utils.data_collector import DataCollector
from submodels.seasonality import seasonal_multiplier
from submodels.mobility import morning_commute, evening_commute
from submodels.transmission import spatial_transmission, commute_transmission
from submodels.disease_progression import apply_disease_progression
from submodels.medical import apply_medical_consultation, daily_report_to_ministry
from policy.health_policy import policy_step

from config.parameters import POPULATION_SIZE, TOTAL_TICKS


class SEIRDModel:
    """
    Main orchestrator for the SEIRD ABM.
    """

    def __init__(
        self,
        population_size: int = POPULATION_SIZE,
        random_seed: int = 42,
    ) -> None:
        self.population_size = population_size
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        self.tick = 0
        self.grid = build_grid()
        self.individuals = build_agents(
            grid=self.grid,
            rng=self.rng,
            population_size=population_size,
        )
        seed_epidemic(self.individuals, self.rng)
        self.health_ministry = build_health_ministry(population_size=population_size)

        run_sanity_checks(self.individuals, self.grid, self.health_ministry)

        self.data_collector = DataCollector()
        self.last_confirmed_today = 0

        self.collect_observation(
            new_exposed=0,
            new_infectious=0,
            new_recovered=0,
            new_dead=0,
            confirmed_today=0,
        )

    @property
    def is_morning_tick(self) -> bool:
        return self.tick % 2 == 0

    @property
    def is_evening_tick(self) -> bool:
        return self.tick % 2 == 1

    @property
    def current_day(self) -> int:
        return self.tick // 2

    def step(self) -> None:
        """
        Run a single 12-hour tick.
        """
        policy_step(self.health_ministry)
        seasonal_mod = seasonal_multiplier(self.tick)

        new_exposed = 0
        new_infectious = 0
        new_recovered = 0
        new_dead = 0
        confirmed_today = 0

        if self.is_morning_tick:
            commuters = morning_commute(
                self.individuals,
                self.grid,
                self.health_ministry,
                self.rng,
            )

            new_exposed += commute_transmission(
                commuters,
                self.health_ministry,
                seasonal_mod,
                self.rng,
            )

            new_exposed += spatial_transmission(
                self.individuals,
                self.grid,
                self.health_ministry,
                seasonal_mod,
                self.rng,
            )

            confirmed_today = apply_medical_consultation(
                self.individuals,
                self.health_ministry,
                self.rng,
            )
            self.last_confirmed_today = confirmed_today

        else:
            returnees = evening_commute(
                self.individuals,
                self.grid,
            )

            new_exposed += commute_transmission(
                returnees,
                self.health_ministry,
                seasonal_mod,
                self.rng,
            )

            new_exposed += spatial_transmission(
                self.individuals,
                self.grid,
                self.health_ministry,
                seasonal_mod,
                self.rng,
            )

            daily_report_to_ministry(
                self.health_ministry,
                self.last_confirmed_today,
            )
            confirmed_today = self.last_confirmed_today
            self.last_confirmed_today = 0

        progression = apply_disease_progression(
            self.individuals,
            self.rng,
        )

        new_infectious = progression["new_infectious"]
        new_recovered = progression["new_recovered"]
        new_dead = progression["new_dead"]

        self.tick += 1

        self.collect_observation(
            new_exposed=new_exposed,
            new_infectious=new_infectious,
            new_recovered=new_recovered,
            new_dead=new_dead,
            confirmed_today=confirmed_today,
        )

    def run(self, n_ticks: int) -> None:
        """
        Run the model for n ticks.
        """
        for _ in range(n_ticks):
            self.step()

    def run_until_end(self) -> None:
        """
        Run until TOTAL_TICKS is reached.
        """
        remaining = TOTAL_TICKS - self.tick
        if remaining > 0:
            self.run(remaining)

    def collect_observation(
        self,
        new_exposed: int,
        new_infectious: int,
        new_recovered: int,
        new_dead: int,
        confirmed_today: int,
    ) -> None:
        self.data_collector.collect(
            tick=self.tick,
            individuals=self.individuals,
            health_ministry=self.health_ministry,
            new_exposed=new_exposed,
            new_infectious=new_infectious,
            new_recovered=new_recovered,
            new_dead=new_dead,
            confirmed_today=confirmed_today,
        )

    def get_results_df(self) -> pd.DataFrame:
        return self.data_collector.to_dataframe()

    def export_results_csv(self, path: str = "data/outputs/simulation_results.csv") -> str:
        df = self.get_results_df()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        return path
