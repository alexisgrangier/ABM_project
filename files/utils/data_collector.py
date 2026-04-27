# utils/data_collector.py
"""
Observation and output collection (ODD §4.8).

Collects aggregate SEIRD counts, confirmed cases, and policy state
at every tick. Exports to pandas DataFrame for analysis and Streamlit.
"""

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

from agents.individual import Individual, EpiState
from agents.health_ministry import HealthMinistry


@dataclass
class TickSnapshot:
    """One row of simulation output."""
    tick: int
    day: float
    susceptible: int
    exposed: int
    infectious_asymp: int
    infectious_symp: int
    recovered: int
    dead: int
    total_infectious: int
    total_living: int
    new_exposed: int
    new_infectious: int
    new_recovered: int
    new_dead: int
    confirmed_cases: int
    alert_state: str
    seven_day_prev: float


class DataCollector:
    """Collects simulation outputs at each tick."""

    def __init__(self) -> None:
        self._history: list[TickSnapshot] = []

    def _count_states(self, individuals: list[Individual]) -> dict[EpiState, int]:
        counts = {state: 0 for state in EpiState}
        for agent in individuals:
            counts[agent.epi_state] += 1
        return counts

    def collect(
        self,
        tick: int,
        individuals: list[Individual],
        health_ministry: HealthMinistry,
        new_exposed: int = 0,
        new_infectious: int = 0,
        new_recovered: int = 0,
        new_dead: int = 0,
        confirmed_today: int = 0,
    ) -> None:
        """Record one simulation snapshot."""
        counts = self._count_states(individuals)

        snapshot = TickSnapshot(
            tick=tick,
            day=tick / 2.0,
            susceptible=counts[EpiState.SUSCEPTIBLE],
            exposed=counts[EpiState.EXPOSED],
            infectious_asymp=counts[EpiState.INFECTIOUS_ASYMPTOMATIC],
            infectious_symp=counts[EpiState.INFECTIOUS_SYMPTOMATIC],
            recovered=counts[EpiState.RECOVERED],
            dead=counts[EpiState.DEAD],
            total_infectious=(
                counts[EpiState.INFECTIOUS_ASYMPTOMATIC]
                + counts[EpiState.INFECTIOUS_SYMPTOMATIC]
            ),
            total_living=(
                counts[EpiState.SUSCEPTIBLE]
                + counts[EpiState.EXPOSED]
                + counts[EpiState.INFECTIOUS_ASYMPTOMATIC]
                + counts[EpiState.INFECTIOUS_SYMPTOMATIC]
                + counts[EpiState.RECOVERED]
            ),
            new_exposed=new_exposed,
            new_infectious=new_infectious,
            new_recovered=new_recovered,
            new_dead=new_dead,
            confirmed_cases=confirmed_today,
            alert_state=health_ministry.alert_state.name,
            seven_day_prev=health_ministry.seven_day_prevalence,
        )

        self._history.append(snapshot)

    def to_dataframe(self) -> pd.DataFrame:
        """Return collected history as pandas DataFrame."""
        return pd.DataFrame([vars(row) for row in self._history])

    def latest(self) -> TickSnapshot | None:
        """Return the latest snapshot, if present."""
        if not self._history:
            return None
        return self._history[-1]

    def reset(self) -> None:
        """Clear history."""
        self._history = []
