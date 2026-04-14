# agents/health_ministry.py
"""
Health Ministry agent – the single government entity in the model.
State variables from ODD Protocol §2.1.
"""

from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass, field
from collections import deque


class AlertState(Enum):
    NO_ALERT = auto()   # no active containment measures
    ALERT_1  = auto()   # low-coercive measures
    ALERT_2  = auto()   # medium-coercive measures
    ALERT_3  = auto()   # high-coercive measures


@dataclass
class HealthMinistry:
    """
    Singleton agent representing the national Health Ministry.

    Responsibilities (ODD §3, §7.5)
    ──────────────────────────────────
    - Receives daily reports: number of confirmed infected individuals
      (from doctor consultations)
    - Maintains a 7-day rolling window of prevalence
    - Activates / escalates alert plans when thresholds are crossed
    - Cannot downgrade a plan before MIN_POLICY_DURATION_DAYS days have elapsed
    """

    # ── Policy state ─────────────────────────────────────────────────────────
    alert_state:              AlertState = AlertState.NO_ALERT
    ticks_in_current_policy:  int        = 0   # ticks since last escalation
    min_policy_duration_ticks: int       = 28  # 14 days × 2 ticks/day

    # ── Surveillance data ─────────────────────────────────────────────────────
    # Rolling 7-day confirmed-case count (updated once per day = every 2 ticks)
    confirmed_daily_cases: deque = field(
        default_factory=lambda: deque([0] * 7, maxlen=7)
    )
    total_confirmed_cases: int = 0

    # Thresholds (fraction of total population)
    threshold_alert_1: float = 0.02
    threshold_alert_2: float = 0.05
    threshold_alert_3: float = 0.10
    population_size:   int   = 5_000

    # ── Surveillance methods ──────────────────────────────────────────────────

    def receive_daily_report(self, new_cases: int) -> None:
        """Called once per day with freshly reported confirmed cases."""
        self.confirmed_daily_cases.append(new_cases)
        self.total_confirmed_cases += new_cases

    @property
    def seven_day_prevalence(self) -> float:
        """7-day rolling prevalence as a fraction of population."""
        return sum(self.confirmed_daily_cases) / self.population_size

    # ── Policy decision ───────────────────────────────────────────────────────

    def update_policy(self) -> AlertState:
        """
        Called once per day. Escalates or maintains alert state based on
        7-day prevalence. Downgrading is only allowed after
        min_policy_duration_ticks ticks have elapsed.
        """
        prev        = self.alert_state
        can_change  = self.ticks_in_current_policy >= self.min_policy_duration_ticks
        prevalence  = self.seven_day_prevalence

        if prevalence >= self.threshold_alert_3:
            target = AlertState.ALERT_3
        elif prevalence >= self.threshold_alert_2:
            target = AlertState.ALERT_2
        elif prevalence >= self.threshold_alert_1:
            target = AlertState.ALERT_1
        else:
            target = AlertState.NO_ALERT

        # Escalation is always immediate; downgrading requires waiting period
        is_escalation = target.value > prev.value
        if is_escalation or can_change:
            if target != prev:
                self.alert_state             = target
                self.ticks_in_current_policy = 0

        self.ticks_in_current_policy += 1
        return self.alert_state

    # ── repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"HealthMinistry(state={self.alert_state.name}, "
            f"7d_prevalence={self.seven_day_prevalence:.2%}, "
            f"total_confirmed={self.total_confirmed_cases})"
        )
