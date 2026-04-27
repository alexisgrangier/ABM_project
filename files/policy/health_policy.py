# policy/health_policy.py
"""
The HealthMinistry agent is defined in agents/health_ministry.py.
This module provides the step-level hook called by the simulation engine
and any helper functions for querying current policy state.
"""

from __future__ import annotations

from agents.health_ministry import HealthMinistry, AlertState
from config.parameters import POLICY_COMMUTE_REDUCTION


def get_commute_reduction(health_ministry: HealthMinistry) -> float:
    """
    Return the current policy-driven commute reduction fraction.

    Used by the mobility submodel to scale individual compliance.
    """
    mapping = {
        AlertState.NO_ALERT: POLICY_COMMUTE_REDUCTION["no_alert"],
        AlertState.ALERT_1:  POLICY_COMMUTE_REDUCTION["alert_1"],
        AlertState.ALERT_2:  POLICY_COMMUTE_REDUCTION["alert_2"],
        AlertState.ALERT_3:  POLICY_COMMUTE_REDUCTION["alert_3"],
    }
    return mapping[health_ministry.alert_state]


def policy_step(health_ministry: HealthMinistry) -> AlertState:
    """
    Advance the ministry's policy counter by one tick.
    Called every tick by the simulation engine.

    Note: actual escalation/de-escalation happens in
    daily_report_to_ministry() (medical.py) once per day.
    Here we only increment the duration counter.
    """
    health_ministry.ticks_in_current_policy += 1
    return health_ministry.alert_state
