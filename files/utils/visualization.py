# utils/visualization.py
"""
Visualization helpers for Streamlit and analysis.
"""

from __future__ import annotations
import pandas as pd


def prepare_epi_curve_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide epidemic state table into long format for plotting.
    """
    state_cols = [
        "susceptible",
        "exposed",
        "infectious_asymp",
        "infectious_symp",
        "recovered",
        "dead",
    ]

    return df.melt(
        id_vars=["tick", "day"],
        value_vars=state_cols,
        var_name="state",
        value_name="count",
    )


def prepare_alert_df(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "NO_ALERT": 0,
        "ALERT_1": 1,
        "ALERT_2": 2,
        "ALERT_3": 3,
    }

    out = df[["tick", "day", "alert_state", "seven_day_prev"]].copy()
    out["alert_level"] = out["alert_state"].map(mapping).fillna(0).astype(int)
    out["seven_day_prev"] = pd.to_numeric(out["seven_day_prev"], errors="coerce").fillna(0.0)
    return out


def latest_grid_positions(individuals: list) -> pd.DataFrame:
    """
    Prepare agent positions for scatter plotting.
    """
    rows = []
    for agent in individuals:
        rows.append(
            {
                "agent_id": agent.agent_id,
                "x": agent.position[0],
                "y": agent.position[1],
                "state": agent.epi_state.name,
                "at_work": agent.at_work,
                "residence": agent.residence.value,
            }
        )
    return pd.DataFrame(rows)
