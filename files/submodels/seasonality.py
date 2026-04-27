# submodels/seasonality.py
"""
Seasonal transmission modifier (ODD §7.3).

Transmission peaks in Winter (tick 0 = 1 Jan) and troughs in Summer,
mirroring the seasonal profile of European influenza.
"""

from __future__ import annotations
import numpy as np
from config.parameters import SEASONAL_AMPLITUDE, TICKS_PER_DAY


def seasonal_multiplier(tick: int, amplitude: float = SEASONAL_AMPLITUDE) -> float:
    """
    Return a multiplicative modifier for the base transmission rate.

    Uses a cosine curve so that:
      - tick 0   (1 Jan, Winter)  → maximum  (1 + amplitude)
      - tick 365 (1 Jul, Summer)  → minimum  (1 - amplitude)

    Parameters
    ----------
    tick      : current simulation tick (0-indexed)
    amplitude : seasonal swing (default ±0.30 from parameters.py)

    Returns
    -------
    float in [1 - amplitude, 1 + amplitude]
    """
    ticks_per_year = 365 * TICKS_PER_DAY                  # 730
    angle          = 2 * np.pi * tick / ticks_per_year    # full year cycle
    return 1.0 + amplitude * np.cos(angle)
