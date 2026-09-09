"""utils_traj_simple.py

Minimal alternative to `utils_traj`'s quartic-fit reference generator for
`erhu_env.desired_velocity_and_pressure`: a sine-wave velocity command as a
function of *time* (not bow position) plus a constant pressure, with the
sine's period and the pressure level each drawn once per episode from a
uniform interval.

    velocity(t) = v_limit * sin(2*pi*t / period),   period ~ U(period_range)
    pressure    = p        ,                        p      ~ U(p_min, p_max)

Unlike `utils_traj`, there is no position-based segment/resample state:
`query_traj` here is a pure function of `t` and the two per-episode
constants carried in `info`, so it does not need to be called every step to
stay consistent (no `maybe_resample` equivalent).
"""

from typing import Dict, Tuple

import jax
import jax.numpy as jp


def init_traj_info(
    rng: jax.Array,
    p_min: float,
    p_max: float,
    period_range: Tuple[float, float],
) -> Dict[str, jax.Array]:
    """Samples this episode's sine period and constant pressure level.

    `period_range`: (min, max) seconds for the velocity sine's period.
    `p_min`/`p_max`: interval the constant pressure target is drawn from.
    """
    k_period, k_pressure = jax.random.split(rng)
    period = jax.random.uniform(
        k_period, (), minval=period_range[0], maxval=period_range[1]
    )
    pressure = jax.random.uniform(k_pressure, (), minval=p_min, maxval=p_max)
    return {
        "traj_period": period,
        "traj_pressure": pressure,
    }


def query_traj(
    info: Dict[str, jax.Array], t: jax.Array, v_limit: float, p_min: float, p_max: float
) -> Tuple[jax.Array, jax.Array]:
    """Evaluates the scripted reference at time `t` (e.g. `data.time`).

    Returns `(velocity, pressure)`: velocity is a sine wave of amplitude
    `v_limit` and period `info["traj_period"]`; pressure is the constant
    `info["traj_pressure"]` sampled at `init_traj_info`, clipped to
    [p_min, p_max] as a safety net.
    """
    phase = 2.0 * jp.pi * t / info["traj_period"]
    velocity = v_limit * jp.sin(phase)
    pressure = jp.clip(info["traj_pressure"], p_min, p_max)
    return velocity, pressure
