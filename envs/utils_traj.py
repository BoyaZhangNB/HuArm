"""utils_traj.py

Synthetic reference bow-stroke generator used by
`erhu_env.desired_velocity_and_pressure` as a scripted stand-in for a real
motion-capture + pressure-sensor reference (see the TODO note near the top
of erhu_env.py -- this is exactly the kind of placeholder meant to be
swapped out at inference time).

Unlike the original placeholder (a fixed sine wave of `data.time`), the
velocity/pressure targets here are scripted as functions of the bow's own
normalized lateral position `x`, not of time: `x` in [-1, 1] is the bow's
position along its own length (see `ErhuEnv._lateral_axis_local` /
`_bow_geometry`), 0 at the centre, -1/+1 at the frog/tip ends.

Each "segment" runs the bow from its current position x0 to a freshly
sampled target x, along a randomly-shaped speed/pressure profile:

    - four points fix each profile: (x0, 0), (x, 0), and two more
      randomly-placed points (x1, y1), (x2, y2) shared between the two
      profiles in their x-coordinate (only the sampled y-values --
      speed/pressure -- differ between the two).
    - the velocity profile additionally has to hit a sampled bound on its
      average curvature, `int_{x0}^{x} v'(x)^2 dx / (x - x0) == a_bar`
      (falls back to the smoothest -- minimum-curvature -- quartic through
      the 4 points if `a_bar` turns out to be infeasibly tight). The
      pressure profile just uses that minimum-curvature quartic directly
      (no separate curvature bound is specified for it).

A quartic (degree 4, 5 coefficients) is the lowest-degree polynomial that
can satisfy 4 point constraints and 1 additional (curvature) constraint at
once, hence "fit a quartic" below.

Everything here operates on a single (unbatched) env -- like the rest of
`erhu_env.py`, batching is handled by an outer `vmap` in the training loop.
State is a handful of jax arrays carried as flat `"traj_*"` keys inside
`State.info` (see `init_traj_info` / `maybe_resample`) so it passes through
jit/vmap/scan for free.

Position is *not* carried in `info`: the bow's current normalized position
`x` is supplied by the caller on every `query_traj`/`maybe_resample` call
-- in `erhu_env.py` this is the real, measured bow position (see
`ErhuEnv._bow_stroke_position`), not anything integrated here. `info` only
carries the current segment's shape (`traj_x0`, `traj_target`,
`traj_vcoef`, `traj_pcoef`) plus the RNG key used to resample the next one.
"""

from typing import Dict, Tuple

import jax
import jax.numpy as jp


_N_COEF = 5  # quartic: c0 + c1 x + c2 x^2 + c3 x^3 + c4 x^4

# Bow-hair length is 0.5 m (`bow_hair_geom` in huarm/arm.xml runs from the
# frog site to `fromto="0 0 0  0 0 -0.5"`), so a normalized position step of
# `dx` corresponds to `dx * BOW_HALF_LENGTH` metres of real lateral travel.
BOW_HALF_LENGTH = 0.20

# Minimum guaranteed "creep" speed toward the segment target, as a fraction
# of `v_limit` -- see `query_traj`.
_MIN_CREEP_VEL = 0.02


# ---------------------------------------------------------------------------
# Small quartic-algebra helpers.
# ---------------------------------------------------------------------------

def _vander(x: jax.Array) -> jax.Array:
    """[1, x, x^2, x^3, x^4], ascending powers -- built by repeated
    multiplication rather than `x ** arange(5)` so it stays exact and NaN
    -free for negative `x` (a fractional-looking float exponent on a
    negative base is not safe in general)."""
    one = jp.ones_like(x)
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    return jp.stack([one, x, x2, x3, x4])


def _eval_poly(coef: jax.Array, x: jax.Array) -> jax.Array:
    """coef: (5,) ascending powers -> c0 + c1 x + ... + c4 x^4."""
    return jp.dot(coef, _vander(x))


def _deriv_integral_matrix(xlo: jax.Array, xhi: jax.Array) -> jax.Array:
    """5x5 matrix `M` such that `c @ M @ c == int_xlo^xhi v'(x)^2 dx` for
    `v(x) = sum_k c_k x^k`. Always called with `xlo <= xhi` so `M` is PSD
    (it's the Gram matrix of the derivative basis over a real interval).

    v'(x) = sum_k k c_k x^{k-1}, so
      int v'(x)^2 dx = sum_{i,j} (i j) c_i c_j int x^{i+j-2} dx
                      = sum_{i,j} (i j) c_i c_j [x^{i+j-1} / (i+j-1)]_xlo^xhi
    The i=0 or j=0 terms vanish (factor of i or j is 0) before the
    antiderivative's power could go negative, so those entries are just
    hard-coded to 0 rather than evaluated.
    """
    rows = []
    for i in range(_N_COEF):
        row = []
        for j in range(_N_COEF):
            if i == 0 or j == 0:
                row.append(jp.zeros_like(xlo))
            else:
                p = i + j - 1  # >= 1, concrete python int -> exact integer power
                row.append((i * j) * (xhi ** p - xlo ** p) / p)
        rows.append(jp.stack(row))
    return jp.stack(rows)


def _fit_quartic(
    x0: jax.Array,
    xt: jax.Array,
    x1: jax.Array,
    x2: jax.Array,
    y1: jax.Array,
    y2: jax.Array,
    a_bar,
) -> jax.Array:
    """Fits the quartic v(x0)=v(xt)=0, v(x1)=y1, v(x2)=y2.

    `a_bar`: `None` for the plain minimum-curvature fit (used for the
    pressure profile), or a scalar array target for
    `int_{x0}^{xt} v'(x)^2 dx / (xt - x0) == a_bar` (used for the velocity
    profile). Falls back to the minimum-curvature solution whenever the
    requested `a_bar` is infeasible (below what any quartic through these
    4 points can achieve) or the 4 points are degenerate.
    """
    xs = jp.stack([x0, xt, x1, x2])
    A = jp.stack([_vander(xi) for xi in xs], axis=0)  # (4, 5)
    b = jp.stack([jp.zeros_like(y1), jp.zeros_like(y1), y1, y2])  # (4,)

    # 4 linear constraints on 5 unknowns -> a particular (minimum-norm)
    # solution plus a 1-D null space, via SVD: A = U diag(s) V^T, with
    # `s` length 4 (only 4 singular values for a 4x5 matrix) and `vt`
    # (V^T) the full 5x5 -- its last row is the null-space direction.
    u, s, vt = jp.linalg.svd(A, full_matrices=True)
    eps = 1e-8
    s_safe = jp.where(s > eps, s, 1.0)
    coef = jp.where(s > eps, (u.T @ b) / s_safe, 0.0)  # (4,)
    c_particular = coef @ vt[:4, :]  # (5,)
    null_dir = vt[4, :]  # (5,)

    xlo, xhi = jp.minimum(x0, xt), jp.maximum(x0, xt)
    seg_len = xhi - xlo
    M = _deriv_integral_matrix(xlo, xhi)  # PSD

    A_q = null_dir @ M @ null_dir  # >= 0
    B_q = 2.0 * (c_particular @ M @ null_dir)
    A_q_safe = jp.where(A_q > eps, A_q, 1.0)
    t_vertex = jp.where(A_q > eps, -B_q / (2.0 * A_q_safe), 0.0)

    if a_bar is None:
        t = t_vertex
    else:
        C_q = (c_particular @ M @ c_particular) - a_bar * seg_len
        disc = B_q * B_q - 4.0 * A_q * C_q
        feasible = (disc >= 0.0) & (A_q > eps)
        sqrt_disc = jp.sqrt(jp.maximum(disc, 0.0))
        t1 = (-B_q + sqrt_disc) / (2.0 * A_q_safe)
        t2 = (-B_q - sqrt_disc) / (2.0 * A_q_safe)
        # Of the two curves hitting the curvature bound exactly, keep the
        # one closer to the minimum-norm coefficients -- keeps the profile
        # from swinging unnecessarily far before the final value-clip.
        t_eq = jp.where(jp.abs(t1) < jp.abs(t2), t1, t2)
        t = jp.where(feasible, t_eq, t_vertex)

    return c_particular + t * null_dir


def _overshot(x0: jax.Array, target: jax.Array, x: jax.Array) -> jax.Array:
    """True once `x` has travelled past `target` in the segment's direction
    of travel (`x0 -> target`) -- including the case where a single step's
    motion jumps clean across the `margin` band `maybe_resample` uses to
    detect arrival, without ever landing inside it. Shared by `query_traj`
    (to stop commanding *further* travel once this happens) and
    `maybe_resample` (to resample immediately instead of only on a
    margin hit)."""
    direction = jp.sign(target - x0)
    direction = jp.where(direction == 0.0, 1.0, direction)
    return direction * (x - target) > 0.0


# ---------------------------------------------------------------------------
# Segment sampling / trajectory state.
# ---------------------------------------------------------------------------

def sample_segment(
    rng: jax.Array,
    x0: jax.Array,
    v_limit: float,
    p_min: float,
    p_max: float,
    accel_range: Tuple[float, float],
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Samples one new bow-stroke segment starting at position `x0`.

    Returns `(target, vcoef, pcoef)`:
      target -- new normalized target position in [-1, 1]
      vcoef  -- (5,) ascending-power quartic coefficients for the desired
                lateral velocity v(x) (m/s), v(x0) = v(target) = 0
      pcoef  -- (5,) ascending-power quartic coefficients for the desired
                pressure p(x) (N), p(x0) = p(target) = 0
    """
    k_target, k_x1, k_x2, k_yv1, k_yv2, k_yp1, k_yp2, k_a = jax.random.split(rng, 8)

    target = jax.random.uniform(k_target, (), minval=-1.0, maxval=1.0)

    # The two extra parametrization points are sampled *within* the
    # travelled segment [x0, target] rather than over the full [-1, 1] --
    # otherwise they'd typically land outside the segment the profile
    # actually needs to shape (segments are often much shorter than the
    # full bow range), leaving the interior of the segment essentially
    # unconstrained/extrapolated.
    seg_lo, seg_hi = jp.minimum(x0, target), jp.maximum(x0, target)
    x1 = jax.random.uniform(k_x1, (), minval=seg_lo, maxval=seg_hi)
    x2 = jax.random.uniform(k_x2, (), minval=seg_lo, maxval=seg_hi)

    # Speed (unsigned magnitude) at the two interior points -- sampled >= 0
    # and folded in with the segment's direction of travel below, rather
    # than sampled as a signed velocity directly. A quartic pinned to two
    # independently-signed interior values can (and in practice does) come
    # out net-negative across a segment whose target is in the +x
    # direction, which would drive the virtual bow position *away* from
    # the target it's supposed to reach -- see the module docstring: a
    # stroke moves consistently in one direction, speeding up and slowing
    # down, not reversing mid-segment.
    direction = jp.sign(target - x0)
    direction = jp.where(direction == 0.0, 1.0, direction)
    yv1 = jax.random.uniform(k_yv1, (), minval=0.0, maxval=v_limit)
    yv2 = jax.random.uniform(k_yv2, (), minval=0.0, maxval=v_limit)
    yp1 = jax.random.uniform(k_yp1, (), minval=p_min, maxval=p_max)
    yp2 = jax.random.uniform(k_yp2, (), minval=p_min, maxval=p_max)

    a_bar = jax.random.uniform(k_a, (), minval=accel_range[0], maxval=accel_range[1])

    # `a_bar` is a bound on int v'(x)^2 dx, which is invariant to an
    # overall sign flip, so fitting the unsigned speed profile and then
    # applying `direction` afterwards is equivalent to (and simpler than)
    # threading the sign through the fit itself.
    vcoef = direction * _fit_quartic(x0, target, x1, x2, yv1, yv2, a_bar)
    pcoef = _fit_quartic(x0, target, x1, x2, yp1, yp2, None)
    return target, vcoef, pcoef


def init_traj_info(
    rng: jax.Array,
    x0: jax.Array,
    v_limit: float,
    p_min: float,
    p_max: float,
    accel_range: Tuple[float, float],
) -> Dict[str, jax.Array]:
    """Builds the initial `"traj_*"` fields to seed into `State.info` at
    `reset()`. `x0` should be the *real* measured bow position (e.g.
    `ErhuEnv._bow_stroke_position(data)` right after the episode's domain
    randomization is applied) so the first segment starts where the bow
    actually is, not an arbitrary independent default."""
    key, subkey = jax.random.split(rng)
    x0 = jp.asarray(x0)
    target, vcoef, pcoef = sample_segment(subkey, x0, v_limit, p_min, p_max, accel_range)
    return {
        "traj_key": key,
        "traj_x0": x0,
        "traj_target": target,
        "traj_vcoef": vcoef,
        "traj_pcoef": pcoef,
    }


def query_traj(
    info: Dict[str, jax.Array], x: jax.Array, v_limit: float, p_min: float, p_max: float
) -> Tuple[jax.Array, jax.Array]:
    """Evaluates the current segment's velocity/pressure profiles at the
    given normalized bow position `x` (the real, measured position --
    see the module docstring), clipping the output to the sampling limits
    ("ensure no value exceeds limits [...]").

    Pressure is clipped to [0, p_max] rather than [p_min, p_max]: p_min
    only shapes where the profile's *interior* control points are sampled
    -- the profile is still allowed to (and, by construction, does) pass
    through 0 at the segment's start/end positions.
    """
    x = jp.clip(x, -1.0, 1.0)
    velocity = _eval_poly(info["traj_vcoef"], x)
    # The fit only pins the profile's sign at the two interior points, not
    # everywhere in between (see `sample_segment`) -- a general quartic
    # through those 4 points can still dip to (and past) zero somewhere
    # inside the segment. Floor the direction-aligned component at a small
    # positive creep `_MIN_CREEP_VEL` (rather than 0): without it, a dip
    # would hand the policy a genuine "stop or reverse mid-stroke" reward
    # target, which is a live incentive to loiter there (the segment would
    # then never come within `margin` of `traj_target`, so it would never
    # resample) -- not just a cosmetic wobble in the reference.
    direction = jp.sign(info["traj_target"] - info["traj_x0"])
    direction = jp.where(direction == 0.0, 1.0, direction)
    velocity = direction * jp.maximum(direction * velocity, _MIN_CREEP_VEL)
    pressure = _eval_poly(info["traj_pcoef"], x)
    velocity = jp.clip(velocity, -v_limit, v_limit)
    pressure = jp.clip(pressure, 0.0, p_max)
    return velocity, pressure


def maybe_resample(
    info: Dict[str, jax.Array],
    x: jax.Array,
    v_limit: float,
    p_min: float,
    p_max: float,
    accel_range: Tuple[float, float],
    margin: float = 0.02,
) -> Dict[str, jax.Array]:
    """Once the real bow position `x` comes within `margin` of the current
    segment's target, resamples a fresh segment starting there. Also
    resamples immediately if `x` has overshot the target outright -- e.g. a
    single step's motion can jump clean across the `margin` band without
    ever landing inside it, which would otherwise strand the old (by then
    extrapolated) segment in place indefinitely (see `_overshot`).

    Call once per env step, after `query_traj` has been used for that
    step's reward/obs, so the resampled state is ready for the next step
    (mirrors how `ErhuEnv.step` updates `info["prev_bow_mid"]`).
    """
    reached = (jp.abs(x - info["traj_target"]) < margin) | _overshot(
        info["traj_x0"], info["traj_target"], x
    )

    # Always draw a candidate next segment (cheap: a handful of uniform
    # samples + a 4x5 SVD) and select it only where `reached` -- keeps this
    # branch-free and jit/vmap-safe instead of needing `lax.cond`.
    key, subkey = jax.random.split(info["traj_key"])
    cand_target, cand_vcoef, cand_pcoef = sample_segment(
        subkey, x, v_limit, p_min, p_max, accel_range
    )

    out = dict(info)
    out["traj_x0"] = jp.where(reached, x, info["traj_x0"])
    out["traj_target"] = jp.where(reached, cand_target, info["traj_target"])
    out["traj_vcoef"] = jp.where(reached, cand_vcoef, info["traj_vcoef"])
    out["traj_pcoef"] = jp.where(reached, cand_pcoef, info["traj_pcoef"])
    out["traj_key"] = jp.where(reached, key, info["traj_key"])
    return out


# ---------------------------------------------------------------------------
# `python -m envs.utils_traj` (or `python envs/utils_traj.py`): plot one
# generated velocity/pressure profile, regenerate on "r".
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    _V_LIMIT = 0.1
    _P_MIN, _P_MAX = 0.5, 2.5
    _ACCEL_RANGE = (0.005, 0.05)
    _GRID = jp.linspace(-1.0, 1.0, 400)

    def _profile_for_plot(info: Dict[str, jax.Array]) -> Tuple[np.ndarray, np.ndarray]:
        """Velocity/pressure over the full [-1, 1] grid, with the same
        direction-floor + limit-clip `query_traj` applies at a single
        point (see `query_traj`) -- i.e. what the generator would actually
        hand out at any position, not just the raw unclipped fit."""
        direction = jp.sign(info["traj_target"] - info["traj_x0"])
        direction = jp.where(direction == 0.0, 1.0, direction)
        raw_v = _eval_poly(info["traj_vcoef"], _GRID)
        v = direction * jp.maximum(direction * raw_v, _MIN_CREEP_VEL)
        v = jp.clip(v, -_V_LIMIT, _V_LIMIT)
        p = jp.clip(_eval_poly(info["traj_pcoef"], _GRID), 0.0, _P_MAX)
        return np.asarray(v), np.asarray(p)

    def _generate(rng: jax.Array) -> Dict[str, jax.Array]:
        return init_traj_info(rng, 0.0, _V_LIMIT, _P_MIN, _P_MAX, _ACCEL_RANGE)

    def _draw(fig, ax_v, ax_p, info: Dict[str, jax.Array]) -> None:
        v, p = _profile_for_plot(info)
        x0, xt = float(info["traj_x0"]), float(info["traj_target"])
        seg_lo, seg_hi = min(x0, xt), max(x0, xt)
        grid_np = np.asarray(_GRID)

        for ax in (ax_v, ax_p):
            ax.clear()
            ax.axvspan(seg_lo, seg_hi, color="gray", alpha=0.12, label="travelled segment")
            ax.axvline(x0, color="tab:gray", ls="--", lw=1, label="x0 (start)")
            ax.axvline(xt, color="black", ls="--", lw=1, label="target")
            ax.set_xlim(-1.0, 1.0)

        ax_v.plot(grid_np, v, color="tab:blue", lw=2)
        ax_v.axhline(0.0, color="gray", lw=0.5)
        ax_v.set_ylim(-_V_LIMIT * 1.15, _V_LIMIT * 1.15)
        ax_v.set_ylabel("velocity (m/s)")
        ax_v.set_title(f"x0={x0:.3f}   target={xt:.3f}")
        ax_v.legend(loc="upper right", fontsize=8)

        ax_p.plot(grid_np, p, color="tab:orange", lw=2)
        ax_p.set_ylim(-0.1, _P_MAX * 1.15)
        ax_p.set_ylabel("pressure (N)")
        ax_p.set_xlabel("normalized bow position x  (-1 = frog, +1 = tip)")

        fig.tight_layout()
        fig.canvas.draw_idle()
        plt.pause(0.001)

    rng = jax.random.PRNGKey(np.random.SeedSequence().entropy % (2**31 - 1))

    plt.ion()
    fig, (ax_v, ax_p) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    info = _generate(rng)
    _draw(fig, ax_v, ax_p, info)

    print("Generated velocity/pressure profile -- see the plot window.")
    print("Enter to regenerate, press q to quit.")
    while True:
        try:
            cmd = input("> ").strip().lower()
        except EOFError:
            break
        if cmd == "q":
            exit(0)
        rng, subkey = jax.random.split(rng)
        info = _generate(subkey)
        _draw(fig, ax_v, ax_p, info)

    plt.ioff()
    plt.show()
