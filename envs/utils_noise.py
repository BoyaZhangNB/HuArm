"""utils_noise.py

Observation noise for domain randomization: this module randomizes what the
policy *sees*, `utils_dr.py` randomizes the physics it acts on.

Noise types
-----------
    white    independent Gaussian samples, clipped to +/- `clip` sigma.
    uniform  independent uniform samples in +/- `scale`.
    pink     Ornstein-Uhlenbeck drift, s <- a*s + sqrt(1 - a^2)*eps, i.e.
             temporally correlated noise with stationary std `scale`. A high
             `alpha` (0.99 at 25 Hz is a ~4 s correlation time) models the
             slow zero drift of a real encoder or load cell far better than
             independent white noise does.

Two further types from the same taxonomy apply to observations `ErhuEnv`
does not (yet) have, so they are provided as standalone functions rather
than as `NoiseSpec` kinds: `corrupt` for discrete observations such as a
string/contact id, and `block_noise` for image or segmentation
observations.

`ObsNoise` bundles the additive types over a set of named observation
blocks and hands back a single vector to add to the observation, so callers
never need per-block offsets. Its (pink) state and rng live in `State.info`
via `init_noise_info` / `step_noise_info`, mirroring how `utils_traj` keeps
the reference stroke's state there.
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import jax
import jax.numpy as jp


ADDITIVE_KINDS = ("white", "uniform", "pink")


@dataclass(frozen=True)
class NoiseSpec:
    """One additive noise source.

    `scale` is a standard deviation for "white"/"pink" and a half-width for
    "uniform"; `clip` bounds a sample at that many multiples of `scale`
    (<= 0 disables clipping). `alpha` is the AR(1) retention factor of the
    "pink" process, per env step.
    """

    kind: str = "white"
    scale: float = 0.0
    clip: float = 3.0
    alpha: float = 0.99

    def __post_init__(self):
        if self.kind not in ADDITIVE_KINDS:
            raise ValueError(f"unknown noise kind {self.kind!r}; expected one of {ADDITIVE_KINDS}")
        if not 0.0 <= self.alpha < 1.0:
            raise ValueError(f"pink noise alpha must be in [0, 1), got {self.alpha}")
        if self.scale < 0.0:
            raise ValueError(f"noise scale must be non-negative, got {self.scale}")

    @classmethod
    def make(cls, spec: Any) -> "NoiseSpec":
        """Accepts a NoiseSpec or a plain mapping (e.g. straight out of a
        yaml config) and returns a NoiseSpec."""
        return spec if isinstance(spec, cls) else cls(**dict(spec))

    def scaled(self, factor: float) -> "NoiseSpec":
        return NoiseSpec(self.kind, self.scale * factor, self.clip, self.alpha)


def _clip(spec: NoiseSpec, x: jax.Array) -> jax.Array:
    if spec.clip <= 0.0:
        return x
    limit = spec.clip * spec.scale
    return jp.clip(x, -limit, limit)


def white_noise(rng: jax.Array, shape: Tuple[int, ...], spec: NoiseSpec) -> jax.Array:
    return _clip(spec, spec.scale * jax.random.normal(rng, shape))


def uniform_noise(rng: jax.Array, shape: Tuple[int, ...], spec: NoiseSpec) -> jax.Array:
    return jax.random.uniform(rng, shape, minval=-spec.scale, maxval=spec.scale)


def pink_noise_init(rng: jax.Array, shape: Tuple[int, ...], spec: NoiseSpec) -> jax.Array:
    """Draws the OU process from its stationary distribution, so an episode
    starts with a drift offset already in place instead of ramping up from
    zero over its first few seconds."""
    return white_noise(rng, shape, spec)


def pink_noise_next(rng: jax.Array, state: jax.Array, spec: NoiseSpec) -> jax.Array:
    innovation = spec.scale * jax.random.normal(rng, state.shape)
    return _clip(spec, spec.alpha * state + jp.sqrt(1.0 - spec.alpha ** 2) * innovation)


def corrupt(rng: jax.Array, x: jax.Array, choices: jax.Array, prob: float) -> jax.Array:
    """Replaces each element of `x` with a uniform draw from `choices` with
    probability `prob` -- a model for misclassified discrete observations
    (a wrong string id, a spurious contact flag)."""
    mask_rng, draw_rng = jax.random.split(rng)
    mask = jax.random.uniform(mask_rng, x.shape) < prob
    draws = jax.random.choice(draw_rng, jp.asarray(choices), x.shape)
    return jp.where(mask, draws.astype(x.dtype), x)


def block_noise(rng: jax.Array, image: jax.Array, num_blocks: int = 1,
                max_frac: float = 0.25, fill: float = 0.0) -> jax.Array:
    """Stamps `num_blocks` random rectangles of `fill` onto the last two axes
    of `image` -- a model for occlusions in a camera/segmentation
    observation. `max_frac` caps a block's side as a fraction of the image."""
    h, w = image.shape[-2:]
    rows = jp.arange(h)[:, None]
    cols = jp.arange(w)[None, :]

    def one_block(key):
        r_rng, c_rng, bh_rng, bw_rng = jax.random.split(key, 4)
        bh = jax.random.randint(bh_rng, (), 1, max(1, int(h * max_frac)) + 1)
        bw = jax.random.randint(bw_rng, (), 1, max(1, int(w * max_frac)) + 1)
        r0 = jax.random.randint(r_rng, (), 0, h)
        c0 = jax.random.randint(c_rng, (), 0, w)
        return (rows >= r0) & (rows < r0 + bh) & (cols >= c0) & (cols < c0 + bw)

    mask = jp.any(jax.vmap(one_block)(jax.random.split(rng, num_blocks)), axis=0)
    return jp.where(mask, fill, image)


class ObsNoise:
    """Additive noise over a contiguous prefix of an observation vector.

    `sizes` maps block name -> width, in the order the blocks appear in the
    observation; `specs` maps a subset of those names to the noise sources
    applied to that block. `sample` returns a single vector of length
    `size` == sum(sizes) (zeros where a block has no specs) to add to the
    observation's leading `size` entries.

    `scale` multiplies every spec's magnitude, so a single knob (0.0) turns
    all observation noise off for an ablation or a debugging run.
    """

    def __init__(self, sizes: Mapping[str, int],
                 specs: Mapping[str, Sequence[Any]] = None, scale: float = 1.0):
        specs = dict(specs or {})
        unknown = set(specs) - set(sizes)
        if unknown:
            raise ValueError(f"obs noise for unknown observation block(s): {sorted(unknown)}")

        blocks = []
        for name, width in sizes.items():
            block_specs = tuple(
                NoiseSpec.make(s).scaled(scale) for s in specs.get(name, ())
            )
            blocks.append((name, int(width), tuple(s for s in block_specs if s.scale > 0.0)))
        self._blocks: Tuple[Tuple[str, int, Tuple[NoiseSpec, ...]], ...] = tuple(blocks)
        any_noise = any(block_specs for _, _, block_specs in self._blocks)
        self.size = int(sum(sizes.values())) if any_noise else 0

    def init_state(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Initial state of every "pink" source, keyed `<block>_<index>`."""
        state = {}
        for key, width, spec in self._pink_sources():
            rng, sub = jax.random.split(rng)
            state[key] = pink_noise_init(sub, (width,), spec)
        return state

    def sample(self, rng: jax.Array, state: Mapping[str, jax.Array]
               ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """Returns (noise vector, advanced pink state) for one env step."""
        if self.size == 0:
            return jp.zeros((0,)), dict(state)

        state = dict(state)
        chunks = []
        for name, width, specs in self._blocks:
            noise = jp.zeros((width,))
            for i, spec in enumerate(specs):
                rng, sub = jax.random.split(rng)
                if spec.kind == "white":
                    noise = noise + white_noise(sub, (width,), spec)
                elif spec.kind == "uniform":
                    noise = noise + uniform_noise(sub, (width,), spec)
                else:  # pink -- carried across steps
                    key = f"{name}_{i}"
                    state[key] = pink_noise_next(sub, state[key], spec)
                    noise = noise + state[key]
            chunks.append(noise)
        return jp.concatenate(chunks), state

    def block(self, vector: jax.Array, name: str) -> jax.Array:
        """The slice of a `sample` vector belonging to one block, for a
        caller that needs a single block's noise on its own -- e.g. to store
        the noisy *measurement* of a signal, not just to perturb the
        observation. Zeros when noise is off, so callers need no branch."""
        offset = 0
        for block_name, width, _ in self._blocks:
            if block_name == name:
                return jp.zeros((width,)) if self.size == 0 else vector[offset:offset + width]
            offset += width
        raise KeyError(f"no observation block named {name!r}")

    def _pink_sources(self):
        for name, width, specs in self._blocks:
            for i, spec in enumerate(specs):
                if spec.kind == "pink":
                    yield f"{name}_{i}", width, spec


def init_noise_info(noise: ObsNoise, rng: jax.Array) -> Dict[str, Any]:
    """Builds the `"noise_*"` / `"obs_noise"` fields to seed into
    `State.info` at `reset()`. `"obs_noise"` is the vector `_get_obs` adds
    to the observation, so the observation stays a pure function of
    (data, info)."""
    key, state_rng, sample_rng = jax.random.split(rng, 3)
    vector, state = noise.sample(sample_rng, noise.init_state(state_rng))
    return {"noise_key": key, "noise_state": state, "obs_noise": vector}


def step_noise_info(info: Mapping[str, Any], noise: ObsNoise) -> Dict[str, Any]:
    """Draws the next step's noise vector, advancing any pink state. Call
    once per env step, before `_get_obs`."""
    key, sample_rng = jax.random.split(info["noise_key"])
    vector, state = noise.sample(sample_rng, info["noise_state"])
    return {"noise_key": key, "noise_state": state, "obs_noise": vector}
