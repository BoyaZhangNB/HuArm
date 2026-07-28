# Modular MJX RL Environment

A small, dependency-light framework for reinforcement learning on top of
**MJX** (MuJoCo XLA), built so that the **physics model** and the
**training algorithm** are both swappable without touching the other.

## Layout

| File | Responsibility |
|---|---|
| `mjx_env.py` | `MjxEnv` base class + `State` pytree. Loads any MJCF model, handles physics substeps (`pipeline_init`/`pipeline_step`). Subclass this per task. |
| `wrappers.py` | `EpisodeWrapper` (max length + truncation), `AutoResetWrapper` (auto-reset inside a scan), `VmapWrapper` (batch N envs in parallel). Reusable for any task. |
| `training_interface.py` | `Agent` protocol + `Transition` pytree + generic `rollout()`/`train()` loop. Reusable for any algorithm. |
| `example_env.py` | Concrete task: cartpole swing-up, showing how to subclass `MjxEnv`. |
| `example_agent.py` | Concrete algorithm: a minimal REINFORCE agent, showing how to satisfy the `Agent` protocol. |
| `train_demo.py` | Wires everything together end to end. |

## Core API every environment exposes

```python
state = env.reset(rng)          # -> State(pipeline_state, obs, reward, done, metrics, info)
state = env.step(state, action) # -> State
env.observation_size            # int
env.action_size                 # int
env.dt                          # float, seconds per env.step()
```

`State` is a `flax.struct.dataclass`, i.e. a JAX pytree, so `reset`/`step`
are `jit`/`vmap`/`scan`-safe out of the box.

## Plugging in a custom model

Subclass `MjxEnv`, pass your MJCF via `xml_path=` or `xml_string=`, and
implement `reset`, `step`, `_get_obs`:

```python
class MyRobotTask(MjxEnv):
    def __init__(self, **kwargs):
        super().__init__(xml_path="my_robot.xml", n_frames=4, **kwargs)

    def reset(self, rng):
        ...
        return State(pipeline_state, obs, reward, done, metrics, info)

    def step(self, state, action):
        data = self.pipeline_step(state.pipeline_state, action)
        ...
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

    def _get_obs(self, data, info):
        return jp.concatenate([data.qpos, data.qvel])
```

Everything else (model loading, substepping, `dt`, sizes) is handled by
the base class.

## Plugging in a custom training algorithm

Implement the `Agent` protocol from `training_interface.py`:

```python
class MyAlgo:
    def init(self, rng) -> train_state: ...
    def act(self, train_state, obs, rng) -> (action, extra): ...
    def update(self, train_state, batch: Transition) -> (train_state, metrics): ...
```

Then run it with the stock loop:

```python
from training_interface import train
train(env, MyAlgo(...), rng, num_iterations=100, steps_per_iteration=200)
```

Because `rollout()`/`train()` only depend on the `Agent` protocol and the
`env.reset`/`env.step` contract, you can swap PPO, SAC, evolutionary
strategies, or a hand-coded controller in for `ReinforceAgent` with zero
changes to the environment code, and swap environments with zero changes
to the algorithm code.

## Scaling to many parallel envs

```python
env = MyRobotTask()
env = EpisodeWrapper(env, episode_length=1000)
env = AutoResetWrapper(env)
env = VmapWrapper(env, batch_size=4096)   # simulate 4096 envs in lockstep
```

`observation_size`/`action_size` stay per-env; `state.obs` etc. gain a
leading batch dimension automatically via `vmap`.

## Requirements

```
pip install mujoco mujoco-mjx jax jaxlib flax optax
```

(GPU/TPU `jaxlib` recommended for realistic env-batch sizes; CPU works
for small examples like `train_demo.py`.)

## Try it

```bash
python example_env.py    # smoke test: random rollout on one env
python train_demo.py     # end-to-end: 64 parallel envs, REINFORCE agent
```
