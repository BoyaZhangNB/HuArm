# HuArm
A robot arm that plays Erhu

## Observation Space

Built by `ErhuEnv._get_obs` (see [erhu_env.py](envs/erhu_env.py)) as a single
flat vector via `jp.concatenate`, in this order. Dims are for the current
model (`nq=6`, `nv=6`, `nu=5`, force dim `1`, `n_stack=3`). `action_size` is
`nu + 1 = 6` -- the 5 arm ctrl deltas plus a 6th dim that sets
`bow_frog_hinge`'s friction-clamp stiffness rather than driving an actuator
(see `ErhuEnv`'s class docstring) -- giving a total observation size of
**51** (verified via `ErhuEnv.observation_size` / `state.obs.shape`).

The model has 6 joints (`joint1, joint2, joint5, joint3, joint4,
bow_frog_hinge`), but `bow_frog_hinge` is a passive, unsprung joint with no
real-world sensor -- nothing measures or tracks it at inference time -- so
its qpos/qvel are excluded from the observation (`ErhuEnv._obs_qpos_idxs` /
`_obs_qvel_idxs`). It's still simulated and part of the physics state, just
not observed, so `qpos`/`qvel` below are `nq - 1` / `nv - 1`, not `nq`/`nv`.

| Component | Dim | Description |
|---|---|---|
| `qpos` | 5 | Joint positions, excluding the unmeasurable `bow_frog_hinge` (`nq - 1`) |
| `qvel` | 5 | Joint velocities, excluding the unmeasurable `bow_frog_hinge` (`nv - 1`) |
| `sound_box_rel` | 3 | Sound box position, relative to base |
| `rel_quat` | 4 | Bow orientation quaternion, relative to erhu root |
| `frog_rel` | 3 | Bow frog position, relative to erhu root |
| `tip_rel` | 3 | Bow tip position, relative to erhu root |
| `mid_rel` | 3 | Bow midpoint position, relative to erhu root |
| `force` | 1 | Uni-directional bow/arm contact force: the raw 3-axis `bow_arm_contact` sensor reading, rotated to world and projected onto the arm's last-link axis (`ErhuEnv._axial_force`) |
| `desired_velocity` | 1 | Target bow velocity |
| `desired_pressure` | 1 | Target bow pressure |
| `forbidden_dist` | 1 | Distance to forbidden bowing area |
| `action_history` | 18 | Last `n_stack=3` actions, flattened (`3 x action_size=6`) |
| `force_history` | 3 | Last `n_stack=3` force readings as observed (noise included), flattened (`3 x force_dim=1`) -- the newest entry is this step's, so it repeats the `force` slot above |
| **Total** | **51** | |

Observation noise is added to the measurement blocks -- `qpos`, `qvel`, the
five pose blocks and `force`, i.e. the first 27 entries. The reference
values (`desired_*`, `forbidden_dist`) stay exact, being commands rather
than measurements. The history blocks get no *second* noise draw, but
`force_history` is not clean either: it stores each step's force as it was
observed, noise included, so the history a policy reads in training is the
same kind of signal a real sensor's history gives it. `action_history` is
exact, since actions are known. See the Domain Randomization section below.

## Domain Randomization

Both halves are drawn fresh every `reset()` and can be tuned per training
run through `ErhuEnv`'s `dr_config` / `obs_noise` / `obs_noise_scale`
arguments (i.e. from a config's `env:` block).

**Physics** -- [envs/utils_dr.py](envs/utils_dr.py), carried through the
episode in `state.info["dr_params"]` and merged onto the base model by
`ErhuEnv.effective_model`:

| What | How |
|---|---|
| Bow weight | One factor for the whole bow assembly (mass and inertia together), on top of the per-body mass jitter applied to the rest of the model |
| Actuator params | Position-actuator gains `kp`/`kv` and the first-order filter time constant that stands in for actuation delay, per actuator |
| Erhu placement | Drawn from the pre-solved pose pool (`utils_envs.build_erhu_pose_pool`), then drifted slowly across the episode: a cylinder is drawn around the instrument, one point sampled on each end cap, and the erhu walks toward the pose that best fits its top/bottom centres to them |
| Bow placement | Start pose comes with the drawn pool entry; the reference stroke it is scored against is resampled continuously by [utils_traj.py](envs/utils_traj.py) |
| Contact params | `solref` (time constant, damping ratio) and `solimp` (`d0`, `d1`, width) on the bow-hair/string pairs only -- the contact that is the task; every other contact in the scene is a backstop the bow should never reach |
| String friction | Sliding friction of the bow-hair/string pairs (and the string geoms), on an extra factor of its own on top of the per-element friction jitter |
| Joint damping | Per-dof factor |

The drifted pose *is* the erhu's real pose: it goes into the model, so MJX
recomputes the instrument's kinematics from it and every consumer --
observations, contacts, the reference stroke's bow position, the reward
terms -- reads it back out of `data`, all seeing the same pose within a
step. Bow velocity is differenced in the erhu's own frame for the same
reason (`prev_bow_mid_local`): a stroke is motion of the hair relative to
the strings, so an instrument sliding under a stationary bow counts as
bowing, and a world-frame difference would miss it.

**Observations** -- [envs/utils_noise.py](envs/utils_noise.py), drawn into
`state.info["obs_noise"]` each step and added to the observation by
`_get_obs`. `qpos`, `qvel`, pose and `force` carry independent standard
deviations, each as a per-step read error (white) plus a slowly drifting
bias (pink, an Ornstein-Uhlenbeck process). `corrupt` and `block_noise` are
implemented there too, for discrete and image observations this env does
not currently have.
