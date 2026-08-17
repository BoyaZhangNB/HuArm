# HuArm
A robot arm that plays Erhu

## Observation Space

Built by `ErhuEnv._get_obs` (see [erhu_env.py](envs/erhu_env.py)) as a single
flat vector via `jp.concatenate`, in this order. Dims are for the current
model (`nq=6`, `nv=6`, `nu=5` (`action_size`), force dim `1`,
`n_stack=3`), giving a total observation size of **48** (verified via
`ErhuEnv.observation_size` / `state.obs.shape`).

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
| `action_history` | 15 | Last `n_stack=3` actions, flattened (`3 x action_size=5`) |
| `force_history` | 3 | Last `n_stack=3` force readings, flattened (`3 x force_dim=1`) |
| **Total** | **48** | |
