# HuArm
A robot arm that plays Erhu

## Observation Space

Built by `ErhuEnv._get_obs` (see [erhu_env.py](envs/erhu_env.py)) as a single
flat vector via `jp.concatenate`, in this order. Dims are for the current
model (`nq=5`, `nv=5`, `nu=4` (`action_size`), force sensor dim `3`,
`n_stack=3`), giving a total observation size of **53**.

| Component | Dim | Description |
|---|---|---|
| `qpos` | 5 | Joint positions (`nq`) |
| `qvel` | 5 | Joint velocities (`nv`) |
| `sound_box_rel` | 3 | Sound box position, relative to base |
| `rel_quat` | 4 | Bow orientation quaternion, relative to erhu root |
| `frog_rel` | 3 | Bow frog position, relative to erhu root |
| `tip_rel` | 3 | Bow tip position, relative to erhu root |
| `mid_rel` | 3 | Bow midpoint position, relative to erhu root |
| `force` | 3 | Bow/string contact force sensor reading |
| `desired_velocity` | 1 | Target bow velocity |
| `desired_pressure` | 1 | Target bow pressure |
| `forbidden_dist` | 1 | Distance to forbidden bowing area |
| `action_history` | 12 | Last `n_stack=3` actions, flattened (`3 x action_size=4`) |
| `force_history` | 9 | Last `n_stack=3` force readings, flattened (`3 x force_dim=3`) |
| **Total** | **53** | |
