# Project Journal

## Week 0 - Repository Setup & First Erhu Model (July 12-18 2026)
**Goals**: Set up repo, build a first MuJoCo model of the erhu and bow, get the simulation to run stably

**Completed**:
- Set up GitHub repo (`HuArm`)
- Initial erhu model in MJCF (`huarm/erhu.xml`). Claude generated the first attempt, which I then revised
- Bow hair modelled as a deformable `flexcomp` with tension; strings as thin bodies

**Blockers**:
- During teleoperation in sim, the scene kept snapping back to its initial pose. The high-stiffness erhu strings were causing a numerical explosion, and by default MuJoCo resets the simulation when it detects that kind of divergence

**Learned**:

MuJoCo
- Increasing damping on the high-stiffness erhu strings stopped the numerical explosion and let the simulation render
- MuJoCo's `connect` equality constraint keeps two bodies at the distance they *started* at. So pre-stressing the bow hair can't be done by just connecting the hair ends to the bow ends. The bow has to be bent and allowed to snap open, or the anchor offsets have to be zeroed
- Disabling a connection, moving the two joints, and re-enabling it does *not* change the compiled distance, so the bodies get pulled back to their original separation regardless

**Next Week**:
- Pre-tension the bow hair properly
- Attach the bow to the arm and get the hair inserted between the strings

## Week 1 - Bow Tension, Arm-Bow Connection, Choosing JAX/MJX (July 19-25 2026)
**Goals**: Tension in bow hair and strings, connect bow to arm, place the bow between the strings, pick an RL simulation framework

**Completed**:
- Teleoperation of the bow through a mocap body
- Proper pre-stress in the bow hair (tension in both bow hair and strings, with printed tension readout)
- Removed the hair interpolation step after testing showed it wasn't helping
- Connected the bow to the arm
- IK routine that inserts the bow hair between the two erhu strings and keeps it there
- Force sensor on the bow-arm connection
- Arm set in a resting pose at initialization

**Blockers**: None significant

**Learned**:

Modelling
- The bow-arm joint shouldn't be a free joint. The bow needs a defined mount on the arm

Framework choice: JAX/MJX vs. MLX
- There was a long debate between MJX on the official JAX backend and a community MLX port (MLX runs natively on Apple GPUs, which JAX does not). I chose official JAX because I needed extensive documentation for rapid prototyping. It also gives the same CPU baseline as PyTorch, while keeping the option of CUDA-vectorized training on a rented GPU
- Tried the MJX tutorial notebook: ~2 min of JIT compilation and ~11 min to train a full humanoid policy on a T4 GPU. A comparable run on my MacBook CPU with PyTorch + Gym would take ~6 hours. That gap is the case for vectorized simulation

**Next Week**: Get the erhu environment running inside MJX

## Week 2 - Making the Model MJX-Compatible, First Training Loop (July 26-August 1 2026)
**Goals**: Run the erhu environment in MJX, verify the pipeline with a dummy training run, design the RL environment

**Completed**:
- Replaced the `flex` bow hair with a rigid-body version (MJX doesn't support flex objects)
- "Soft rigid body" version: rigid geometry with compliant contact parameters
- Merged the bow geoms, combined the hair into the same kinematic tree as the bow and arm, disabled unnecessary contacts, and removed the bow's flex joints, all to cut JIT time
- Erhu env runnable in MJX with JIT-compiled step functions, plus a performance timer
- Model save/load (including on keyboard interrupt), inference script, evaluation with eval reward
- Action space changed to delta control
- Wrote out the full environment design: action space, observation space, task reward, regularization reward, termination conditions

**Blockers**:
- The first MJX-compatible model wouldn't finish compiling after 30+ minutes. One logged run showed a 2m52s `jit_scan` compile, then `loss=nan`, then 16 minutes for a single iteration
- The REINFORCE baseline didn't account for early termination, and evaluation had to handle the batch shape of vectorized, auto-resetting environments

**Learned**:

MJX / JAX
- Many MuJoCo features aren't implemented in MJX, including flex objects and energy computation. That is a real limitation compared to CUDA-based Isaac Sim
- **Fancy model == slow physics.** The bow had 8 joints to mimic its flexibility, which doesn't matter for policy learning. After simplifying the joints, geoms and contacts, the model compiled in ~30 seconds
- Moved the IK solve into model initialization and cached the resulting model/data, so `reset()` reuses them instead of re-solving the Jacobian each time. Physics constants can be randomized with `.replace()` / `.at[].set()`, but changing geometry forces a recompile
- Every Python dict passed through JIT must keep the same keys (pytree structure). The `info` dict has to be initialized before compilation so its structure is known
- I wasn't caching the JIT-compiled function, so every training iteration recompiled. After fixing that, the first two iterations were still slower than the rest. Per Claude, *weak types* at initialization triggered a recompile once the types became strong on the second iteration
- Throughput after the fixes: 2000 steps x 64 envs = 128,000 agent steps in ~23 s. By comparison, a 2.5M-step PPO run in my earlier project took at least half an hour; MJX would need ~8 minutes for the same count
- Solver timestep is a direct accuracy vs. compute trade-off

RL
- Termination means the next state's value is zero. Truncation means the episode was cut short but the final state still has value. The two have to be handled differently in the value target

Claude
- Claude wrote the environment to my spec, but its biggest weakness is geometric understanding. Code involving directions and axes needs careful review. Algorithmic code not specific to my `erhu.xml` model was generally correct

**Next Week**: Reward and observation design, then the first real training runs

<!-- TODO image: simplified bow/arm kinematic tree -- "Screenshot 2026-07-29 at 12.46.05 PM.png" in Obsidian vault; upload and replace -->

## Week 3 - Rewards, Termination, First Real Training, SAC & Domain Randomization (August 2-8 2026)
**Goals**: Implement and validate rewards and termination, start training, try different RL algorithms

**Completed**:
- Environment rewards and termination conditions (first draft by Claude, then reviewed and corrected)
- Tested the rewards with a renderer and teleop agent, with real-time logging of kinematics and reward terms
- Cauchy kernel tracking rewards
- Clipped force penalty, tuned reward scales, disabled the bow-crosses-string termination
- Logger in eval, gradient clipping, `tanh` squashing with the proper log-prob correction
- SAC agent alongside PPO; config files for training runs
- Gravity enabled, time-varying desired velocity (sine profile), erhu pose variation
- First domain randomization pass, with a vectorized initial IK solver

**Blockers**:
- Training collapsed: the model converged by ~125 iterations but at low reward. With no `tanh` clipping, parameters saturated and reward terms went flat after a few epochs, which suggested exploding gradients
- Domain randomization made `env.step()` very slow

**Learned**:

Environment correctness
- Caught a mistake in Claude's version: the max-force termination only checked the bow-arm sensor, but it should trigger on *any* contact force
- Tuned `solref` / `solimp` on the bow hair to soften it for the smaller timestep. Some interpenetration (穿模) remained
- Real servos have an internal PD controller and actuator lag. In MuJoCo the PD part can be modelled with `gainprm` / `biasprm` and the lag with `dynprm`

Reward design
- The Cauchy kernel is a common choice for tracking-error rewards. It keeps the reward positive and its gradient stays large in the tails. Rule of thumb: α = 1 / e²_acceptable
- Checking reward-term magnitudes in the live plot showed the contact (max-force) penalty was far too small to discourage violent behaviour, the string-centre penalty also needed rescaling, and the contact penalty had to be clipped at -10

RL algorithms
- Clipping or transforming a network's output distorts the action distribution, so the log-probability has to be corrected (tanh-squashed Gaussian)

Domain randomization performance
- `step()` was slow because the whole model pytree was being passed around. The fix was to store only the randomized model variables in `state.info` and rebuild the model each step. After that, domain randomization ran at a decent speed

**Next Week**: Fix the observation scale, termination behaviour and IK initialization

## Week 4 - Normalization, IK Reachability, a 5th Joint, iPhone Teleoperation (August 9-15 2026)
**Goals**: Get episodes to last more than a few steps, stabilize bow-string contact, build a teleoperation pipeline for demonstrations

**Completed**:
- Cauchy kernel on the `bow_touch` and `string_center` rewards; all reward terms brought into roughly [-1, 1]
- Pre-solved erhu pose pool sampled at reset
- Observation normalizer (running mean/std)
- Termination reward of -100
- Erhu position noise limited to the x-axis; larger string gap
- `teleop.py`: teleoperating the arm in sim
- Joint tracking, contact margin and gap for contact detection
- **Added a joint (now 5 DoF)** and made the IK solver stable
- iOS teleoperation app (`Teleop/`, SwiftUI + ARKit): iPhone pose streamed over UDP, control data over TCP, and a slider for the pitch of `joint3`
- EMA filtering on force and contact readings; softened hair-string contact
- Logged *actual* bow velocity and pressure during teleop, used as the "desired" values for demonstrations

**Blockers**:
- **Episodes terminated after 1-2 steps.** Bow-string contact is so sensitive that pure RL couldn't collect meaningful data
- The initialization IK solver couldn't solve for orientation: some poses were simply unreachable with 4 DoF
- Contact kept dropping. When the arm moved the bow, the hair bounced away from the string, then registered contact again when motion stopped (sometimes even with a visible gap). This made the contact-duration reward unstable
- Briefly tried **MuJoCo Warp** for GPU training with a deformable bow hair, but it was very unstable with edge contact on such thin strings, so I abandoned it

**Learned**:

RL
- Raw observations ranged from about 1e-7 to 1e3, so normalizing them was necessary
- **If an agent only ever gets negative reward, it learns to die as early as possible.** Adding a large termination penalty is one fix; a survival reward is another
- Worth initializing from demonstrations (imitation learning or residual RL) rather than learning from scratch

IK & robot geometry
- Things I tried for the IK failures: adding the unactuated hinge to the solver, adding orientation tracking (worked only for some angles), adding an extra rotational joint
- Limiting erhu position randomization to the x-axis and **significantly widening the string gap** sharply increased IK success rate. The gap may cause sim-to-real issues later, but I kept it for stability
- 4 DoF couldn't reach all the required bowing poses, so I added a 5th joint

Contact
- Fixes for the bouncing contact: add a margin/gap to contact detection, increase damping and the contact time constant, soften `solimp`, and define "touching" by (EMA-filtered) force rather than by raw contact detection

Networking (teleop)
- Pose data goes over **UDP** (cheap, real-time, a dropped packet doesn't matter). Control data goes over **TCP** (reliable delivery). Positions and angles are sent as deltas from the starting pose
- Controlling the bow pitch with a manual slider instead of ARKit orientation gave much more stable convergence

**Next Week**: Analyze the demonstration data, then behaviour cloning

<!-- TODO image: iPhone teleop app screenshot / video -->

## Week 5 - Behaviour Cloning, RLPD, Trajectory Generator (August 16-22 2026)
**Goals**: Clean and analyze demonstrations, warm-start the policy, design a more diverse bowing trajectory

**Completed**:
- Demonstration playback and EDA (`demonstrations/plot_demo.py`, `test_demonstration.py`)
- Cleaned short demos; EMA on velocity
- **Observation space fix:** removed `bow_frog_hinge` qpos/qvel because nothing measures that hinge in real life
- **Behaviour cloning** warm start (`bc/train_bc.py`) for SAC, and later for PPO
- **RLPD**: SAC with demonstration transitions in the replay buffer (`demonstrations/demo_buffer.py`)
- Ran a 4-way comparison: SAC / SAC+BC / SAC+RLPD / SAC+BC+RLPD
- Prevented the bow from leaving the strings
- Rigid bow-hinge variant (`arm_rigid.xml`) and comparison runs
- **Position-based trajectory generator** (`envs/utils_traj.py`), see below
- Policy inference driven live from the iPhone: velocity and pressure commands sent over UDP to `inference.py`
- Fixed-RNG evaluation for comparable eval scores; saved config with each run

**Blockers**:
- Pressure tracking failed in every configuration. That points to an environment or modelling problem rather than the training algorithm
- The new random trajectory may be too messy to learn: a policy trained on a plain sine trajectory tracked *just as well*

**Learned**:

Demonstration data
- Reward is roughly uniform, with rhythmic downward spikes that probably line up with bow direction changes
- Joints 1, 2 and 4 are the noisiest, probably because their short links are easily perturbed during the IK solve
- The erhu was drifting a few centimetres relative to the arm base during collection
- The demonstration velocity and pressure distributions gave a good basis for setting tracking tolerances (around one standard deviation)

Behaviour cloning
- BC eval loss fell together with training loss, so the policy was learning to bow rather than memorizing. **About 20 minutes (~30,000 steps) of teleop data was enough for BC to learn basic erhu bowing**

BC vs. RLPD (110k-step runs, ~33 min on M1 Pro CPU)
- BC initialization can *hurt* RLPD. BC+RLPD oscillated heavily early on, probably because cloned behaviour conflicted with RL-guided behaviour. Pure RLPD reached near-max reward faster, and also mastered velocity and bow-pose control faster
- BC bakes in a stronger bias: better for survival (BC-initialized policies survived more consistently), but slower to reach reward-maximizing behaviour. Expert demonstrations include suboptimal actions, and BC copies them blindly
- A SAC baseline with no demonstrations in the better-normalized env also did well, just slower
- **Why did 110k steps with `num_envs=1` match 1.6M steps with `num_envs=64`?** SAC performs a fixed number of gradient updates per iteration, regardless of how many transitions are in the buffer

Trajectory design
- Requirements: the trajectory depends on **bow position, not time** (this gives a stronger learning signal when the policy performs poorly), randomizes velocity, acceleration and pressure, and never pushes the bow beyond ±k (or it collides with the strings)
- Implementation: sample a target x ∈ [-1, 1] (normalized position along the bow), two shape points (x₁, y₁), (x₂, y₂), and an average curvature bound ∫|v′(x)|²dx / (x − x₀) = ā. Five constraints fix a **quartic**. Fall back to the minimum-curvature quartic when ā is infeasible. Resample when the bow gets near the target
- Changing velocity/pressure tracking to *percent* error didn't help, and it breaks RLPD because the reward scale of the stored demonstrations no longer matches
- **Lesson:** I changed too many things at once. The goal was for the previous working run to continue its learning trend; nothing should have been brand new. Isolate one change at a time

**Next Week**: Finish domain randomization, train with the simple sine trajectory, and see whether that fixes pressure tracking

## Week 6 - Full Domain Randomization, Audio Synthesis, Adjustable Frog Friction (August 23-29 2026)
**Goals**: Complete domain randomization and observation noise, make the sim audible, give the policy a way to apply pressure

**Completed**:
- Full domain randomization (`envs/utils_dr.py`): bow mass, actuator gains/delay, erhu placement drift, `solref`/`solimp` on hair-string pairs, string friction, joint damping
- Observation noise (`envs/utils_noise.py`): white per-step noise plus an Ornstein-Uhlenbeck (pink) drifting bias, with separate scales for qpos, qvel, pose and force
- **Training succeeded with domain randomization**
- `synthesis.py`: real-time erhu sound from the sim's bow state, using Faust's bowed-string physical model (via DawDreamer) with an erhu-specific body/resonator. Hooked into inference and teleop
- **New action dimension: adjustable friction on the bow-frog hinge**, a stand-in for a clamp that a servo tightens. Applied as a smoothed Coulomb torque `-scale·tanh(qvel/v_eps)`
- Removed `arm_rigid.xml` after the compliant-vs-rigid comparison
- Teleop app redesign: stiffness control and a new theme

**Blockers**:
- The safety termination made the arm avoid the sound box too much, and terminations still happened late in training
- The custom friction torque overshot and produced NaN parameters

**Learned**:

Physics
- An explicit force in `qfrc_applied` isn't covered by the implicit integrator. At the frog dof's tiny inertia, an unclamped friction torque overshoots qvel's zero crossing and diverges. **Clamping the torque each substep to the impulse that would exactly zero qvel stabilized training, and reward increased substantially**
- MJX fixes friction constraint rows at compile time, so a per-step adjustable friction has to be written as an explicit force law

Domain randomization & noise
- Ornstein-Uhlenbeck noise, s_{t+1} = μ + α(s_t − μ) + √(1−α²)·ε, with high α gives slow, smooth drift. That is a better model of real sensor bias and actuator variation than independent white noise (approach from Boshi An's noise types)
- Noise goes only on measurements, not on commanded or reference values (desired velocity/pressure)

Results
- Training *without* demonstrations gave better pressure but worse velocity tracking. Velocity tracking needs many joints to coordinate, which makes it harder to discover through exploration
- The compliant-hinge model can track both pressure and velocity, but held the bow well above the sound box
- A lower alpha learning rate (`alpha_lr = 2e-4`) gave a lower maximum reward

**Next Week**: Adjust erhu placement and termination parameters

## Week 7 - Erhu Placement & Termination Experiments (August 30-September 5 2026)
**Goals**: Find out why the policy is so conservative near the sound box

**Completed**:
- Periodically saved the best checkpoint
- Run with the erhu positioned higher (it had been too low for the arm design)
- Bow-axis alignment heuristic in the IK/initialization
- Termination parameter sweep: termination penalty `-1` instead of `-100`, and `f_max` 30 N instead of 10 N

**Blockers**:
- The higher-erhu run got more total reward, but the bow was initialized in the wrong place, so the result needs a properly threaded rerun
- Balancing pressure tracking against friction control was hard, even with RLPD
- A 2-stage curriculum was too slow to compile on GPU and produced NaN on CPU

**Learned**:

Termination as an escape hatch
- Counterintuitive result: a *more tolerant* environment (smaller termination penalty, higher force limit) produced a *more conservative* policy that just lifted the bow off the erhu
- Explanation: (1) the policy can no longer terminate early to escape large negative rewards; (2) the reachable state space grows, so the correct behaviour is harder to discover
- RLPD mainly causes a noisier start. Once performance converges, peak performance is still driven by RL
- Parallel environments gave higher, more stable peak performance in less time

**Next Week**: Simplify the trajectory to isolate the problem

<!-- TODO image: misplaced bow with higher erhu (Huarm-wrong-init.png in Obsidian vault); upload and replace -->

## Weeks 8, 9 - Simple Trajectory Baseline (September 6-19 2026)
**Goals**: Get stable bowing on the easiest possible command profile, then add difficulty

**Completed**:
- `envs/utils_traj_simple.py`: sine-wave velocity over time plus a constant pressure, with period and pressure sampled per episode
- Tuned the sine period range and desired velocity
- Adjusted bow and arm geometry

**Blockers**:
- The first simple-trajectory run held the bow stable and applied adequate pressure, **but stayed at the bow tip and didn't move when commanded**. Likely causes: fear of termination, plus a velocity profile that changed too fast
- The school term started, so progress slowed

**Learned**:
- Modern robot learning roughly follows imitation pre-training followed by RL post-training
- Reflection: pure RL episodes ended after ~3 steps because bow-string contact is so sensitive. Instead of continuing to tune the environment, I let go of doing pure RL, built teleoperation, and used demonstrations to warm-start learning. **Seemingly suboptimal solutions can beat picture-perfect methods that fail in practice**
- Status at this point: episodes run to the full 512-step limit and the policy reaches **~78% velocity-tracking accuracy** in evaluation. Pressure tracking is still the weak point

## Week 10 - Actuator Physics & Restoring the Markov Property (September 20-26 2026)
**Goals**: Understand why pressure tracking fails, and check that the MDP assumptions hold

**Completed**:
- **Added frog clamp stiffness to the observation** to restore the MDP property
- Switched the arm to **joint-space velocity (velocity-field) control** actuators
- Logged and compared velocity control against position control with stiffness in the observation
- `requirements.txt`

**Blockers**:
- Velocity control gave lower reward, and both velocity and pressure tracking were worse than with position control
- Occasional NaN critic loss in training

**Learned**:

Control (thanks to a conversation with Boshi An)
- **High positional gain makes contact very stiff.** A stiff position servo is dexterous and tracks velocity well, but it is bad at regulating pressure. A velocity-type law, τ = K_d(v̄ − v), behaves differently in contact

MDP
- **The joint stiffness wasn't in the observation at all.** The policy controlled a state it couldn't see, which breaks the Markov property. Actuators with internal state (like a held stiffness or setpoint) must have that state observed
- I committed the observation fix separately from the controller change, so their effects could be evaluated independently

RLPD (from reading Ball et al. 2023 and the RLinf docs)
- Symmetric sampling: half of each minibatch from offline data, half from online
- Critic divergence with offline data can explain NaN losses; the paper's remedy is layer-normalizing the Q-network
- Demonstration bootstrapping fundamentally changes the RL problem

**Next Week**: Torque-driven frog joint, proper RLPD sampling

## Week 11 - Torque-Driven Frog, RLPD Symmetric Sampling (September 27-October 3 2026)
**Goals**: Replace the friction-based frog with a direct torque actuator, train RLPD on the complex trajectory

**Completed**:
- **Changed the bow-frog hinge to direct torque drive** (a `<motor>`, delta-commanded). Updated the teleop app's packet and UI to match
- Raised the torque limit to 1 N·m
- RLPD with **symmetric 50/50 sampling** from a separate offline demo buffer that rollouts never overwrite (`agents/sac_agent.py`)
- **RLPD on the complex quartic trajectory works**

**Blockers**:
- The velocity-controlled end effector twitches when commanded to stay at rest
- The arm sometimes folds onto itself: small mistakes compound, and the policy doesn't know how to recover (the classic covariate-shift failure)

**Learned**:
- Reference point: a violinist's wrist exerts ~0.16 N·m of torque, which guided the frog torque range
- With RLPD and some newly collected data, reward reached ~800, against ~400 for purely online training. **RLPD changes the training landscape completely**
- But across simple and complex trajectories, with and without RLPD, **RLPD did not change *asymptotic* performance**. Online-only training gets close (velocity tracking may be weaker)
- RLPD beat pure offline RL
- Complex trajectory + RLPD gave stable velocity and pressure tracking
- **Robot geometry and actuation mattered more than the choice of RL method.** All of this week's improvements came after switching the frog to a torque actuator

## Week 12 - Velocity Commands on Position Servos (October 4-5 2026, ongoing)
**Goals**: Move toward an action space that real hobby/bus servos can execute, and evaluate the policies qualitatively

**Completed**:
- Env reset from the inference viewer
- **Switched the arm back to position servos** with delta-position control (real bus servos only accept position targets). Removed the old `example/` scaffold
- **Velocity-integrated position control** (on branch `vel-ctrl-on-pos-servo`): the policy outputs joint velocities, which are integrated into servo setpoints every physics substep, the way a motor driver emulates a velocity mode on top of position control
  - Anti-windup: the setpoint can't lead the measured joint by more than `max_setpoint_lead` (0.15 rad), so a blocked joint doesn't build up error that's released as a jerk
  - The setpoint lead (setpoint − qpos) is added to the observation, because it's hidden state the command integrates (the same MDP lesson as in Week 10)

**Blockers**: Still comparing pure position vs. velocity-integrated control

**Learned**:

Qualitative evaluation of the velocity-controlled policies
- Pre-training on the complex trajectory prepares the policy for real-world, suddenly changing velocity commands
- The policy trained without RLPD couldn't complete a full push stroke (推弓)
- Oscillation under a zero-velocity command stays within 0.2 m/s at 1 N, and grows roughly linearly with pressure
- All policies handled inappropriate commands, e.g. being told to keep pushing when the arm is already fully extended

Control
- An admittance controller, f = M·ẍ + K(x − x_v) + K_d·ẋ, treats the end effector as a mass-spring system under an external force: solve for the "intended" acceleration and integrate it into a position target. Reading on learned compliance shows the hard part is deciding *where and when* to be stiff, which depends on contact direction and the scene
