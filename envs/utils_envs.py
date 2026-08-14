import functools
from dataclasses import dataclass
from typing import Tuple
import jax
import jax.numpy as jnp
from mujoco import mjx
from mujoco.mjx._src import support as mjx_support
import mujoco
import numpy as np


# Listed in actuator order (joint5 sits between joint2 and joint3 in the
# kinematic chain, and its actuator is declared there too), with the
# unactuated frog hinge last.
ARM_JOINT_NAMES = ("joint1", "joint2", "joint5", "joint3", "joint4", "bow_frog_hinge")

# Geoms that define the threading problem (see `solve_bow_insertion`).
BOW_HAIR_GEOM = "bow_hair_geom"
BOW_STICK_GEOM = "bow_geom_1"
INNER_STRING_GEOM = "string_D_geom"   # the string nearest the arm
OUTER_STRING_GEOM = "string_A_geom"   # the string caught inside the bow loop
FROG_HINGE_JOINT = "bow_frog_hinge"

# Bow/erhu geom pairs that must stay apart while threading. Only the stick and
# the hair collide at all (everything else on the arm is contype=0), and the
# hair/string pairs are deliberately left out -- those are the ones threading
# brings together.
CLEARANCE_PAIRS = (
    (BOW_STICK_GEOM, "sound_box_geom"),
    (BOW_STICK_GEOM, "neck_geom"),
    (BOW_HAIR_GEOM, "sound_box_geom"),
    (BOW_HAIR_GEOM, "neck_geom"),
)


def get_string_contact_point(model, data, string_body_name):
    """
    Returns the world position of the free (bottom) end of a string capsule.
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, string_body_name)
    xmat = data.xmat[bid].reshape(3, 3)
    local_tip = np.array([0.0, 0.0, -0.6])
    return data.xpos[bid] + xmat @ local_tip


def between_strings_target(model, data):
    """
    Midpoint between the two string contact points, lifted clear of the sound box.
    """
    midpoint = (get_string_contact_point(model, data, "string_D")
                + get_string_contact_point(model, data, "string_A")) / 2.0

    sound_box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "sound_box")
    sound_box_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "sound_box_geom")
    sound_box_top_z = data.xpos[sound_box_id][2] + model.geom_size[sound_box_geom_id][0]
    midpoint[2] = max(midpoint[2], sound_box_top_z) + 0.01  # 1 cm clearance
    return midpoint

def weighted_point_and_jacobian(model, data, body_points, dof_idxs):
    """
    Computes weighted-average world position and position Jacobian for given
    (body_name, local_offset, weight) triples. local_offset is a 3-vector in
    the body's own local frame, letting the target point be somewhere other
    than the body origin (e.g. the midpoint of a capsule that is welded to,
    but geometrically offset/rotated from, the body it's attached to).
    """
    point = np.zeros(3)
    J = np.zeros((3, len(dof_idxs)))
    for name, local_offset, w in body_points:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        xmat = data.xmat[bid].reshape(3, 3)
        world_pt = data.xpos[bid] + xmat @ local_offset
        point += w * world_pt
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, world_pt, bid)
        J += w * jacp[:, dof_idxs]
    return point, J


def jacobian_ik(model, data, body_points, target_pos, joint_names,
                max_iters=200, damping=1e-2, step_clip=0.1, tol=1e-4):
    """
    Damped-least-squares IK to position specified target point combination.

    body_points is a list of (body_name, local_offset, weight) triples, see
    weighted_point_and_jacobian.

    joint_names may include unactuated joints (e.g. the passive bow_frog_hinge)
    as extra free DOFs for the solver to use -- their qpos gets solved for and
    written just like any actuated joint, it just never gets copied into ctrl
    (see set_joint_ctrl). Joints with hard limits (jnt_limited) are clamped to
    model.jnt_range after every step so the solver can't swing them past their
    physical stops; unlimited joints (e.g. joint1..5) are unaffected.
    """
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in joint_names]
    dof_idxs = [model.jnt_dofadr[jid] for jid in jids]
    qpos_idxs = [model.jnt_qposadr[jid] for jid in jids]
    limits = [model.jnt_range[jid] if model.jnt_limited[jid] else None for jid in jids]

    for it in range(max_iters):
        mujoco.mj_forward(model, data)
        point, J = weighted_point_and_jacobian(model, data, body_points, dof_idxs)
        err = target_pos - point
        err_norm = np.linalg.norm(err)
        if err_norm < tol:
            return it, err_norm

        dtheta = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(3), err)
        step_norm = np.linalg.norm(dtheta)
        if step_norm > step_clip:
            dtheta *= step_clip / step_norm
        for qidx, d, lim in zip(qpos_idxs, dtheta, limits):
            data.qpos[qidx] += d
            if lim is not None:
                data.qpos[qidx] = np.clip(data.qpos[qidx], lim[0], lim[1])

    mujoco.mj_forward(model, data)
    point, _ = weighted_point_and_jacobian(model, data, body_points, dof_idxs)
    return max_iters, np.linalg.norm(target_pos - point)


def set_joint_ctrl(model, data, joint_names):
    """
    Sets actuator controls to match the current qpos for selected joints.

    The arm's actuators are `dyntype="filter"`, so what the PD law actually
    tracks is the activation state, not ctrl -- and that state starts at zero.
    Setting ctrl alone would leave every actuator pulling towards qpos = 0 for
    the ~50 ms the filter takes to catch up, which is more than enough to drag
    the bow out of the strings, so the activation is seeded to match.
    """
    for jn in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        aid = joint_to_actuator_id(model, jn)
        if aid >= 0:
            data.ctrl[aid] = data.qpos[model.jnt_qposadr[jid]]
            act_adr = model.actuator_actadr[aid]
            if act_adr >= 0:
                data.act[act_adr] = data.ctrl[aid]


def joint_to_actuator_id(model, joint_name):
    """
    Looks up the actuator driving joint_name.
    """
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return -1
    for aid in range(model.nu):
        if (model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT
                and model.actuator_trnid[aid, 0] == jid):
            return aid
    return -1


# ==================================================
# Bow-hair insertion
# ==================================================
#
# A real erhu bow is *threaded* onto the instrument: the hair runs through the
# ~12 mm corridor between the two strings while the stick stays outside the
# outer (A) string, so that string is permanently caught inside the closed
# loop the stick and the hair form. Reaching that configuration is what the
# plain position IK above cannot do reliably once the erhu pose is jittered:
#
#   * it constrains only where the hair *midpoint* lands and leaves the bow's
#     orientation to whatever the 5-DoF arm happens to produce. At the nominal
#     erhu pose that incidentally comes out square; a few cm of erhu jitter
#     yaws the bow 20-40 deg off, which swings the stick across the strings
#     and drives it into the sound box.
#   * it has no notion of the bow colliding with anything, and the bowing
#     point is only ~1 cm above the sound box, so the stick routinely ends up
#     a millimetre or two inside it.
#   * joint1..5 are unlimited revolutes while their actuators are ctrllimited
#     to [-3.14, 3.14], so it can return a pose that is kinematically right
#     and physically uncommandable (see `_joint_addressing`).
#
# `solve_bow_insertion` below drives a full insertion residual instead -- hair
# line through the corridor, hair square to the strings, contact point
# mid-hair, bow clear of the instrument -- searching only poses the arm can
# hold, and accepts a solution only after `bow_insertion_status` confirms the
# threading geometrically. The tolerances a human exploits when threading a
# bow by hand are what make it solvable, and each is a band rather than a
# target: where along the hair the strings sit (`hair_span`), how high up the
# strings to bow (`string_gap_frame`'s `heights`), and how square the bow has
# to be (`axis_tol`).
#
# One thing this cannot fix from the initialisation side: the bow hangs off
# the unactuated, unsprung bow_frog_hinge, and no pose that reaches the
# strings leaves gravity balanced about it (see `bow_insertion_status`), so a
# threaded pose starts unwinding immediately -- the hair is out of the strings
# within ~0.1 s. Holding it needs stiffness on that hinge in the model.


def _capsule_endpoints(model, data, geom_name):
    """World-frame endpoints of a capsule/cylinder geom's axis."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    center = data.geom_xpos[gid]
    axis = data.geom_xmat[gid].reshape(3, 3)[:, 2]
    half_len = model.geom_size[gid][1]
    return center - axis * half_len, center + axis * half_len


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def string_gap_frame(model, data, clearance=0.01, height_band=0.04):
    """
    Describes the corridor between the two strings, in world coordinates:

      center      point on the corridor's centre line at nominal bowing
                  height (just clear of the sound box)
      bottom      the corridor's centre line at the strings' lower end;
                  `bottom + string_axis * h` walks up the centre line
      sep_axis    unit vector from the inner (D) to the outer (A) string
      string_axis unit vector along the strings, pointing up
      bow_axis    unit vector the hair should run along -- perpendicular to
                  both of the above, i.e. straight through the corridor
      half_gap    half the string separation, the lateral room the hair has
      heights     (low, high) band of bowing heights along the centre line
                  that the solver may pick from. Bowing anywhere in the lower
                  few cm of the strings is fine, and that freedom is what lets
                  the bow clear the sound box: at the nominal height the stick
                  passes within a millimetre of the box's shoulder.
      span        length of the strings

    Requires up-to-date kinematics (mj_kinematics / mj_forward).
    """
    d_a, d_b = _capsule_endpoints(model, data, INNER_STRING_GEOM)
    a_a, a_b = _capsule_endpoints(model, data, OUTER_STRING_GEOM)
    # Strings are near-vertical, so order their endpoints by height rather
    # than assuming which way round the geom frame points.
    d_bot, d_top = (d_a, d_b) if d_a[2] < d_b[2] else (d_b, d_a)
    a_bot, a_top = (a_a, a_b) if a_a[2] < a_b[2] else (a_b, a_a)

    sep = a_bot - d_bot
    half_gap = 0.5 * float(np.linalg.norm(sep))
    sep_axis = _unit(sep)
    string_axis = _unit((d_top - d_bot) + (a_top - a_bot))

    sound_box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "sound_box")
    sound_box_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "sound_box_geom")
    box_top_z = data.xpos[sound_box_id][2] + model.geom_size[sound_box_geom_id][0]

    # Slide along the corridor (not straight up in world z) to bowing height,
    # so the point stays exactly between the strings even when the erhu is
    # tilted.
    bottom = 0.5 * (d_bot + a_bot)
    span = float(np.linalg.norm(d_top - d_bot))
    rise = (box_top_z + clearance - bottom[2]) / max(string_axis[2], 1e-6)
    low = float(np.clip(rise, 0.0, span))

    return dict(
        center=bottom + string_axis * low,
        sep_axis=sep_axis,
        string_axis=string_axis,
        bow_axis=_unit(np.cross(sep_axis, string_axis)),
        half_gap=half_gap,
        bottom=bottom,
        heights=(low, min(low + height_band, span)),
        span=span,
    )


def frog_hinge_gravity(model, data, hinge_joint=FROG_HINGE_JOINT):
    """
    Gravity's pull on the passive frog hinge, for the current pose.

    The whole bow hangs off this hinge with nothing but 0.05 N m s/rad of
    damping to hold it, so a pose is only worth initialising to if gravity
    exerts (almost) no torque about the hinge there -- otherwise the bow
    rotates straight back out of the strings. Returns

      torque_norm  gravity torque about the hinge, normalised by
                   m_subtree * |g| * lever arm, so it reads as the sine of
                   how far the bow hangs from its equilibrium (0 = balanced)
      stable       True when that equilibrium is the stable one (bow CoM
                   below the hinge axis) rather than the inverted one
      tilt         sine of the hinge axis' tilt from vertical; near 0 the
                   hinge is neutral -- gravity cannot turn it either way, and
                   the bow stays wherever it is put

    Requires kinematics + CoM (mj_kinematics + mj_comPos, or mj_forward).
    """
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, hinge_joint)
    bid = model.jnt_bodyid[jid]
    axis = data.xaxis[jid]
    lever = data.subtree_com[bid] - data.xanchor[jid]
    mass = float(model.body_subtreemass[bid])
    g = np.asarray(model.opt.gravity)

    torque = float(axis @ np.cross(lever, mass * g))
    lever_perp = lever - (lever @ axis) * axis
    g_perp = g - (g @ axis) * axis
    scale = mass * float(np.linalg.norm(g)) * max(float(np.linalg.norm(lever_perp)), 1e-9)

    tilt = float(np.linalg.norm(g_perp) / (np.linalg.norm(g) + 1e-12))
    return dict(
        torque_norm=torque / scale,
        stable=bool(lever_perp @ g_perp > 0.0),
        tilt=tilt,
    )


def _hair_line(model, data):
    """(origin, unit axis, length) of the bow hair, world frame."""
    h0, h1 = _capsule_endpoints(model, data, BOW_HAIR_GEOM)
    length = float(np.linalg.norm(h1 - h0))
    return h0, (h1 - h0) / (length + 1e-12), length


def _plane_crossing(p0, p1, origin, normal):
    """
    Fraction along p0->p1 at which it crosses the plane through `origin` with
    `normal`, and the crossing point; None if the segment stays on one side.
    """
    denom = normal @ (p1 - p0)
    if abs(denom) < 1e-12:
        return None
    s = float((normal @ (origin - p0)) / denom)
    if not 0.0 <= s <= 1.0:
        return None
    return s, p0 + s * (p1 - p0)


def bow_clearance(model, data, pairs=CLEARANCE_PAIRS, distmax=0.05):
    """
    Signed gap (m) for each pair in `pairs`, clipped at `distmax`. Negative
    means the two geoms overlap.
    """
    return np.array([
        mujoco.mj_geomDistance(
            model, data,
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g1),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g2),
            distmax, None)
        for g1, g2 in pairs
    ])


def _band_excess(x, lo, hi):
    """0 inside [lo, hi], and how far outside it otherwise (signed)."""
    return max(x - hi, 0.0) + min(x - lo, 0.0)


def _closest_between_lines(p0, u, q0, v):
    """
    Closest points between the lines p0 + s*u and q0 + t*v (both unit
    direction). Returns (s, t, offset = q_t - p_s).
    """
    uv = float(u @ v)
    w0 = p0 - q0
    det = 1.0 - uv * uv
    if abs(det) < 1e-9:            # parallel: any s does, take the foot of w0
        s, t = 0.0, float(-(w0 @ v))
    else:
        s = float((-(w0 @ u) + uv * (w0 @ v)) / det)
        t = float(((w0 @ v) - uv * (w0 @ u)) / det)
    return s, t, (q0 + v * t) - (p0 + u * s)


def bow_insertion_residual(model, data, gap, hair_span=(0.3, 0.7),
                           tol_offset=0.004, tol_axis=0.30, tol_span=0.15,
                           tol_height=0.015, tol_clear=0.002,
                           clearance_margin=0.004, pairs=CLEARANCE_PAIRS):
    """
    Residual vector whose zero is a threaded bow. Every entry is divided by
    that constraint's own tolerance, so each reads as "how many times its
    budget this term is out by" and least-squares trades them off sensibly
    instead of by accident of units:

      [0:3] shortest offset between the hair *line* and the corridor's centre
            line -- zero when the hair runs through the gap, at whatever point
            along either line
      [3]   hair axis vs. the strings: 0 when the bow lies square across them
      [4]   hair axis vs. the string separation: 0 when the bow runs straight
            through the corridor instead of skewing along it
      [5]   how far the contact point has slid outside `hair_span`
      [6]   how far the bowing height has slid outside `gap["heights"]`
      [7:]  how far each pair in `pairs` is inside `clearance_margin` of
            touching, and zero once it is clear

    Terms [3:7] are the tolerances a human exploits when threading a bow by
    hand, which is why they are bands rather than targets: the bow may cross
    anywhere along the middle of the hair, at any height in the lower few cm
    of the strings, and a few degrees off square. Only the offset is tight --
    the hair has barely 5 mm of room either side of the corridor centre.

    Those bands are also what makes the clearance terms satisfiable. At the
    nominal bowing height the stick passes within a millimetre of the sound
    box's shoulder, so with a fixed target point the solver can only choose
    between threading the strings and clipping the box; free to ride a
    centimetre higher up the corridor, it can do both.
    """
    h0, axis, length = _hair_line(model, data)
    s, height, offset = _closest_between_lines(
        h0, axis, gap["bottom"], gap["string_axis"])

    intrusion = np.minimum(bow_clearance(model, data, pairs) - clearance_margin, 0.0)

    return np.concatenate([
        offset / tol_offset,
        [float(axis @ gap["string_axis"]) / tol_axis],
        [float(axis @ gap["sep_axis"]) / tol_axis],
        [_band_excess(s / length, *hair_span) / tol_span],
        [_band_excess(height, *gap["heights"]) / tol_height],
        intrusion / tol_clear,
    ])


def bow_insertion_status(model, data, gap=None, clearance=0.0015,
                         hair_span=(0.15, 0.85), axis_tol=0.35,
                         torque_tol=0.05, penetration_tol=1e-4,
                         require_balanced=False):
    """
    Checks whether the bow is genuinely threaded onto the strings, rather than
    merely close to them. Reports each condition separately so a caller can
    see *how* a candidate pose failed:

      in_gap      the hair crosses the strings' plane inside the corridor,
                  with `clearance` (m) to spare on both sides
      in_span     ... at a height where the strings actually are
      on_hair     ... at a point inside `hair_span` of the hair's length, not
                  off near the frog or the tip
      threaded    the stick crosses that plane beyond the outer string, i.e.
                  the string is caught inside the stick/hair loop the way it
                  is on a real erhu (this is the condition a bow cannot be
                  flipped out of)
      square      the hair is within `axis_tol` rad of perpendicular to the
                  strings and to their separation
      balanced    gravity holds the bow where it is (see `frog_hinge_gravity`)
      clear       nothing is interpenetrating

    `balanced` is reported but, unless `require_balanced` is set, is left out
    of `ok`, because no pose satisfies it *and* the rest: reaching down to the
    strings from a base 0.55 m above them forces |joint1 + joint2 + joint5| >= ~0.6
    rad, which tilts the frog hinge's axis at least that far from vertical,
    and the bow's centre of mass sits within 4 deg of the hair axis -- so the
    angle gravity would hold the bow at is one drooping ~35 deg out of the
    strings. Holding the bow flat needs a hinge that resists that torque
    (`stiffness`/`springref` on bow_frog_hinge, or removing the joint); see
    `frog_hinge_gravity` for the torque a given pose leaves unbalanced.

    Runs mj_forward, since the `clear` condition needs collision detection.
    """
    mujoco.mj_forward(model, data)
    gap = string_gap_frame(model, data) if gap is None else gap

    center, sep_axis = gap["center"], gap["sep_axis"]
    normal = gap["bow_axis"]  # normal of the plane containing both strings
    st = {}

    h0, h1 = _capsule_endpoints(model, data, BOW_HAIR_GEOM)
    hair_cross = _plane_crossing(h0, h1, center, normal)
    st["crosses"] = hair_cross is not None
    if hair_cross is None:
        st["ok"] = False
        return st

    s, point = hair_cross
    lateral = float((point - center) @ sep_axis)
    height = float((point - gap["bottom"]) @ gap["string_axis"])
    st.update(hair_s=s, lateral=lateral, height=height)

    stick_cross = _plane_crossing(*_capsule_endpoints(model, data, BOW_STICK_GEOM),
                                  center, normal)
    st["stick_lateral"] = None if stick_cross is None else float(
        (stick_cross[1] - center) @ sep_axis)

    _, axis, _ = _hair_line(model, data)
    st["axis_error"] = float(np.arcsin(np.clip(
        np.linalg.norm([axis @ gap["string_axis"], axis @ sep_axis]), 0.0, 1.0)))

    hinge = frog_hinge_gravity(model, data)
    st.update(torque_norm=hinge["torque_norm"], hinge_tilt=hinge["tilt"])

    # The bow-hair/string contact pairs carry a deliberately large `gap` (see
    # arm.xml), so mujoco reports them as contacts long before anything
    # actually overlaps; the corridor test above is what governs the hair.
    # Every other pair -- notably stick vs. sound box -- must be clear.
    hair_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, BOW_HAIR_GEOM)
    st["penetrations"] = [
        (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[i].geom1),
         mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[i].geom2),
         float(data.contact[i].dist))
        for i in range(data.ncon)
        if data.contact[i].dist < -penetration_tol
        and hair_gid not in (data.contact[i].geom1, data.contact[i].geom2)
    ]

    st["in_gap"] = abs(lateral) < gap["half_gap"] - clearance
    st["in_span"] = 0.0 < height < gap["span"]
    st["on_hair"] = hair_span[0] < s < hair_span[1]
    st["threaded"] = (st["stick_lateral"] is not None
                      and st["stick_lateral"] > gap["half_gap"] + clearance)
    st["square"] = st["axis_error"] < axis_tol
    st["balanced"] = (abs(hinge["torque_norm"]) < torque_tol
                      and (hinge["stable"] or hinge["tilt"] < 0.05))
    st["clear"] = not st["penetrations"]
    st["ok"] = bool(st["in_gap"] and st["in_span"] and st["on_hair"]
                    and st["threaded"] and st["square"] and st["clear"]
                    and (st["balanced"] or not require_balanced))
    return st


def _joint_addressing(model, joint_names, respect_ctrlrange=True):
    """
    qpos addresses and the box each joint may be solved inside.

    joint1..5 are unlimited revolutes, so nothing stops a solver from walking
    one of them a few turns away -- kinematically identical, but the actuators
    are ctrllimited to [-3.14, 3.14], so `set_joint_ctrl` would then command a
    pose the arm swings right out of. The box is therefore the joint's own
    range intersected with what its actuator can actually hold.
    """
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in joint_names]
    qpos_idxs = np.array([model.jnt_qposadr[jid] for jid in jids])

    lo, hi = [], []
    for jid, jn in zip(jids, joint_names):
        j_lo, j_hi = ((model.jnt_range[jid][0], model.jnt_range[jid][1])
                      if model.jnt_limited[jid] else (-np.inf, np.inf))
        aid = joint_to_actuator_id(model, jn) if respect_ctrlrange else -1
        if aid >= 0 and model.actuator_ctrllimited[aid]:
            j_lo = max(j_lo, model.actuator_ctrlrange[aid][0])
            j_hi = min(j_hi, model.actuator_ctrlrange[aid][1])
        lo.append(j_lo)
        hi.append(j_hi)
    return qpos_idxs, np.array(lo), np.array(hi)


def solve_bow_insertion(model, data, arm_joint_names=ARM_JOINT_NAMES, q_init=None,
                        max_iters=120, restarts=12, restart_scale=0.35, rng=None,
                        step_clip=0.5, tol_reg=3.0,
                        residual_kwargs=None, status_kwargs=None):
    """
    Solves for an arm pose that threads the bow hair between the erhu strings.

    Damped Gauss-Newton (Levenberg-Marquardt) with a numerical Jacobian, over
    the five actuated arm joints and the passive frog hinge -- the hinge sets
    which way the bow points about its own mount, so the threading needs it,
    and it is bounded by its own limits while the arm joints are bounded by
    what their actuators can command (`_joint_addressing`).

    The residual is smooth but its zero set is neither unique nor always
    reachable, so each solve is verified with `bow_insertion_status` and
    retried from a randomly perturbed seed on failure. `q_init` (defaults to
    the current qpos) is the first seed, and also the pose `tol_reg` pulls the
    answer back towards; passing the previous episode's solution there makes
    the retries almost never fire and keeps start poses consistent.

    Returns a dict with the accepted `qpos`, the final `status`, and how many
    `attempts`/`iters` it took. On failure the best-scoring pose found is left
    in `data` and returned with `status["ok"] == False`.
    """
    rng = np.random.default_rng() if rng is None else rng
    residual_kwargs = residual_kwargs or {}
    status_kwargs = status_kwargs or {}
    qpos_idxs, lo, hi = _joint_addressing(model, arm_joint_names)

    mujoco.mj_kinematics(model, data)
    gap = string_gap_frame(model, data)

    seed = np.array(data.qpos[qpos_idxs] if q_init is None else q_init, dtype=float)
    seed = np.clip(seed, lo, hi)

    def evaluate(q):
        data.qpos[qpos_idxs] = np.clip(q, lo, hi)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        # Threading leaves a whole one-parameter family of arm poses free, so
        # a weak pull back towards the seed picks the nearest member of it
        # instead of an arbitrary one. That keeps successive episodes' start
        # poses close together -- without it the passive hinge in particular
        # comes back anywhere in its +-1.57 range.
        return np.concatenate([
            bow_insertion_residual(model, data, gap, **residual_kwargs),
            (q - seed) / tol_reg,
        ])

    best = (np.inf, seed.copy(), None)
    total_iters = 0

    for attempt in range(restarts + 1):
        if attempt == 0:
            q = np.clip(seed, lo, hi)
        else:
            # Widen the search the longer it takes; the arm joints are
            # unlimited revolutes, so a large kick is a legitimate new branch
            # rather than a wasted seed.
            scale = restart_scale * (1.0 + attempt)
            q = np.clip(seed + rng.normal(0.0, scale, size=seed.shape), lo, hi)

        r = evaluate(q)
        cost = 0.5 * float(r @ r)
        lam = 1e-3

        for _ in range(max_iters):
            total_iters += 1
            J = np.empty((r.size, q.size))
            eps = 1e-6
            for i in range(q.size):
                dq = np.zeros_like(q)
                dq[i] = eps
                J[:, i] = (evaluate(q + dq) - r) / eps
            evaluate(q)  # restore the perturbed kinematics

            JTJ, JTr = J.T @ J, J.T @ r
            improved = False
            for _ in range(10):
                try:
                    step = np.linalg.solve(JTJ + lam * np.diag(np.diag(JTJ) + 1e-9), -JTr)
                except np.linalg.LinAlgError:
                    lam *= 10.0
                    continue
                norm = np.linalg.norm(step)
                if norm > step_clip:
                    step *= step_clip / norm
                q_try = np.clip(q + step, lo, hi)
                r_try = evaluate(q_try)
                cost_try = 0.5 * float(r_try @ r_try)
                if cost_try < cost:
                    q, r, cost = q_try, r_try, cost_try
                    lam = max(lam * 0.3, 1e-9)
                    improved = True
                    break
                lam *= 5.0
            if not improved or np.linalg.norm(step) < 1e-10:
                break

        evaluate(q)
        status = bow_insertion_status(model, data, gap, **status_kwargs)
        if status["ok"]:
            return dict(qpos=q.copy(), status=status, attempts=attempt + 1,
                        iters=total_iters, cost=cost, gap=gap)
        if cost < best[0]:
            best = (cost, q.copy(), status)

    cost, q, status = best
    data.qpos[qpos_idxs] = q
    status = bow_insertion_status(model, data, gap, **status_kwargs)
    return dict(qpos=q, status=status, attempts=restarts + 1, iters=total_iters,
                cost=cost, gap=gap)


def describe_insertion(status):
    """One-line summary of a `bow_insertion_status` dict, for logging."""
    if not status.get("crosses", False):
        return "FAILED: the hair never crosses the plane of the strings"
    failed = [name for name in ("in_gap", "in_span", "on_hair", "threaded",
                                "square", "clear")
              if not status.get(name, False)]
    stick = ("stick never crosses it" if status["stick_lateral"] is None
             else f"stick {status['stick_lateral'] * 1e3:+.1f} mm outside it")
    return (f"{'OK' if status['ok'] else 'FAILED ' + ','.join(failed)}: "
            f"hair {status['lateral'] * 1e3:+.2f} mm off the corridor centre at "
            f"{status['hair_s']:.0%} along the hair, {stick}, "
            f"{np.degrees(status['axis_error']):.1f} deg off square, "
            f"hinge {np.degrees(np.arcsin(status['hinge_tilt'])):.1f} deg off "
            f"vertical (torque {status['torque_norm']:+.3f})")


def perturb_arm_joints(model, data, arm_joint_names=ARM_JOINT_NAMES,
                       joint_noise_std=0.005, gap=None, tries=8, rng=None,
                       clearance=0.0025, status_kwargs=None):
    """
    Adds Gaussian noise (rad) to the solved arm pose so episodes do not all
    start from an identical configuration, while keeping the bow threaded --
    the corridor is only ~12 mm wide, and 5 mrad on every joint moves the hair
    by a few mm, so a noise draw that pushes it out of the strings is rejected
    and redrawn. A draw is only kept if it still leaves `clearance` to each
    string -- more than `bow_insertion_status` demands by default, so jitter
    cannot walk the pose right up to the edge of the corridor. Returns True
    once a draw sticks (or when there is no noise to add), False if all
    `tries` missed and the unperturbed pose was restored.
    """
    if joint_noise_std <= 0:
        return True
    rng = np.random.default_rng() if rng is None else rng
    qpos_idxs, lo, hi = _joint_addressing(model, arm_joint_names)
    q0 = np.array(data.qpos[qpos_idxs], dtype=float)
    status_kwargs = dict(clearance=clearance, **(status_kwargs or {}))

    for _ in range(tries):
        data.qpos[qpos_idxs] = np.clip(
            q0 + rng.normal(0.0, joint_noise_std, size=q0.shape), lo, hi)
        if bow_insertion_status(model, data, gap, **status_kwargs)["ok"]:
            return True

    data.qpos[qpos_idxs] = q0
    mujoco.mj_forward(model, data)
    return False


def insert_hair_between_strings(model, data, arm_joint_names=ARM_JOINT_NAMES,
                                joint_noise_std=0.0, rng=None, verbose=True,
                                **solver_kwargs):
    """
    Places the bow so its hair is threaded between the erhu strings, and bakes
    the resulting pose into the actuator targets so the arm holds it.

    See `solve_bow_insertion` for the solve itself and `bow_insertion_status`
    for what "threaded" is checked to mean. `joint_noise_std` (rad) jitters
    the solved pose without breaking the threading (`perturb_arm_joints`).
    """
    result = solve_bow_insertion(model, data, arm_joint_names, rng=rng, **solver_kwargs)
    if joint_noise_std > 0 and result["status"]["ok"]:
        perturb_arm_joints(model, data, arm_joint_names, joint_noise_std,
                           gap=result["gap"], rng=rng)
        result["status"] = bow_insertion_status(model, data, result["gap"])

    set_joint_ctrl(model, data, arm_joint_names)
    if verbose:
        print(f"Bow insertion: {describe_insertion(result['status'])} "
              f"[{result['attempts']} attempt(s), {result['iters']} iterations]")
    return result


def init_huarm(model, data, id_dict=None, joint_noise_std=0.0):
    """
    Prepares the model/data for a new episode.
    """
    mujoco.mj_forward(model, data)
    insert_hair_between_strings(model, data, joint_noise_std=joint_noise_std)
    return model, data


# ==================================================
def arm_joint_indices(mj_model, arm_joint_names=ARM_JOINT_NAMES):
    """
    Precomputes, once per model (e.g. at env init), the static index arrays
    needed by `domain_randomize_jax` / `_set_joint_ctrl_jax` each reset --
    qpos addresses for all arm joints, plus the (actuator id, qpos address)
    pairs for the subset of those joints that are actuated. Passing these in
    as args avoids repeating `mj_name2id`/`joint_to_actuator_id` python
    lookups on every call.
    """
    jids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in arm_joint_names]
    qpos_idxs = jnp.asarray([mj_model.jnt_qposadr[jid] for jid in jids])

    aids = [joint_to_actuator_id(mj_model, jn) for jn in arm_joint_names]
    ctrl_aids = jnp.asarray([a for a in aids if a >= 0], dtype=jnp.int32)
    ctrl_qpos_idxs = jnp.asarray(
        [mj_model.jnt_qposadr[jid] for jid, a in zip(jids, aids) if a >= 0]
    )
    # Activation addresses for the subset of those actuators that have an
    # activation state (the arm's are dyntype="filter"), so a reset can seed
    # it -- see `set_joint_ctrl` for why leaving it at zero pulls the arm off
    # the pose it was just placed in.
    act_pairs = [(mj_model.actuator_actadr[a], mj_model.jnt_qposadr[jid])
                 for jid, a in zip(jids, aids)
                 if a >= 0 and mj_model.actuator_actadr[a] >= 0]
    act_idxs = jnp.asarray([p[0] for p in act_pairs], dtype=jnp.int32)
    act_qpos_idxs = jnp.asarray([p[1] for p in act_pairs])
    return qpos_idxs, ctrl_aids, ctrl_qpos_idxs, (act_idxs, act_qpos_idxs)


def _set_joint_ctrl_jax(mjx_data, ctrl_aids, ctrl_qpos_idxs, act_idxs=None):
    """
    JAX/MJX counterpart of set_joint_ctrl: sets actuator controls -- and, for
    filtered actuators, their activation state -- to match the current qpos
    for selected joints, given their precomputed index arrays (see
    `arm_joint_indices`).
    """
    ctrl = mjx_data.ctrl.at[ctrl_aids].set(mjx_data.qpos[ctrl_qpos_idxs])
    mjx_data = mjx_data.replace(ctrl=ctrl)
    if act_idxs is not None:
        idxs, qpos_idxs = act_idxs
        mjx_data = mjx_data.replace(
            act=mjx_data.act.at[idxs].set(mjx_data.qpos[qpos_idxs]))
    return mjx_data

def domain_randomize_jax(mjx_model, mjx_data, rng, erhu_pose_pool,
                          arm_qpos_idxs, arm_ctrl_aids, arm_ctrl_qpos_idxs,
                          arm_act_idxs=None,
                          friction_range=(0.7, 1.3),
                          mass_range=(0.8, 1.2),
                          damping_range=(0.8, 1.2),
                          joint_noise_std=0.0):
    """Samples this episode's randomized dynamics params -- contact friction,
    body mass, joint damping.

    `arm_qpos_idxs`, `arm_ctrl_aids`, `arm_ctrl_qpos_idxs`, `arm_act_idxs` are
    the static index arrays from `arm_joint_indices(mj_model,
    arm_joint_names)`, precomputed once at env init and passed in here to
    avoid recomputing them (and the underlying mj_name2id lookups) on every
    reset.

    `joint_noise_std` now defaults to 0 because the poses drawn from the pool
    are threaded through a corridor barely 12 mm wide: a few mrad on each
    joint moves the hair several mm and lifts it straight out of the strings,
    and nothing here can check for that inside jit. The equivalent jitter is
    instead applied -- and verified -- while the pool is built, so every entry
    is a pose known to be threaded (see `build_erhu_pose_pool`)."""
    friction_rng, mass_rng, damping_rng, pool_rng, joint_rng = jax.random.split(rng, 5)

    friction = mjx_model.geom_friction * jax.random.uniform(
        friction_rng, mjx_model.geom_friction.shape,
        minval=friction_range[0], maxval=friction_range[1],
    )
    body_mass = mjx_model.body_mass * jax.random.uniform(
        mass_rng, mjx_model.body_mass.shape,
        minval=mass_range[0], maxval=mass_range[1],
    )
    dof_damping = mjx_model.dof_damping * jax.random.uniform(
        damping_rng, mjx_model.dof_damping.shape,
        minval=damping_range[0], maxval=damping_range[1],
    )

    pose = sample_erhu_pose_pool(erhu_pose_pool, pool_rng)
    body_pos, body_quat, arm_qpos = pose["body_pos"], pose["body_quat"], pose["arm_qpos"]

    dr_params = dict(
        geom_friction=friction,
        body_mass=body_mass,
        dof_damping=dof_damping,
        body_pos=body_pos,
        body_quat=body_quat,
    )
    randomized_model = mjx_model.replace(**dr_params)

    if joint_noise_std > 0:
        arm_qpos = arm_qpos + joint_noise_std * jax.random.normal(joint_rng, arm_qpos.shape)

    mjx_data = mjx_data.replace(qpos=mjx_data.qpos.at[arm_qpos_idxs].set(arm_qpos))
    mjx_data = mjx.forward(randomized_model, mjx_data)
    mjx_data = _set_joint_ctrl_jax(mjx_data, arm_ctrl_aids, arm_ctrl_qpos_idxs,
                                   arm_act_idxs)

    return dr_params, mjx_data

# ==================================================

def build_erhu_pose_pool(mj_model, mjx_model, rng, pool_size,
                          arm_joint_names=ARM_JOINT_NAMES,
                          erhu_pos_std=0.05, erhu_tilt_std=0.05,
                          joint_noise_std=0.005, max_rejects=0.5, seed=0,
                          verbose=True):
    """
    Precomputes `pool_size` random erhu placements (erhu_root's body_pos/
    body_quat, jittered by a small translation and tilt) and, for each, an arm
    pose with the bow threaded between the strings (`solve_bow_insertion`).

    Every entry is verified with `bow_insertion_status` before it goes in, so
    reset() cannot draw a pose whose bow is outside the strings. Placements
    the arm cannot thread (a few percent, at the tails of the jitter) are
    redrawn rather than admitted, up to `max_rejects` * `pool_size` of them.
    Each solve is seeded from the previous one, which is close enough that the
    retry path almost never fires.

    `joint_noise_std` (rad) is the per-episode start-pose jitter, applied here
    rather than in `domain_randomize_jax` so that each draw can be checked for
    still being threaded -- see that function.

    Returns a dict pytree {"body_pos", "body_quat", "arm_qpos"}, each
    array stacked with a leading `pool_size` axis, for
    `sample_erhu_pose_pool` to draw from inside reset().
    """
    erhu_root_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "erhu_root")
    qpos_idxs = [mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jn)]
                 for jn in arm_joint_names]

    # mj_model.body_pos/body_quat are mutated in place per sample (mujoco
    # has no cheap immutable "with field set" for MjModel) and restored
    # afterwards, so the shared mj_model object comes out exactly as it
    # went in regardless of how this loop exits.
    base_body_pos = np.array(mjx_model.body_pos)   # (nbody, 3), full-model template
    base_body_quat = np.array(mjx_model.body_quat)  # (nbody, 4)
    nominal_pos = base_body_pos[erhu_root_id].copy()
    nominal_quat = base_body_quat[erhu_root_id].copy()
    solver_rng = np.random.default_rng(seed)

    body_pos_pool, body_quat_pool, arm_qpos_pool = [], [], []
    warm_start, rejected, jitter_rejected, last_failure = None, 0, 0, None
    try:
        while len(arm_qpos_pool) < pool_size:
            if verbose:
                print(f"Building erhu pose pool: sample "
                      f"{len(arm_qpos_pool) + 1}/{pool_size}", end="\r")
            rng, pos_rng, tilt_rng = jax.random.split(rng, 3)

            pos_noise = erhu_pos_std * np.asarray(
                [*jax.random.uniform(pos_rng, (1,), minval=-1.0, maxval=1.0), 0.0, 0.0])
            body_pos = nominal_pos + pos_noise
            # Small-angle random tilt composed onto the nominal orientation,
            # as a unit quaternion (w, x, y, z); avoids re-normalizing a
            # hand-built axis so the result stays a valid rotation even at
            # sampled extremes.
            tilt_axis_angle = erhu_tilt_std * np.asarray(
                [0.0, 0.0, *jax.random.uniform(tilt_rng, (1,), minval=-1.0, maxval=1.0)])
            tilt_angle = float(np.linalg.norm(tilt_axis_angle)) + 1e-8
            tilt_axis = tilt_axis_angle / tilt_angle
            tilt_quat = np.concatenate([[np.cos(tilt_angle / 2)], tilt_axis * np.sin(tilt_angle / 2)])
            w0, x0, y0, z0 = nominal_quat
            w1, x1, y1, z1 = tilt_quat
            body_quat = np.array([
                w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            ])

            mj_model.body_pos[erhu_root_id] = body_pos
            mj_model.body_quat[erhu_root_id] = body_quat

            data = mujoco.MjData(mj_model)
            mujoco.mj_forward(mj_model, data)
            result = solve_bow_insertion(mj_model, data, arm_joint_names,
                                         q_init=warm_start, rng=solver_rng)
            if not result["status"]["ok"]:
                # An erhu placement the arm cannot thread the bow through is
                # not a start state worth keeping -- draw another one. This
                # truncates the tails of the erhu-pose distribution (the
                # placements far enough out that the bow can only reach them
                # at a bad angle), which is the intended trade: every pose in
                # the pool is one an episode can legitimately start from.
                rejected += 1
                last_failure = describe_insertion(result["status"])
                if rejected > max_rejects * pool_size:
                    raise RuntimeError(
                        f"Gave up building the erhu pose pool after {rejected} "
                        f"unthreadable placements; last was {last_failure}")
                continue

            warm_start = result["qpos"]
            jitter_rejected += not perturb_arm_joints(
                mj_model, data, arm_joint_names, joint_noise_std,
                gap=result["gap"], rng=solver_rng)

            body_pos_full = base_body_pos.copy()
            body_pos_full[erhu_root_id] = body_pos
            body_quat_full = base_body_quat.copy()
            body_quat_full[erhu_root_id] = body_quat

            body_pos_pool.append(body_pos_full)
            body_quat_pool.append(body_quat_full)
            arm_qpos_pool.append(np.array([data.qpos[qi] for qi in qpos_idxs]))
    finally:
        mj_model.body_pos[erhu_root_id] = nominal_pos
        mj_model.body_quat[erhu_root_id] = nominal_quat

    if verbose:
        print(f"Built erhu pose pool: {pool_size} poses, all with the bow "
              f"threaded between the strings"
              + (f"; redrew {rejected} unthreadable erhu placement(s), last "
                 f"was {last_failure}" if rejected else "")
              + (f"; {jitter_rejected} kept unjittered to stay threaded"
                 if jitter_rejected else ""))

    return dict(
        body_pos=jnp.asarray(np.stack(body_pos_pool)),
        body_quat=jnp.stack(body_quat_pool),
        arm_qpos=jnp.stack(arm_qpos_pool),
    )


def sample_erhu_pose_pool(pool, rng):
    """Draws one (body_pos, body_quat, arm_qpos) triple from a pool built by
    `build_erhu_pose_pool`: a random index plus a gather on each leaf --
    cheap enough to call every reset(), unlike the IK solve it replaces."""
    pool_size = pool["body_pos"].shape[0]
    idx = jax.random.randint(rng, (), 0, pool_size)
    return jax.tree_util.tree_map(lambda x: x[idx], pool)