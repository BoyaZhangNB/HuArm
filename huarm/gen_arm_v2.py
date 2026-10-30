"""Generate huarm/arm_v2.xml: MuJoCo replica of the ErhuArm v2 Fusion design."""
import re
import numpy as np
import mujoco

ROOT = __import__('os').path.dirname(__import__('os').path.abspath(__file__))
src = open(f'{ROOT}/arm.xml').read()

# ---- CAD data (mm, zero pose, arm base frame) ----
import struct
MESH_DIR = f'{ROOT}/meshes/erhu_arm_v2'
# Printed mass: PLA 1.24 g/cm^3, 3 perimeters (~1.2 mm solid shell) + 15 % infill.
# The thin-walled links are almost all shell, so they stay close to solid density.
RHO, SHELL_MM, INFILL = 1.24e-3, 1.2, 0.15


def printed_kg(mesh):
    d = open(f'{MESH_DIR}/{mesh}.stl', 'rb').read(); n = struct.unpack('<I', d[80:84])[0]
    v = np.frombuffer(d[84:84 + 50 * n], dtype=np.dtype([('n', '<3f4'), ('v', '<9f4'), ('a', '<u2')]))['v']
    v = v.reshape(-1, 3, 3).astype(float); a, b, c = v[:, 0], v[:, 1], v[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    vol = abs(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6)
    shell = min(vol, area * SHELL_MM)
    return RHO * (shell + INFILL * (vol - shell)) / 1000


SERVO_KG = 0.055   # STS3215, 12 V version (same case and weight as the 7.4 V one)
SERVO_STALL_NM = 2.94   # 12 V STS3215: 30 kg*cm stall
# Links between servo axes are capped at 130 mm: J1-J2 and J3-J4 are 130 mm
# (arm.xml: 200 and 250 mm); J2-J5 100, J5-J3 50 and J4-frog 100 mm are unchanged.
L12, L34 = 130.0, 130.0
Z2 = -L12              # J2 and the bow plane
X4 = -150.0 - L34      # J4
ZC = Z2 - 44.0         # J3/J4/frog servo centres, slid 44 mm down their vertical axes
# servo frames: axis point P, output axis A, case direction C (see helpers.py)
SERVOS = {
    'j1': ((0, 0, 0), (0, 1, 0), (0, 0, 1)),
    'j2': ((0, 0, Z2), (0, 1, 0), (0, 0, 1)),
    'j5': ((-100, 0, Z2), (0, 1, 0), (0, 0, -1)),
    'j3': ((-150, 0, ZC), (0, 0, 1), (-1, 0, 0)),
    'j4': ((X4, 0, ZC), (0, 0, 1), (1, 0, 0)),
    'frog': ((X4 - 100, 0, ZC), (0, 0, 1), (1, 0, 0)),
}
# body origins in the base frame (mm); same body tree as arm.xml
BODY_W = {'base': (0, 0, 0), 'link1': (0, 0, 0), 'joint2_body': (0, 0, Z2), 'joint5_body': (-100, 0, Z2),
          'joint3_body': (-150, 0, Z2), 'joint4_body': (X4, 0, Z2), 'end_effector': (X4 - 100, 0, Z2)}
# collision-free limits measured on the CAD (deg), MuJoCo sign convention
RANGE = {'joint1': (-90, 90), 'joint2': (-180, 55), 'joint5': (-10, 112), 'joint3': (-110, 110), 'joint4': (-150, 150)}


def f(v, n=6):
    return ' '.join(f'{x:.{n}g}' if abs(x) > 1e-12 else '0' for x in v)


def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1))
    return q


def servo_geom(name, body):
    P, A, C = (np.array(v, float) for v in SERVOS[name])
    sx = -C; sz = A; sy = np.cross(sz, sx)
    R = np.column_stack([sx, sy, sz])
    org = P + 12.5 * C - np.array(BODY_W[body], float)
    return (f'<geom name="servo_{name}_geom" class="arm_visual" mesh="sts3215" mass="{SERVO_KG}" '
            f'pos="{f(org / 1000)}" quat="{f(mat2quat(R))}" rgba="0.12 0.12 0.13 1"/>')


def link_geom(mesh, body, rgba='0.82 0.80 0.76 1'):
    pos = -np.array(BODY_W[body], float) / 1000
    m = printed_kg(mesh)
    return (f'<geom name="{mesh}_geom" class="arm_visual" mesh="{mesh}" mass="{m:.4f}" '
            f'pos="{f(pos)}" rgba="{rgba}"/>')


# bow_frog frame (child of end_effector) uses euler="-1.5708 3.14 0", same as arm.xml
qf = np.zeros(4); mujoco.mju_euler2Quat(qf, np.array([-1.5708, 3.14, 0.0]), 'xyz')
Rf = np.zeros(9); mujoco.mju_quat2Mat(Rf, qf); Rf = Rf.reshape(3, 3)


def frog_geom(mesh):
    # mesh vertices are base-frame mm; express the base frame inside bow_frog
    pos = Rf.T @ (-np.array(BODY_W['end_effector'], float) / 1000)
    m = printed_kg(mesh)
    return (f'<geom name="{mesh}_geom" class="arm_visual" mesh="{mesh}" mass="{m:.4f}" '
            f'pos="{f(pos)}" quat="{f(mat2quat(Rf.T))}" rgba="0.82 0.80 0.76 1"/>')


def rng(j):
    lo, hi = RANGE[j]
    return f'{np.radians(lo):.4f} {np.radians(hi):.4f}'


# ---- reuse arm.xml's defaults, bow subtree, actuators and contacts verbatim ----
defaults = src[src.index('    <default>'):src.index('\n    </default>\n') + len('\n    </default>')]
defaults = defaults[:-len('\n    </default>')] + '''\n
        <!-- CAD meshes: visual + inertia only. Like the capsules they replace
             (contype=0), the arm never collides. Part masses are 15 %-infill
             PLA prints (1.2 mm solid shell), servos 55 g each, inertia from the
             mesh at uniform density. -->
        <default class="arm_visual">
            <geom type="mesh" contype="0" conaffinity="0" group="1"/>
        </default>
    </default>'''
bow_start = src.index('                                    <body name="bow_frog"')
bow_end = src.index('                                </body>\n\n                            </body>', bow_start)
bow = src[bow_start:bow_end]
bow = bow.replace(
    '<geom name="frog_geom"',
    frog_geom('bow_clamp') + '\n                                        ' + frog_geom('bow_jaw') +
    '\n\n                                        <geom name="frog_geom"')
tail = src[src.index('    <actuator>'):]
# arm position servos: cap every joint at what a 12 V STS3215 can actually deliver
tail = re.sub(r'(<general joint="joint\d"[^>]*?)forcerange="-?[\d.]+ [\d.]+"',
              lambda m_: m_.group(1) + f'forcerange="-{SERVO_STALL_NM} {SERVO_STALL_NM}"', tail)
# Softer position servos: kp is 1/4 of arm.xml's (natural frequency halved), and kv
# restores critical damping at arm_v2's own effective joint inertias (diagonal of
# the mass matrix, the largest over 128 threaded start poses, so no start pose is
# underdamped), counting the 0.5 N*m*s joint damping from the "arm" default:
# kv + 0.5 = 2*sqrt(kp*I).
KP = {'joint1': 150, 'joint2': 150, 'joint5': 100, 'joint3': 100, 'joint4': 50}
I_EFF = {'joint1': 0.05164, 'joint2': 0.03099, 'joint5': 0.01464, 'joint3': 0.01186, 'joint4': 0.00468}
JOINT_DAMPING = 0.5
KV = {j: round(2 * np.sqrt(KP[j] * I_EFF[j]) - JOINT_DAMPING, 2) for j in KP}
for j in KP:
    tail = re.sub(rf'(<general joint="{j}"[^>]*?)gainprm="[^"]*"([^>]*?)biasprm="[^"]*"',
                  lambda m_: m_.group(1) + f'gainprm="{KP[j]} 0 0"' + m_.group(2) +
                  f'biasprm="0 -{KP[j]} -{KV[j]:g}"', tail)
tail = tail.replace('    <actuator>', f'''    <actuator>
        <!-- arm_v2: forcerange of joint1..5 is the 12 V STS3215 stall torque
             (2.94 N*m) instead of arm.xml's tuned 8/4/3/3/2 N*m. kp is softened
             to 1/4 of arm.xml's (natural frequency halved) and kv is critical
             damping at arm_v2's largest effective inertias over the threaded start poses
             ({', '.join(f'{I_EFF[j]:g}' for j in KP)} kg*m^2), counting the
             {JOINT_DAMPING:g} N*m*s joint damping: kv = 2*sqrt(kp*I) - {JOINT_DAMPING:g}.
             The comment below describes arm.xml's original tuning; the filter
             time constants and the frog motor are unchanged. -->''', 1)

assets = '\n'.join(
    f'        <mesh name="{n}" file="meshes/erhu_arm_v2/{n}.stl" scale="0.001 0.001 0.001" inertia="exact"/>'
    for n in ['base_mount', 'link1', 'link2', 'link3', 'link4', 'link5', 'bow_clamp', 'bow_jaw', 'sts3215'])

xml = f'''<mujoco model="erhu_arm_v2">
    <!-- Replica of the ErhuArm v2 Fusion design (Default Project / ErhuArm v2).
         Body/joint names, actuators, bow and contact pairs are identical to
         arm.xml, so ErhuEnv(xml_path="huarm/arm_v2.xml") loads it. What changes:
         - links between servo axes are capped at 130 mm: joint1->joint2 is
           130 mm (arm.xml 200) and joint3->joint4 is 130 mm (arm.xml 250);
         - every link is the printed CAD part plus its STS3215 servo, with mass
           and inertia from the meshes instead of density-1000 capsules;
         - the base mount is fixed: joint1 moves from the "base" body (which in
           arm.xml spun a 0.52 kg sphere with it) to "link1";
         - joint1..5 ranges are the collision-free limits measured on the CAD
           (joint5 stops at 112 deg, where link4 reaches the base mount at joint2 = 0);
         - servo forcerange is the 12 V STS3215 stall torque (2.94 N*m);
         - the J3/J4/frog servos sit 44 mm below the bow plane along their own
           axes, which changes mass distribution but not the joint axes. -->

    <compiler angle="radian" autolimits="true"/>

{defaults}

    <asset>
{assets}
    </asset>

    <worldbody>
        <body name="base" pos="0.3 -0.3 1" childclass="arm" euler="0 0 -2.3708">
            {link_geom('base_mount', 'base')}
            {servo_geom('j1', 'base')}

            <body name="link1" pos="0 0 0">
                <joint name="joint1" type="hinge" axis="0 1 0" range="{rng('joint1')}"/>
                {link_geom('link1', 'link1')}
                {servo_geom('j2', 'link1')}

                <body name="joint2_body" pos="0 0 {-L12 / 1000:g}">
                    <joint name="joint2" type="hinge" axis="0 1 0" range="{rng('joint2')}"/>
                    {link_geom('link2', 'joint2_body')}
                    {servo_geom('j5', 'joint2_body')}

                    <body name="joint5_body" pos="-0.1 0 0">
                        <joint name="joint5" type="hinge" axis="0 1 0" range="{rng('joint5')}"/>
                        {link_geom('link3', 'joint5_body')}
                        {servo_geom('j3', 'joint5_body')}

                        <body name="joint3_body" pos="-0.05 0 0">
                            <joint name="joint3" type="hinge" axis="0 0 1" range="{rng('joint3')}"/>
                            {link_geom('link4', 'joint3_body')}
                            {servo_geom('j4', 'joint3_body')}

                            <body name="joint4_body" pos="{-L34 / 1000:g} 0 0">
                                <joint name="joint4" type="hinge" axis="0 0 1" range="{rng('joint4')}"/>
                                {link_geom('link5', 'joint4_body')}
                                {servo_geom('frog', 'joint4_body')}

                                <body name="end_effector" pos="-0.1 0 0">
{bow}                                </body>

                            </body>

                        </body>

                    </body>

                </body>

            </body>

        </body>

    </worldbody>

{tail}'''
open(f'{ROOT}/arm_v2.xml', 'w').write(xml)
print('written', len(xml))
