"""Pose-goal follower for the G1, ported from G1_sim/g1_sim/goto_trace.py (sim-verified
on the GR00T WBC with ground-truth pose). Same control law and limits; gains and
tolerances retuned for the real robot's lag; here the pose is PhaseSpace and the output goes to LocoClient.SetVelocity.

Far from the goal (> NEAR): turn toward it, walk forward once the bearing error is
under TURN_FIRST. Near: body-frame holonomic step toward it while turning to the
goal heading. A goal is reached after holding POS_TOL / YAW_TOL for SETTLE_S.
"""

import math

POS_TOL = 0.20                 # m; real G1 stalls 0.1-0.3 m short (walk deadband), 0.05 timed out
YAW_TOL = math.radians(1.0)    # rad
SETTLE_S = 0.0                 # s; the real G1 lags ~3 s and drifts out while holding, so a goal counts once entered
GOAL_TIMEOUT_S = 60.0          # s per goal
NEAR = 0.5                     # m
TURN_FIRST = math.radians(35)  # rad
VX = (-0.3, 0.5)               # m/s
VY = 0.3                       # m/s
WZ = 0.6                       # rad/s
K_POS, K_YAW = 0.3, 0.3  # real G1: ~3 s velocity-to-motion lag; 1.0/1.2 circled the goal (2026-10-07 run 4c86ae77)
MIN_CMD = 0.08                 # below the walk deadband the G1 just stands


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def control(gx, gy, gyaw, x, y, yaw):
    """(vx, vy, wz, phase) body-frame command toward goal (gx, gy, gyaw); angles in rad."""
    dx, dy = gx - x, gy - y
    dist = math.hypot(dx, dy)
    if dist > NEAR:
        phase = "walk"
        bearing = wrap(math.atan2(dy, dx) - yaw)
        wz = clip(K_YAW * bearing, -WZ, WZ)
        vx = 0.0 if abs(bearing) > TURN_FIRST else clip(K_POS * dist, *VX) * math.cos(bearing)
        vy = 0.0
    else:
        phase = "align"
        c, s = math.cos(yaw), math.sin(yaw)
        vx = clip(K_POS * (c * dx + s * dy), *VX)
        vy = clip(K_POS * (-s * dx + c * dy), -VY, VY)
        wz = clip(K_YAW * wrap(gyaw - yaw), -WZ, WZ)
    norm = math.sqrt(vx * vx + vy * vy + wz * wz)
    if 0.0 < norm < MIN_CMD:
        vx, vy, wz = (v * MIN_CMD / norm for v in (vx, vy, wz))
    return vx, vy, wz, phase


def in_tolerance(gx, gy, gyaw, x, y, yaw):
    return math.hypot(gx - x, gy - y) < POS_TOL and abs(wrap(gyaw - yaw)) < YAW_TOL


class GoalRunner:
    """Feed step() the measured pose; it returns (vx, vy, wz) and advances through the goals."""

    def __init__(self, goals, final_yaw=True):
        # final_yaw=False: drive position only; once inside POS_TOL the goal sits in phase "turn"
        # (zero command) until the caller turns to the facing itself and calls finish().
        self.final_yaw = final_yaw
        self.goals = [dict(g, status="pending") for g in goals]
        self.i, self.phase = 0, "walk"
        self.t_goal = self.t_in_tol = None

    @property
    def done(self):
        return self.i >= len(self.goals)

    def step(self, t, x, y, yaw):
        if self.done:
            return 0.0, 0.0, 0.0
        g = self.goals[self.i]
        gyaw = math.radians(g["yaw_deg"])
        if self.t_goal is None:
            self.t_goal, g["status"] = t, "active"
        if not self.final_yaw:
            # a rotate call turns in place: its x, y is only the pose the planner saw
            if (self.phase == "turn" or g.get("call") == "rotate"
                    or math.hypot(g["x"] - x, g["y"] - y) < POS_TOL):
                self.phase = "turn"
                return 0.0, 0.0, 0.0
            gyaw = yaw  # facing is the caller's turn, after arrival
        if in_tolerance(g["x"], g["y"], gyaw, x, y, yaw):
            self.t_in_tol = t if self.t_in_tol is None else self.t_in_tol
            if t - self.t_in_tol >= SETTLE_S:
                self.finish(t, "reached", x, y, yaw)
            return 0.0, 0.0, 0.0
        if t - self.t_goal > GOAL_TIMEOUT_S:
            self.finish(t, "timeout", x, y, yaw)
            return 0.0, 0.0, 0.0
        self.t_in_tol = None
        vx, vy, wz, self.phase = control(g["x"], g["y"], gyaw, x, y, yaw)
        return vx, vy, wz

    def finish(self, t, status, x, y, yaw):
        g = self.goals[self.i]
        g.update(status=status, duration_s=round(t - self.t_goal, 2),
                 reached_pose=[round(x, 3), round(y, 3), round(math.degrees(yaw), 1)],
                 pos_err_m=round(math.hypot(g["x"] - x, g["y"] - y), 3),
                 yaw_err_deg=round(math.degrees(wrap(math.radians(g["yaw_deg"]) - yaw)), 1))
        self.i += 1
        self.t_goal = self.t_in_tol = None
        self.phase = "walk"


def predict(goals, x, y, yaw_deg, dt=0.1):
    """Kinematic rollout of the controller from (x, y, yaw): the velocity preview.
    Returns (path [(x, y)], per-goal [{duration_s, max_vx, max_vy, max_wz, status}])."""
    run, yaw, t = GoalRunner(goals), math.radians(yaw_deg), 0.0
    path, stats = [(x, y)], [{"max_vx": 0.0, "max_vy": 0.0, "max_wz": 0.0} for _ in goals]
    limit = GOAL_TIMEOUT_S * max(1, len(goals)) + 1
    while not run.done and t < limit:
        i = run.i
        vx, vy, wz = run.step(t, x, y, yaw)
        if i < len(stats):
            st = stats[i]
            st["max_vx"], st["max_vy"] = max(st["max_vx"], abs(vx)), max(st["max_vy"], abs(vy))
            st["max_wz"] = max(st["max_wz"], abs(wz))
        x += (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
        y += (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
        yaw += wz * dt
        t += dt
        path.append((x, y))
    for st, g in zip(stats, run.goals):
        st.update(duration_s=g.get("duration_s"), status=g["status"])
    return path, stats
