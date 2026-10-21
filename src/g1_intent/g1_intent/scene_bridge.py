"""intent-sim import + PhaseSpace <-> intent-sim canvas conversion.

intent-sim's canvas is y-down with the origin at the top-left; facing_deg is
CCW from east with 90 = north (= canvas up). PhaseSpace is x/y on the floor,
z up, yaw CCW from +x. With (ox, oy) the PhaseSpace point placed at the canvas
top-left:

    canvas_x = x - ox        canvas_y = oy - y        facing_deg = yaw_deg

Default (ox, oy) = (-5, 3.5) centres the 10 x 7 m canvas on the PhaseSpace origin.
"""

import math
import os
import sys
import time
from pathlib import Path


def import_intent_sim(intent_sim_dir: str):
    """Put intent-sim (and its .venv's packages: openai, pygame, dotenv) on sys.path.

    The venv is APPENDED so ROS / workspace packages (numpy etc.) win any clash;
    intent-sim's own modules are prepended so its top-level `config`, `llm`,
    `sim`, `experiments` resolve to it."""
    root = Path(os.path.expanduser(intent_sim_dir)).resolve()
    if not (root / "llm" / "llm_client.py").exists():
        raise FileNotFoundError(f"intent-sim not found at {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    for site in sorted((root / ".venv" / "lib").glob("python3*/site-packages")):
        if str(site) not in sys.path:
            sys.path.append(str(site))
    _hide_old_brotli()
    return root


def _hide_old_brotli():
    """Ubuntu's python3-brotli predates the Decompressor.process(output_buffer_limit=)
    API that intent-sim's httpx uses, so every brotli-encoded OpenAI reply fails with
    APIConnectionError. intent-sim's own venv has no brotli; hide the system one so
    httpx stops advertising `br`, exactly as it behaves there."""
    try:
        import brotli
        brotli.Decompressor().process(b"", output_buffer_limit=1)
    except ImportError:
        return
    except TypeError:
        sys.modules["brotli"] = None  # later `import brotli` raises ImportError
    except Exception:  # new API, b"" just isn't a valid stream
        return


class CanvasFrame:
    def __init__(self, ox: float, oy: float):
        self.ox, self.oy = ox, oy

    def to_canvas(self, x, y):
        return x - self.ox, self.oy - y

    def to_world(self, cx, cy):
        return cx + self.ox, self.oy - cy


def obstacle_boxes(obs, recv_t, frame, max_age=3.0):
    """[x0, y0, x1, y1] boxes of the last /intent/obstacles message (lidar_obstacles), or []
    when there is none, it is stale (node stopped) or it is in another frame than the agents."""
    if obs is None or time.monotonic() - recv_t > max_age or obs.get("frame") != frame:
        return []
    return obs["boxes"]


def build_scene(agents: list[dict], frame: CanvasFrame, boxes=()):
    """intent-sim Scene from /mocap/agents entries (+ PhaseSpace-frame [x0, y0, x1, y1]
    obstacle boxes from lidar_obstacles). Returns (scene, warnings)."""
    from config import CANVAS_H, CANVAS_W
    from sim.entities import Human, Obstacle, Robot
    from sim.scene import Scene

    warnings = []
    robots = [a for a in agents if a["role"] == "robot"]
    humans = [a for a in agents if a["role"] == "human"]
    if len(robots) != 1:
        raise ValueError(f"need exactly one agent with role 'robot', have {len(robots)}")
    r = robots[0]
    if r["yaw_deg"] is None:
        raise ValueError(f"robot '{r['name']}' has no heading; capture its hat (Ctrl+P in the labeler)")
    rx, ry = frame.to_canvas(r["x"], r["y"])
    scene = Scene(robot=Robot(rx, ry, r["yaw_deg"] % 360))
    for h in humans:
        hx, hy = frame.to_canvas(h["x"], h["y"])
        scene.humans.append(Human(h["name"], hx, hy, (h["yaw_deg"] or 0.0) % 360))
    scene.orientation = all(h["yaw_deg"] is not None for h in humans)
    if not scene.orientation:
        warnings.append("some humans have no heading: facing withheld from the prompt")
    for x0, y0, x1, y1 in boxes:  # canvas is y-down: the box's top-left is (x0, y1)
        cx, cy = frame.to_canvas(x0, y1)
        scene.obstacles.append(Obstacle(round(cx, 2), round(cy, 2), round(x1 - x0, 2), round(y1 - y0, 2)))
    for a in agents:
        cx, cy = frame.to_canvas(a["x"], a["y"])
        if not (0 <= cx <= CANVAS_W and 0 <= cy <= CANVAS_H):
            warnings.append(f"{a['name']} at canvas ({cx:.1f}, {cy:.1f}) is outside the {CANVAS_W}x{CANVAS_H} m canvas")
    return scene, warnings


def goals_from_trace(trace: dict, frame: CanvasFrame, terminal_calls=("interact", "listen")):
    """Pose goals (PhaseSpace frame) from an intent-sim trace: the 2D robot's pose after
    every non-terminal call, as in G1_sim/g1_sim/goto_trace.py::load_goals."""
    goals, terminal = [], None
    for s in trace["steps"]:
        if s.get("result") is None:  # per-tick line; the call's result is on its last tick
            continue
        if s["function"] in terminal_calls:
            terminal = {"function": s["function"], "args": s["args"], "result": s["result"]}
            break
        x, y = frame.to_world(s["robot_x"], s["robot_y"])
        g = {"call": s["function"], "args": s["args"], "x": round(x, 3), "y": round(y, 3),
             "yaw_deg": round(s["robot_facing_deg"], 1)}
        if goals and math.hypot(g["x"] - goals[-1]["x"], g["y"] - goals[-1]["y"]) < 0.01 \
                and abs((g["yaw_deg"] - goals[-1]["yaw_deg"] + 180) % 360 - 180) < 1.0:
            continue
        goals.append(g)
    return goals, terminal
