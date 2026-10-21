"""Browser console for the live G1 intent loop: one page instead of the Tk labeler
and the pygame review window.

Serves intent-sim's own run browser (browser/index.html + sim.js: scene canvas,
System Prompt / Steps / SNGNN / Nav / VO tabs) over this workspace's runs_dir, with
three extra tabs (web/live_tabs.html + live.js):

  Live     task, provider / model / prompt variant / style / anchor (variant ra),
           prompt preview, plan + LLM calls, low-level planner, Approve, E-STOP
  Labeler  name / role / heading / capture / save for every PhaseSpace group; static
           humans placed with one tracker (add / delete)
           (g1_mocap_labeler's Labeler node, run headless in this process)
  Robot    robot_bridge buttons (walk mode, stop, damp, wave, shake hand, FSM)

A synthetic run "live" holds the current /mocap/agents scene, so every intent-sim
tab works on the live scene too.

Low-level planner: "direct" runs the plan's goals as published. A*, GD, Adam and
AdamW paths are computed in the page (sim.js navAstar / navWalk on the SNGNN + GMM +
obstacle cost map) from the robot to each goal; they come back as waypoint goals
("waypoint": True, see goto_controller.WAYPOINT_TOL) inserted before each goal, and
the extended plan is republished as <plan id>~<planner> before it is approved.

In:  /mocap/agents, /intent/plan, /intent/exec_state, /intent/status, /intent/obstacles (LiDAR boxes)
Out: /intent/task (JSON), /intent/plan, /intent/approve, /intent/estop, /mocap/agents (labeler)
"""

import collections
import json
import math
import os
import sys
import threading
import time
import webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Empty, String

from g1_intent.goto_executor import bridge_call
from g1_intent.scene_bridge import CanvasFrame, build_scene, import_intent_sim, obstacle_boxes

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
ROBOT_CMDS = {
    "ensure_walk_mode": ({"cmd": "ensure_walk_mode"}, 15.0),
    "get_fsm": ({"cmd": "get_fsm"}, 5.0),
    "stop": ({"cmd": "stop"}, 5.0),
    "damp": ({"cmd": "damp"}, 5.0),
    "wave": ({"cmd": "wave", "wave_type": "face", "hold_seconds": 3.0}, 20.0),
    "shake_hand": ({"cmd": "shake_hand", "hold_seconds": 3.0}, 20.0),
}
WAYPOINT_SPACING = 0.5  # m between inserted waypoint goals


class WebConsole(Node):
    def __init__(self):
        super().__init__("intent_web_console")
        p = lambda n, v: self.declare_parameter(n, v).value  # noqa: E731
        self.intent_sim = import_intent_sim(p("intent_sim_dir", "~/Projects/thesis/intent-sim"))
        self.runs_dir = Path(os.path.expanduser(p("runs_dir", "~/Projects/thesis/g1_perception_ws/intent_runs")))
        self.frame = CanvasFrame(p("canvas_origin_x", -5.0), p("canvas_origin_y", 3.5))
        self.socket = p("bridge_socket", "/tmp/g1_robot_bridge.sock")
        self.dry_run = p("dry_run", False)
        self.port = p("port", 8090)
        self.open_browser = p("open_browser", True)
        # defaults shown in the page; the planner's own params apply when a field is left empty
        self.defaults = {k: p(k, v) for k, v in (("provider", "openai"), ("model", "gpt-6.1-sol"),
                                                  ("prompt_variant", "a"), ("prompt_style", "proxemics"),
                                                  ("scene_format", "json"), ("temperature", 0.7))}

        self.lock = threading.Lock()
        self.agents, self.exec_state = None, None
        self.obstacles, self.obstacles_t = None, 0.0
        self.plan, self.plans = None, {}  # latest plan; planner plans by id (the approve base)
        self.log = collections.deque(maxlen=60)
        self.labeler = None  # set by main()

        self.task_pub = self.create_publisher(String, "/intent/task", 10)
        self.plan_pub = self.create_publisher(String, "/intent/plan", LATCHED)
        self.approve_pub = self.create_publisher(String, "/intent/approve", 10)
        self.estop_pub = self.create_publisher(Empty, "/intent/estop", 10)
        self.create_subscription(String, "/mocap/agents", self.on_agents, 10)
        self.create_subscription(String, "/intent/plan", self.on_plan, LATCHED)
        self.create_subscription(String, "/intent/exec_state", self.on_exec, 10)
        self.create_subscription(String, "/intent/status", self.on_status, 10)
        self.create_subscription(String, "/intent/obstacles", self.on_obstacles, LATCHED)

    # ── ROS in ──────────────────────────────────────────────────────────
    def on_agents(self, msg):
        with self.lock:
            self.agents = json.loads(msg.data)

    def on_obstacles(self, msg):
        with self.lock:
            self.obstacles, self.obstacles_t = json.loads(msg.data), time.monotonic()

    def on_plan(self, msg):
        plan = json.loads(msg.data)
        with self.lock:
            self.plan = plan
            if "~" not in plan["id"]:
                self.plans[plan["id"]] = plan

    def on_exec(self, msg):
        with self.lock:
            self.exec_state = json.loads(msg.data)

    def on_status(self, msg):
        self.note(msg.data)

    def note(self, text):
        with self.lock:
            self.log.append([round(time.time(), 1), text])

    # ── page API ────────────────────────────────────────────────────────
    def live_scene(self):
        """(intent-sim scene dict in canvas coords, warnings) from the latest /mocap/agents."""
        with self.lock:
            agents = self.agents
            obs, obs_t = self.obstacles, self.obstacles_t
        if agents is None:
            raise ValueError("no /mocap/agents yet")
        scene, warnings = build_scene(agents["agents"], self.frame, obstacle_boxes(obs, obs_t, agents.get("frame")))
        return scene, warnings

    def live_trace(self):
        try:
            scene, warnings = self.live_scene()
            sd, err = scene.to_dict(include_orientation=True), None
        except ValueError as e:  # no robot labelled yet: empty canvas, the overlay still shows the groups
            sd, warnings, err = {"robot": {"x": -1, "y": -1, "facing_deg": 0}, "humans": [], "objects": [],
                                 "obstacles": []}, [], str(e)
        return {"experiment": "g1_live", "task": "(live scene)", "model": "-", "provider": "-",
                "prompt_variant": self.defaults["prompt_variant"], "prompt_style": self.defaults["prompt_style"],
                "scene_format": "json", "orientation": True, "fold": 0, "path_used": "-", "scene": sd,
                "scene_params": {}, "steps": [], "_live": {"warnings": warnings, "error": err}}

    def state(self):
        with self.lock:
            out = {"agents": self.agents, "plan": self.plan, "exec": self.exec_state, "log": list(self.log),
                   "dry_run": self.dry_run, "canvas_origin": [self.frame.ox, self.frame.oy]}
            if self.agents is not None:  # LiDAR boxes in canvas coords, refreshed into the live scene by live.js
                boxes = obstacle_boxes(self.obstacles, self.obstacles_t, self.agents.get("frame"))
                out["obstacles"] = [{"x": round(x0 - self.frame.ox, 2), "y": round(self.frame.oy - y1, 2),
                                     "w": round(x1 - x0, 2), "h": round(y1 - y0, 2)} for x0, y0, x1, y1 in boxes]
        lab = self.labeler
        if lab is not None:
            with lab.lock:
                hats = {h: dict(st) for h, st in lab.state.items()}
                statics = {n: dict(v) for n, v in lab.statics.items()}
            out["labeler"] = {"hats": hats, "statics": statics, "status": lab.status, "frame": lab.frame,
                              "placing": lab.static_capture is not None}
        return out

    def options(self):
        from sim import prompt_creator
        from sim.scene_to_prompt import PROMPT_STYLES, TEMPLATE_VARIANTS
        runs = []
        for t in self.runs_dir.glob("*/trace.json"):
            try:
                d = json.loads(t.read_text())
                runs.append({"provider": d.get("provider"), "model": d.get("model")})
            except Exception:
                continue
        o = prompt_creator.options(runs)
        for p in o["providers"].values():  # the launch default first in its provider's list
            if p is o["providers"].get(self.defaults["provider"]) and self.defaults["model"] not in p["models"]:
                p["models"].insert(0, self.defaults["model"])
        return {"providers": o["providers"], "variants": sorted(TEMPLATE_VARIANTS),
                "styles": sorted(PROMPT_STYLES), "defaults": self.defaults, "dry_run": self.dry_run}

    def prompt_preview(self, body):
        from sim.scene_to_prompt import render_system_prompt
        scene, warnings = self.live_scene()
        variant, anchor = body.get("prompt_variant") or "a", body.get("anchor") or None
        if variant == "ra" and anchor not in {h.name for h in scene.humans}:
            raise ValueError("variant ra: pick an anchor (a labelled human)")
        if variant == "r" and scene.obstacles:  # intent-sim's relative encoding has no obstacles
            scene.obstacles, warnings = [], warnings + ["variant r: LiDAR obstacles left out of the prompt"]
        text = render_system_prompt(scene, body.get("task") or "(task)", variant=variant,
                                    scene_format=body.get("scene_format") or "json",
                                    prompt_style=body.get("prompt_style") or "base",
                                    anchor=anchor if variant == "ra" else None)
        return {"system_prompt": text, "warnings": warnings}

    def send_task(self, body):
        if not str(body.get("task", "")).strip():
            raise ValueError("empty task")
        self.task_pub.publish(String(data=json.dumps(body)))
        self.note(f"task sent: {body.get('task')!r} ({body.get('provider')}/{body.get('model')}, "
                  f"variant {body.get('prompt_variant')})")
        return {"ok": True}

    def approve(self, body):
        """body: plan_id, planner (direct|astar|gd|adam|adamw), waypoints: per goal, a list of
        [canvas_x, canvas_y] path points from the previous goal (start: the robot) to it."""
        base_id = str(body.get("plan_id", "")).split("~")[0]
        with self.lock:
            base = self.plans.get(base_id)
            exec_id = (self.exec_state or {}).get("plan_id")
            exec_st = (self.exec_state or {}).get("state")
        if base is None:
            raise ValueError(f"unknown plan {base_id!r}")
        if exec_st in ("running", "gesture"):
            raise ValueError("executor is busy: E-STOP or wait")
        planner = body.get("planner") or "direct"
        plan = base
        if planner != "direct":
            wps = body.get("waypoints") or []
            if len(wps) != len(base["goals"]):
                raise ValueError(f"{len(wps)} waypoint lists for {len(base['goals'])} goals")
            goals = []
            for g, pts in zip(base["goals"], wps):
                goals += self.waypoint_goals(pts, planner)
                goals.append(g)
            plan = dict(base, id=f"{base_id}~{planner}", goals=goals, low_level=planner)
        if exec_id != plan["id"]:
            self.plan_pub.publish(String(data=json.dumps(plan)))
        threading.Thread(target=self.approve_when_loaded, args=(plan["id"],), daemon=True).start()
        n_wp = sum(1 for g in plan["goals"] if g.get("waypoint"))
        self.note(f"approve {plan['id']}: {len(plan['goals'])} goals ({n_wp} {planner} waypoints)")
        return {"ok": True, "plan_id": plan["id"], "n_goals": len(plan["goals"]), "n_waypoints": n_wp}

    def waypoint_goals(self, pts, planner):
        """Waypoint goals (PhaseSpace) every WAYPOINT_SPACING m along a canvas path; the
        path's last point is the goal itself, so it is left out."""
        out, last, acc = [], None, 0.0
        world = [self.frame.to_world(float(x), float(y)) for x, y in pts]
        for k in range(1, len(world) - 1):
            (ax, ay), (bx, by) = world[k - 1], world[k]
            acc += math.hypot(bx - ax, by - ay)
            if acc < WAYPOINT_SPACING:
                continue
            gx, gy = world[-1]
            if math.hypot(gx - bx, gy - by) < WAYPOINT_SPACING:
                break
            px, py = last if last else world[0]
            out.append({"call": "waypoint", "args": {"planner": planner}, "x": round(bx, 3), "y": round(by, 3),
                        "yaw_deg": round(math.degrees(math.atan2(by - py, bx - px)), 1), "waypoint": True})
            last, acc = (bx, by), 0.0
        return out

    def approve_when_loaded(self, plan_id, timeout=3.0):
        """/intent/plan and /intent/approve are separate topics: approve only once the
        executor reports the plan, or it would approve (or reject) the previous one."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            with self.lock:
                if (self.exec_state or {}).get("plan_id") == plan_id:
                    break
            time.sleep(0.05)
        else:
            self.note(f"executor did not load {plan_id} within {timeout:.0f} s: approve not sent")
            return
        self.approve_pub.publish(String(data=plan_id))

    def estop(self):
        self.estop_pub.publish(Empty())
        self.note("E-STOP")
        return {"ok": True}

    def robot(self, body):
        cmd = body.get("cmd")
        if cmd not in ROBOT_CMDS:
            raise ValueError(f"unknown robot command {cmd!r}")
        with self.lock:
            busy = (self.exec_state or {}).get("state") in ("running", "gesture")
        if busy and cmd != "stop":
            raise ValueError("executor is running a plan: only Stop / E-STOP")
        if self.dry_run:
            self.note(f"robot {cmd}: dry run, not sent")
            return {"ok": False, "error": "dry_run: not sent to robot_bridge"}
        req, timeout = ROBOT_CMDS[cmd]
        try:
            r = bridge_call(self.socket, req, timeout=timeout)
        except Exception as e:
            r = {"ok": False, "error": f"robot_bridge {self.socket}: {e}"}
        self.note(f"robot {cmd}: {r}")
        return r

    def label(self, body):
        lab, hat, action = self.labeler, body.get("hat", ""), body.get("action")
        if action == "apply":
            lab.apply(hat, str(body.get("name", "")), str(body.get("heading", "")))
        elif action == "toggle_role":
            lab.toggle_role(hat)
        elif action == "capture":
            lab.start_capture(hat, str(body.get("heading", "")))
        elif action == "add_static":
            lab.add_static(hat, str(body.get("name", "")), str(body.get("heading", "")))
        elif action == "delete_static":
            lab.delete_static(str(body.get("name", "")))
        elif action == "save":
            lab.save()
        else:
            raise ValueError(f"unknown labeler action {action!r}")
        return {"ok": True, "status": lab.status}


def make_handler(console: WebConsole):
    """intent-sim's browser/server.py Handler (runs, traces, system prompts, SNGNN,
    static sim.js / style.css) pointed at runs_dir, plus the live routes."""
    browser_dir = str(console.intent_sim / "browser")  # server.py imports its siblings (prompt_creator)
    if browser_dir not in sys.path:
        sys.path.insert(0, browser_dir)
    import server as srv
    srv.RUNS_DIR = console.runs_dir
    web = Path(get_package_share_directory("g1_intent")) / "web"

    def page():
        html = (srv.BROWSER_DIR / "index.html").read_text()
        tabs = (web / "live_tabs.html").read_text()
        buttons, panels = tabs.split("<!-- panels -->")
        edits = [
            ("<title>Intent-Sim: Run Browser</title>", "<title>G1 Live Console</title>"),
            ('<link rel="stylesheet" href="style.css">',
             '<link rel="stylesheet" href="style.css">\n<link rel="stylesheet" href="live.css">'),
            ('<h1>Intent-Sim <span class="dim">Run Browser</span></h1>',
             '<h1>G1 <span class="dim">Live Console</span></h1><span id="liveBadge"></span>'),
            ('<button class="tab active" data-tab="promptTab">', '<button class="tab" data-tab="promptTab">'),
            ('<div id="tabBar">', '<div id="tabBar">\n' + buttons),
            ('<div id="promptTab" class="tabpanel active"></div>', '<div id="promptTab" class="tabpanel"></div>'),
            ('<div id="tabContent">', '<div id="tabContent">\n' + panels),
            ('<script src="sim.js"></script>', '<script src="sim.js"></script>\n<script src="live.js"></script>'),
        ]
        for old, new in edits:
            if old not in html:
                raise RuntimeError(f"intent-sim index.html changed, not found: {old}")
            html = html.replace(old, new, 1)
        return html.encode()

    class Handler(srv.Handler):
        def do_GET(self):
            path = self.path.split("?")[0]
            try:
                if path in ("/", "/index.html", "/live.html"):
                    self._send(page(), "text/html")
                elif path in ("/live.js", "/live.css"):
                    self._serve_file(web / path[1:])
                elif path == "/api/live":
                    self._json_response(console.state())
                elif path == "/api/options":
                    self._json_response(console.options())
                elif path == "/api/run/live":
                    self._json_response(console.live_trace())
                elif path == "/api/runs":
                    runs = sorted(self._list_runs(), key=lambda r: r["timestamp"], reverse=True)
                    n_humans = len(console.live_trace()["scene"]["humans"])
                    self._json_response([{"run_id": "live", "experiment": "g1_live", "dataset": "LIVE",
                                          "model": "", "provider": "", "variant": "a", "n_steps": 0,
                                          "scene_params": {"n_humans": n_humans, "n_moving": 0, "n_groups": 0,
                                                           "identifier": "mocap scene"}}] + runs)
                else:
                    super().do_GET()
            except Exception as e:
                self._json_response({"ok": False, "error": f"{type(e).__name__}: {e}"})

        def do_POST(self):
            routes = {"/api/task": console.send_task, "/api/prompt_preview": console.prompt_preview,
                      "/api/approve": console.approve, "/api/estop": lambda b: console.estop(),
                      "/api/robot": console.robot, "/api/labeler": console.label}
            if self.path not in routes:
                return super().do_POST()
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                self._json_response(routes[self.path](body))
            except Exception as e:  # shown in the page
                self._json_response({"ok": False, "error": f"{type(e).__name__}: {e}"})

        def _send(self, body, ct):
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):  # the page polls /api/live at 5 Hz
            pass

    return Handler


def main():
    rclpy.init()
    console = WebConsole()
    from g1_mocap_labeler.labeler_gui import Labeler  # after import_intent_sim: same process, no Tk window
    console.labeler = Labeler()
    executor = MultiThreadedExecutor()
    executor.add_node(console)
    executor.add_node(console.labeler)
    server = ThreadingHTTPServer(("0.0.0.0", console.port), make_handler(console))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://localhost:{console.port}/"
    console.get_logger().info(f"G1 live console: {url} (runs {console.runs_dir}, "
                              f"{'DRY RUN' if console.dry_run else 'robot_bridge ' + console.socket})")
    if console.open_browser:
        webbrowser.open(url)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        server.shutdown()
        console.labeler.destroy_node()
        console.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
