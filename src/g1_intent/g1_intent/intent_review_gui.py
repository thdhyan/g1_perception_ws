"""intent-sim style review window: type a task, check the LLM plan, approve or stop.

Left: the live PhaseSpace scene drawn with intent-sim's own renderer (Hall zones,
personal space, facing), plus the LLM's 2D trace (orange), the numbered pose goals
with their headings, the goto controller's predicted path from the live robot pose
(dashed blue) and the robot's walked path (green).
Right: task box, planner status, the LLM's function calls (ok / error), per-goal
velocity preview (max |vx|, |vy|, |wz|, predicted time) and the executor state.

Keys:
  type            edit the task
  Enter           task typed -> send to the planner (/intent/task)
                  empty box + plan waiting -> arm; Enter again within 1.5 s -> APPROVE (/intent/approve)
  Ctrl+R          replan the last task
  Esc             E-STOP / reject (/intent/estop)
  Ctrl+Q          quit

Every plan's review session is recorded to <run_dir>/review.mp4 (10 fps).
"""

import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Empty, String

from g1_intent.goto_controller import predict
from g1_intent.scene_bridge import CanvasFrame, build_scene, import_intent_sim

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
PANEL_W, FPS, VIDEO_FPS = 460, 30, 10
ARM_S = 1.5  # double-Enter window for approving a plan
TRACE_C, GOAL_C, PRED_C, WALK_C = (230, 120, 20), (200, 60, 30), (40, 90, 220), (20, 150, 60)
TEXT_C, DIM_C, OK_C, BAD_C = (30, 30, 30), (110, 110, 110), (20, 140, 40), (200, 30, 30)
STATE_C = {"running": (20, 120, 220), "gesture": (20, 120, 220), "done": OK_C,
           "estopped": BAD_C, "aborted": BAD_C, "idle": DIM_C}


class ReviewNode(Node):
    def __init__(self):
        super().__init__("intent_review_gui")
        p = lambda n, v: self.declare_parameter(n, v).value  # noqa: E731
        import_intent_sim(p("intent_sim_dir", "~/Projects/thesis/intent-sim"))
        self.frame = CanvasFrame(p("canvas_origin_x", -5.0), p("canvas_origin_y", 3.5))
        self.record = p("record", True)
        self.lock = threading.Lock()
        self.agents, self.plan, self.exec_state = [], None, {}
        self.status = ["waiting for a task"]
        self.task_pub = self.create_publisher(String, "/intent/task", 10)
        self.approve_pub = self.create_publisher(String, "/intent/approve", 10)
        self.estop_pub = self.create_publisher(Empty, "/intent/estop", 10)
        self.create_subscription(String, "/mocap/agents", self.on_agents, 10)
        self.create_subscription(String, "/intent/plan", self.on_plan, LATCHED)
        self.create_subscription(String, "/intent/status", self.on_status, 10)
        self.create_subscription(String, "/intent/exec_state", self.on_exec, 10)

    def on_agents(self, msg):
        with self.lock:
            self.agents = json.loads(msg.data)["agents"]

    def on_plan(self, msg):
        plan = json.loads(msg.data)
        s = plan["start"]
        path, stats = predict(plan["goals"], s["x"], s["y"], s["yaw_deg"])
        plan["pred_path"], plan["pred_stats"] = path, stats
        with self.lock:
            self.plan = plan

    def on_status(self, msg):
        with self.lock:
            self.status = (self.status + [msg.data])[-4:]

    def on_exec(self, msg):
        with self.lock:
            self.exec_state = json.loads(msg.data)


class Recorder:
    def __init__(self, path, size):
        self.path = path
        self.proc = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{size[0]}x{size[1]}", "-r", str(VIDEO_FPS), "-i", "-",
             "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23", str(path)],
            stdin=subprocess.PIPE)

    def write(self, surface, pygame):
        try:
            self.proc.stdin.write(pygame.image.tostring(surface, "RGB"))
        except BrokenPipeError:
            pass

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


class Gui:
    def __init__(self, node):
        import pygame
        from config import CANVAS_H_PX, CANVAS_W_PX, PX_PER_METER
        from sim.render import BG_COLOR, draw_scene, dotted_polyline

        self.pg, self.node, self.ppm = pygame, node, PX_PER_METER
        self.draw_scene, self.dotted, self.bg = draw_scene, dotted_polyline, BG_COLOR
        self.cw, self.ch = int(CANVAS_W_PX), int(CANVAS_H_PX)
        pygame.init()
        pygame.key.start_text_input()
        self.screen = pygame.display.set_mode((self.cw + PANEL_W, self.ch))
        pygame.display.set_caption("g1_intent review")
        self.canvas = pygame.Surface((self.cw, self.ch))
        self.font = pygame.font.SysFont("dejavusansmono", 14)
        self.big = pygame.font.SysFont("dejavusans", 18, bold=True)
        self.task, self.last_task = "", ""
        self.walk, self.walk_plan = [], None
        self.recorder, self.rec_plan, self.last_frame_t = None, None, 0.0
        self.armed_t = 0.0  # first Enter on an empty box; the second within ARM_S approves

    # ── drawing ──────────────────────────────────────────────────────────
    def px(self, x, y):
        cx, cy = self.node.frame.to_canvas(x, y)
        return int(cx * self.ppm), int(cy * self.ppm)

    def draw_canvas(self, agents, plan, ex):
        pg = self.pg
        try:
            scene, _ = build_scene(agents, self.node.frame)
            exempt = set(plan["terminal"]["args"].get("target_names", [])) if plan and plan["terminal"] else None
            self.draw_scene(self.canvas, scene, show_orientation=True, hall_zones=True, exempt_names=exempt)
        except Exception as e:  # no robot labelled yet etc.: still show the humans as dots
            self.canvas.fill(self.bg)
            for a in agents:
                pg.draw.circle(self.canvas, (40, 110, 200), self.px(a["x"], a["y"]), 12)
            self.text(self.canvas, f"scene: {e}", (10, self.ch - 24), BAD_C)
        for a in agents:
            x, y = self.px(a["x"], a["y"])
            self.text(self.canvas, f"{a['name']} ({a['yaw_source']})", (x + 16, y - 22), TEXT_C)
        if plan:
            if len(plan["trace_path"]) > 1:
                pg.draw.lines(self.canvas, TRACE_C, False,
                              [(int(cx * self.ppm), int(cy * self.ppm)) for cx, cy in plan["trace_path"]], 3)
            pred = [self.px(x, y) for x, y in plan["pred_path"]]
            if len(pred) > 1:
                self.dotted(self.canvas, pred, PRED_C, dot=2, gap=6)
            for i, g in enumerate(plan["goals"]):
                x, y = self.px(g["x"], g["y"])
                yaw = math.radians(g["yaw_deg"])
                pg.draw.circle(self.canvas, GOAL_C, (x, y), 9, 2)
                pg.draw.line(self.canvas, GOAL_C, (x, y), (x + 24 * math.cos(yaw), y - 24 * math.sin(yaw)), 3)
                self.text(self.canvas, str(i + 1), (x - 4, y + 10), GOAL_C)
        if len(self.walk) > 1:
            pg.draw.lines(self.canvas, WALK_C, False, [self.px(x, y) for x, y in self.walk], 3)
        self.text(self.canvas, "orange: LLM trace   blue dashed: goto prediction   green: walked   "
                               "circles+ticks: pose goals", (10, 8), DIM_C)

    def text(self, surf, s, pos, color=TEXT_C, font=None):
        surf.blit((font or self.font).render(s, True, color), pos)

    def draw_panel(self, plan, ex, status):
        pg, x0 = self.pg, self.cw
        pg.draw.rect(self.screen, (252, 252, 250), (x0, 0, PANEL_W, self.ch))
        pg.draw.line(self.screen, (180, 180, 180), (x0, 0), (x0, self.ch), 2)
        y, x = 10, x0 + 12

        def put(s, c=TEXT_C, f=None, dy=18):
            nonlocal y
            self.text(self.screen, s[:58], (x, y), c, f)
            y += dy

        if time.time() - self.armed_t < ARM_S:
            put("Enter AGAIN to EXECUTE on the robot", (220, 30, 30), self.big, 22)
        else:
            put("TASK  (Enter: send | empty: Enter x2 = approve)", DIM_C)
        pg.draw.rect(self.screen, (255, 255, 255), (x - 4, y, PANEL_W - 20, 26))
        pg.draw.rect(self.screen, (120, 120, 120), (x - 4, y, PANEL_W - 20, 26), 1)
        cursor = "_" if int(time.time() * 2) % 2 else " "
        self.text(self.screen, (self.task + cursor)[-54:], (x, y + 5))
        y += 34
        for s in status:
            put(s, DIM_C)
        y += 6
        if plan:
            put(f"PLAN {plan['id']}", TEXT_C, self.big, 24)
            put(f"{plan['provider']}/{plan['model']}  path={plan.get('path_used')}", DIM_C)
            put(f"task: {plan['task']}", DIM_C)
            y += 4
            put("LLM calls (PhaseSpace frame)", TEXT_C, self.big, 22)
            for c in plan["calls"][:10]:
                args = ",".join(f"{k}={v}" for k, v in c.get("ps_args", c["args"]).items())
                put(f"{'ok ' if c['ok'] else 'ERR'} {c['function']}({args})", OK_C if c["ok"] else BAD_C)
                if c.get("error"):
                    put(f"    {c['error']}", BAD_C)
            y += 4
            put("Goals -> velocity preview (max |vx| |vy| |wz|, t)", TEXT_C, self.big, 22)
            cur = ex.get("goal") if ex.get("plan_id") == plan["id"] else None
            for i, (g, st) in enumerate(zip(plan["goals"], plan["pred_stats"])):
                mark = ">" if cur == i and ex.get("state") == "running" else " "
                d = st.get("duration_s")
                put(f"{mark}{i + 1} {g['call'][:7]:7s} ({g['x']:+.2f},{g['y']:+.2f},{g['yaw_deg']:+4.0f})"
                    f" {st['max_vx']:.2f} {st['max_vy']:.2f} {st['max_wz']:.2f} "
                    f"{'-' if d is None else f'{d:.0f}s'}", TEXT_C if st["status"] == "reached" else BAD_C)
            t = plan["terminal"]
            put(f"terminal: {t['function']}({t['args']})" if t else "terminal: none", TEXT_C)
        y = self.ch - 120
        state = ex.get("state", "no executor")
        put(f"EXECUTOR: {state.upper()}{'  (DRY RUN)' if ex.get('dry_run') else ''}",
            STATE_C.get(state, DIM_C), self.big, 24)
        put(ex.get("msg", ""), TEXT_C)
        if ex.get("pose"):
            put(f"pose {ex['pose']}  cmd {ex['cmd']}", DIM_C)
        put("Enter (empty box): APPROVE   Ctrl+R: replan", DIM_C)
        put("Esc: E-STOP / reject         Ctrl+Q: quit", BAD_C)
        if self.recorder:
            pg.draw.circle(self.screen, BAD_C, (x0 + PANEL_W - 20, 18), 7)

    # ── events ───────────────────────────────────────────────────────────
    def handle(self, ev, plan):
        pg, node = self.pg, self.node
        if ev.type == pg.QUIT:
            return False
        if ev.type == pg.TEXTINPUT:
            self.task += ev.text
        elif ev.type == pg.KEYDOWN:
            ctrl = ev.mod & pg.KMOD_CTRL
            if ev.key == pg.K_ESCAPE:
                self.armed_t = 0.0
                node.estop_pub.publish(Empty())
            elif ctrl and ev.key == pg.K_q:
                return False
            elif ctrl and ev.key == pg.K_r and self.last_task:
                node.task_pub.publish(String(data=self.last_task))
            elif ev.key == pg.K_BACKSPACE:
                self.task = self.task[:-1]
            elif ev.key in (pg.K_RETURN, pg.K_KP_ENTER):
                if self.task.strip():
                    self.last_task = self.task.strip()
                    node.task_pub.publish(String(data=self.last_task))
                    self.task = ""
                elif plan and time.time() - self.armed_t < ARM_S:
                    self.armed_t = 0.0
                    node.approve_pub.publish(String(data=plan["id"]))
                elif plan:
                    self.armed_t = time.time()
        return True

    def update_recording(self, plan):
        if not self.node.record or not plan or plan["id"] == self.rec_plan:
            return
        if self.recorder:
            self.recorder.close()
        self.rec_plan = plan["id"]
        self.recorder = Recorder(Path(plan["run_dir"]) / "review.mp4", self.screen.get_size())

    def run(self):
        pg, clock, running = self.pg, self.pg.time.Clock(), True
        while running and rclpy.ok():
            with self.node.lock:
                agents, plan = list(self.node.agents), self.node.plan
                ex, status = dict(self.node.exec_state), list(self.node.status)
            for ev in pg.event.get():
                running = self.handle(ev, plan) and running
            if plan and plan["id"] != self.walk_plan:
                self.walk, self.walk_plan = [], plan["id"]
            if ex.get("state") in ("running", "gesture") and ex.get("pose"):
                self.walk.append(tuple(ex["pose"][:2]))
            self.update_recording(plan)
            self.draw_canvas(agents, plan, ex)
            self.screen.blit(self.canvas, (0, 0))
            self.draw_panel(plan, ex, status)
            pg.display.flip()
            now = time.monotonic()
            if self.recorder and now - self.last_frame_t >= 1.0 / VIDEO_FPS:
                self.recorder.write(self.screen, pg)
                self.last_frame_t = now
            clock.tick(FPS)
        if self.recorder:
            self.recorder.close()
        pg.quit()


def spin_quietly(node):
    try:
        rclpy.spin(node)
    except ExternalShutdownException:  # Ctrl-C / launch shutdown
        pass


def main():
    rclpy.init()
    node = ReviewNode()
    threading.Thread(target=spin_quietly, args=(node,), daemon=True).start()
    try:
        Gui(node).run()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        os._exit(0)  # spin thread may still be blocked in rcl


if __name__ == "__main__":
    main()
