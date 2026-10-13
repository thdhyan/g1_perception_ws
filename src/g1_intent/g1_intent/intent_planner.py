"""intent-sim LLM planner on the live PhaseSpace scene.

On a task (std_msgs/String on /intent/task) it snapshots /mocap/agents, turns
it into an intent-sim Scene, and runs intent-sim's own static policy loop
(experiments.policy_runner.run_policy: system prompt with proxemics guidance,
LLMClient function calling, multi-round execution in the 2D sim). The resulting
trace becomes pose goals in the PhaseSpace frame, published latched on /intent/plan
for the review GUI and the executor. Nothing moves until a plan is approved.

Params: provider, model (any intent-sim provider: openai, openrouter, nim, gemini,
ollama, dl, spark01-03), base_url, prompt_style, prompt_variant, scene_format, temperature,
intent_sim_dir, runs_dir, canvas_origin_x/y.
"""

import json
import os
import threading
import time
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

from g1_intent.scene_bridge import CanvasFrame, build_scene, goals_from_trace, import_intent_sim

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class IntentPlanner(Node):
    def __init__(self):
        super().__init__("intent_planner")
        p = lambda n, v: self.declare_parameter(n, v).value  # noqa: E731
        self.intent_sim = import_intent_sim(p("intent_sim_dir", "~/Projects/thesis/intent-sim"))
        p("provider", "openai")
        p("model", "gpt-6.1-sol")
        p("base_url", "")  # empty: the provider's default endpoint (e.g. http://localhost:11434/v1 for a local ollama)
        p("prompt_style", "proxemics")
        p("prompt_variant", "a")
        p("scene_format", "json")
        p("temperature", 0.7)
        self.runs_dir = Path(os.path.expanduser(p("runs_dir", "~/Projects/thesis/g1_perception_ws/intent_runs")))
        self.frame = CanvasFrame(p("canvas_origin_x", -5.0), p("canvas_origin_y", 3.5))
        self.max_age = p("max_agents_age", 1.0)

        from experiments.policy_runner import run_policy  # noqa: F401  (fail at start, not on first task)

        self.agents, self.agents_t = None, 0.0
        self.busy = False
        self.plan_pub = self.create_publisher(String, "/intent/plan", LATCHED)
        self.status_pub = self.create_publisher(String, "/intent/status", 10)
        self.create_subscription(String, "/mocap/agents", self.on_agents, 10)
        self.create_subscription(String, "/intent/task", self.on_task, 10)
        self.get_logger().info(f"intent-sim at {self.intent_sim}; runs -> {self.runs_dir}; "
                               f"provider={self.param('provider')} model={self.param('model')}")

    def param(self, name):
        return self.get_parameter(name).value

    def status(self, text, level="info"):
        log = self.get_logger()  # rclpy forbids one call site (line) logging at several severities
        if level == "error":
            log.error(text)
        elif level == "warn":
            log.warn(text)
        else:
            log.info(text)
        self.status_pub.publish(String(data=text))

    def ps_args(self, step):
        """The call's args in the PhaseSpace frame (the LLM works on the intent-sim canvas).
        Only move_to_x_y has coordinates; rotate's facing is already PhaseSpace yaw."""
        a = step["args"]
        if step["function"] != "move_to_x_y" or not {"x", "y"} <= a.keys():
            return a
        x, y = self.frame.to_world(float(a["x"]), float(a["y"]))
        return {"x": round(x, 2), "y": round(y, 2)}

    def on_agents(self, msg):
        self.agents, self.agents_t = json.loads(msg.data), time.monotonic()

    def on_task(self, msg):
        task = msg.data.strip()
        if not task:
            return
        if self.busy:
            self.status("planner busy: task ignored", "warn")
            return
        if self.agents is None or time.monotonic() - self.agents_t > self.max_age:
            self.status("no fresh /mocap/agents: is the labeler running and the robot labelled?", "error")
            return
        self.busy = True
        threading.Thread(target=self.plan, args=(task, self.agents), daemon=True).start()

    def plan(self, task, snapshot):
        try:
            from experiments.policy_runner import run_policy
            from llm.llm_client import LLMClient

            scene, warnings = build_scene(snapshot["agents"], self.frame)
            for w in warnings:
                self.status(f"warning: {w}", "warn")
            provider, model = self.param("provider"), self.param("model")
            self.status(f"planning with {provider}/{model}: {task!r}")
            t0 = time.time()
            client = LLMClient(provider=provider, model=model or None, base_url=self.param("base_url") or None)
            run_dir = run_policy(scene, task, client, experiment="g1_live",
                                 temperature=self.param("temperature"),
                                 prompt_variant=self.param("prompt_variant"),
                                 scene_format=self.param("scene_format"),
                                 prompt_style=self.param("prompt_style"),
                                 runs_dir=self.runs_dir)
            trace_path = run_dir / "trace.json"
            trace = json.loads(trace_path.read_text())
            goals, terminal = goals_from_trace(trace, self.frame)
            if not goals and terminal is None:
                self.status(f"LLM made no usable calls (path={trace.get('path_used')}, "
                            f"content={str(trace.get('raw_content'))[:80]!r}): replan or switch model", "warn")
            robot = next(a for a in snapshot["agents"] if a["role"] == "robot")
            trace["g1_intent"] = {"mocap_snapshot": snapshot, "frame": snapshot["frame"],
                                  "canvas_origin": [self.frame.ox, self.frame.oy],
                                  "goals": goals, "terminal": terminal, "llm_seconds": round(time.time() - t0, 1)}
            trace_path.write_text(json.dumps(trace, indent=2))
            plan = {
                "id": run_dir.name, "run_dir": str(run_dir), "task": task, "provider": provider,
                "model": client.model, "status": "ready", "frame": snapshot["frame"],
                "canvas_origin": [self.frame.ox, self.frame.oy],
                "start": {"name": robot["name"], "x": robot["x"], "y": robot["y"], "yaw_deg": robot["yaw_deg"]},
                "humans": [a for a in snapshot["agents"] if a["role"] == "human"],
                "calls": [{"function": s["function"], "args": s["args"], "ps_args": self.ps_args(s),
                           "ok": s["result"].get("ok"), "error": s["result"].get("error")}
                          for s in trace["steps"] if s.get("result") is not None],
                "trace_path": [[s["robot_x"], s["robot_y"]] for s in trace["steps"]],
                "goals": goals, "terminal": terminal, "path_used": trace.get("path_used"),
                "raw_content": trace.get("raw_content"),
            }
            self.plan_pub.publish(String(data=json.dumps(plan)))
            self.status(f"plan {plan['id']}: {len(goals)} goals, terminal "
                        f"{terminal['function'] if terminal else None} ({time.time() - t0:.1f} s)")
        except Exception as e:  # LLM/network/scene errors go to the GUI, the node keeps running
            self.status(f"planning failed: {type(e).__name__}: {e}", "error")
        finally:
            self.busy = False


def main():
    rclpy.init()
    node = IntentPlanner()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
