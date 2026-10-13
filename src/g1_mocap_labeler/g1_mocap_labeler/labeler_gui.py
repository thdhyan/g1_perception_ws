"""Keyboard GUI: name each PhaseSpace LED group, give it a role and a heading.

Reads the groups from phasespace_ros2 (/phasespace/groups) and their TFs
(phasespace -> <hat>, phasespace -> <hat>/led_<id>). For every labelled group:

  TF            phasespace -> <name>    position + yaw (rotation about z only)
  /mocap/agents std_msgs/String JSON, 20 Hz:
                {"frame", "stamp", "agents": [{"name", "group", "role", "x", "y", "z",
                 "yaw_deg", "yaw_source", "age"}]}

Yaw, in order of preference:
  fit      LED layout captured in this GUI (Ctrl+P), fitted every tick (2D Kabsch, >= 3 LEDs)
  tracker  the hat TF's own rotation, when phasespace_ros2 has a --capture reference
  static   the heading typed in the GUI (fine for people standing still)
  none     no heading (yaw_deg is null)

Keys:  Up/Down select group | type Name / Heading, Enter applies | Ctrl+R toggle role
       Ctrl+P capture orientation (hat still, facing the typed heading) | Ctrl+S save
       Ctrl+Left/Right heading -/+ 5 deg (Ctrl+Shift: 1 deg); or drag the dial / slider

The 2D preview (top-down, +x right, +y up) shows every group live: LEDs, position,
measured heading (solid arrow) and, for the selected group, the heading you are about
to apply (dashed orange). Enter on a captured (fit) group re-orients its layout so its
current facing becomes the dial heading, no re-capture needed.
Labels (and captured layouts) are saved to `labels_file` and reloaded at start.
"""

import json
import math
import os
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformBroadcaster, TransformListener

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
ROLES = ("human", "robot")
CAPTURE_SAMPLES = 20  # ticks averaged for a capture (1 s at 20 Hz)


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def fit_yaw(ref, seen):
    """2D Procrustes: yaw and origin that map hat-local `ref` points onto world `seen` points (dicts id -> xyz)."""
    ids = [i for i in ref if i in seen]
    if len(ids) < 3:
        return None
    r = np.array([ref[i][:2] for i in ids])
    c = np.array([seen[i][:2] for i in ids])
    rc, cc = r - r.mean(0), c - c.mean(0)
    yaw = math.atan2(np.sum(rc[:, 0] * cc[:, 1] - rc[:, 1] * cc[:, 0]), np.sum(rc * cc))
    rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    origin = (c - r @ rot.T).mean(0)
    z = float(np.mean([seen[i][2] - ref[i][2] for i in ids]))
    return yaw, float(origin[0]), float(origin[1]), z


class Labeler(Node):
    def __init__(self):
        super().__init__("mocap_labeler")
        self.frame = self.declare_parameter("world_frame", "phasespace").value
        self.labels_file = os.path.expanduser(
            self.declare_parameter("labels_file", "~/.ros/g1_mocap_labels.yaml").value)
        self.max_age = Duration(seconds=self.declare_parameter("max_age", 0.5).value)
        rate = self.declare_parameter("rate", 20.0).value

        self.lock = threading.Lock()
        self.groups = {}  # hat -> {"leds": [...], "oriented": bool}
        self.labels = self.load()  # hat -> {"name", "role", "heading_deg", "ref": {led: [x,y,z]}}
        self.state = {}  # hat -> live dict for the GUI
        self.capture = None  # (hat, heading_deg, [samples])
        self.status = f"labels: {self.labels_file}"

        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.tf_pub = TransformBroadcaster(self)
        self.agents_pub = self.create_publisher(String, "/mocap/agents", 10)
        self.create_subscription(String, "/phasespace/groups", self.on_groups, LATCHED)
        self.create_timer(1.0 / rate, self.tick)

    # ── persistence ──────────────────────────────────────────────────────
    def load(self):
        if not os.path.exists(self.labels_file):
            return {}
        with open(self.labels_file) as f:
            data = yaml.safe_load(f) or {}
        labels = data.get("groups", {})
        for lab in labels.values():
            lab["ref"] = {int(k): v for k, v in (lab.get("ref") or {}).items()}
        return labels

    def save(self):
        os.makedirs(os.path.dirname(self.labels_file), exist_ok=True)
        with self.lock:
            data = {"frame": self.frame, "groups": self.labels}
        text = yaml.safe_dump(data, sort_keys=True)  # dump before opening: a failed dump must not truncate the file
        with open(self.labels_file, "w") as f:
            f.write(text)
        self.status = f"saved {self.labels_file}"

    # ── GUI actions ──────────────────────────────────────────────────────
    def apply(self, hat, name, heading):
        name = name.strip()
        with self.lock:
            if name and (name in self.groups or name == self.frame):
                self.status = f"'{name}' is a PhaseSpace frame name; pick another"
                return
            taken = {lab["name"] for h, lab in self.labels.items() if h != hat}
            if name and name in taken:
                self.status = f"'{name}' already used by another group"
                return
            lab = self.labels.setdefault(hat, {"name": "", "role": "human", "heading_deg": None, "ref": {}})
            lab["name"] = name
            new = float(heading) if heading.strip() else None
            st = self.state.get(hat, {})
            if new is not None and lab.get("ref") and st.get("src") == "fit":
                # captured hat: "it is facing `new` right now" -> rotate the layout so the fit says so
                d = math.radians(new) - st["yaw"]
                c, s = math.cos(-d), math.sin(-d)
                lab["ref"] = {i: [round(c * x - s * y, 4), round(s * x + c * y, 4), z]
                              for i, (x, y, z) in lab["ref"].items()}
            lab["heading_deg"] = new
        self.status = f"{hat} -> '{name}'" + ("" if new is None else f", facing {new:.0f} deg")
        self.save()

    def toggle_role(self, hat):
        with self.lock:
            lab = self.labels.setdefault(hat, {"name": "", "role": "human", "heading_deg": None, "ref": {}})
            lab["role"] = ROLES[(ROLES.index(lab["role"]) + 1) % len(ROLES)]
            if lab["role"] == "robot":  # one robot only
                for h, other in self.labels.items():
                    if h != hat and other["role"] == "robot":
                        other["role"] = "human"
            self.status = f"{hat} role -> {lab['role']}"
        self.save()

    def start_capture(self, hat, heading):
        try:
            heading = float(heading)
        except ValueError:
            self.status = "type the heading the hat is facing now (deg, CCW from +x), then Ctrl+P"
            return
        self.capture = (hat, heading, [])
        self.status = f"capturing {hat} facing {heading:.0f} deg: hold still..."

    # ── ROS ──────────────────────────────────────────────────────────────
    def on_groups(self, msg):
        d = json.loads(msg.data)
        with self.lock:
            self.frame = d.get("frame", self.frame)
            self.groups = d["groups"]
        self.get_logger().info(f"PhaseSpace groups: {sorted(self.groups)}")

    def lookup(self, child, now):
        try:
            tf = self.tf.lookup_transform(self.frame, child, Time())
        except Exception:
            return None
        if now - Time.from_msg(tf.header.stamp) > self.max_age:
            return None
        return tf

    def tick(self):
        now = self.get_clock().now()
        with self.lock:
            groups = dict(self.groups)
            labels = {h: dict(v) for h, v in self.labels.items()}
        state, agents, tfs = {}, [], []
        for hat, g in sorted(groups.items()):
            leds = {}
            for led in g["leds"]:
                tf = self.lookup(f"{hat}/led_{led}", now)
                if tf is not None:
                    p = tf.transform.translation
                    leds[int(led)] = (p.x, p.y, p.z)
            hat_tf = self.lookup(hat, now)
            lab = labels.get(hat, {})
            st = {"leds": f"{len(leds)}/{len(g['leds'])}", "name": lab.get("name", ""),
                  "role": lab.get("role", "human"), "x": None, "y": None, "yaw": None, "src": "-",
                  "heading": lab.get("heading_deg"), "led_xy": [p[:2] for p in leds.values()]}
            if self.capture and self.capture[0] == hat:
                self.capture_tick(hat, leds)
            if hat_tf is not None or leds:
                x, y, z, yaw, src = self.pose(lab, g, hat_tf, leds)
                st.update(x=x, y=y, yaw=yaw, src=src)
                if lab.get("name"):
                    agents.append({"name": lab["name"], "group": hat, "role": st["role"],
                                   "x": round(x, 3), "y": round(y, 3), "z": round(z, 3),
                                   "yaw_deg": None if yaw is None else round(math.degrees(yaw), 1),
                                   "yaw_source": src})
                    tfs.append(self.make_tf(now, lab["name"], x, y, z, yaw or 0.0))
            state[hat] = st
        if tfs:
            self.tf_pub.sendTransform(tfs)
        self.agents_pub.publish(String(data=json.dumps(
            {"frame": self.frame, "stamp": now.nanoseconds * 1e-9, "agents": agents})))
        with self.lock:
            self.state = state

    def pose(self, lab, g, hat_tf, leds):
        ref = lab.get("ref") or {}
        fit = fit_yaw(ref, leds) if ref else None
        if fit is not None:
            yaw, x, y, z = fit
            return x, y, z, yaw, "fit"
        if hat_tf is not None:
            p = hat_tf.transform.translation
            x, y, z = p.x, p.y, p.z
        else:
            x, y, z = (float(v) for v in np.mean(list(leds.values()), axis=0))
        if hat_tf is not None and g.get("oriented"):
            return x, y, z, yaw_of(hat_tf.transform.rotation), "tracker"
        if lab.get("heading_deg") is not None:
            return x, y, z, math.radians(lab["heading_deg"]), "static"
        return x, y, z, None, "none"

    def capture_tick(self, hat, leds):
        _, heading, samples = self.capture
        samples.append(leds)
        if len(samples) < CAPTURE_SAMPLES:
            return
        self.capture = None
        common = set.intersection(*(set(s) for s in samples))
        if len(common) < 3:
            self.status = f"capture {hat} failed: only {len(common)} LEDs visible the whole time (need 3)"
            return
        pts = {i: np.mean([s[i] for s in samples], axis=0) for i in common}
        center = np.mean(list(pts.values()), axis=0)
        c, s = math.cos(-math.radians(heading)), math.sin(-math.radians(heading))
        ref = {}
        for i, p in pts.items():
            dx, dy, dz = p - center
            ref[int(i)] = [round(float(c * dx - s * dy), 4), round(float(s * dx + c * dy), 4), round(float(dz), 4)]
        with self.lock:
            lab = self.labels.setdefault(hat, {"name": "", "role": "human", "heading_deg": None, "ref": {}})
            lab["ref"] = ref
            lab["heading_deg"] = heading
        self.save()
        self.status = f"captured {hat}: {len(ref)} LEDs, facing {heading:.0f} deg"

    def make_tf(self, now, child, x, y, z, yaw):
        t = TransformStamped()
        t.header.stamp = now.to_msg()
        t.header.frame_id = self.frame
        t.child_frame_id = child
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = x, y, z
        t.transform.rotation.z, t.transform.rotation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        return t


class Gui:
    COLS = ("group", "leds", "name", "role", "x", "y", "yaw", "src")
    PREVIEW, DIAL = 420, 150
    HUMAN_C, ROBOT_C, SEL_C, PEND_C = "#2a6ec8", "#c8322a", "#f0b400", "#e07814"

    def __init__(self, node: Labeler):
        self.node = node
        self.root = tk.Tk()
        self.root.title("PhaseSpace labeler")
        self.root.geometry("1320x650")
        self.syncing = False
        right = tk.Frame(self.root)  # packed first so the table can't squeeze the preview
        right.pack(side="right", fill="y", padx=8, pady=6)
        left = tk.Frame(self.root)
        left.pack(side="left", fill="both", expand=True)

        self.tree = ttk.Treeview(left, columns=self.COLS, show="headings", height=12, selectmode="browse")
        for c, w in zip(self.COLS, (130, 60, 130, 70, 70, 70, 70, 70)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=8, pady=6)

        form = tk.Frame(left)
        form.pack(fill="x", padx=8)
        tk.Label(form, text="Name").grid(row=0, column=0)
        self.name = tk.Entry(form, width=20)
        self.name.grid(row=0, column=1, padx=4)
        tk.Label(form, text="Heading deg (CCW from +x)").grid(row=0, column=2)
        self.heading = tk.Entry(form, width=8)
        self.heading.grid(row=0, column=3, padx=4)
        self.heading.bind("<KeyRelease>", lambda e: self.sync_heading(self.heading_value(), "entry"))
        self.role = tk.Label(form, text="role: -", width=14)
        self.role.grid(row=0, column=4)
        buttons = tk.Frame(left)
        buttons.pack(fill="x", padx=8, pady=4)
        tk.Button(buttons, text="Apply (Enter)", command=self.apply).pack(side="left")
        tk.Button(buttons, text="Role (Ctrl+R)", command=self.toggle_role).pack(side="left", padx=4)
        tk.Button(buttons, text="Capture (Ctrl+P)", command=self.capture).pack(side="left")
        tk.Button(buttons, text="Save (Ctrl+S)", command=self.node.save).pack(side="left", padx=4)

        tk.Label(left, anchor="w", justify="left", text=(
            "Up/Down: select group   Enter: apply name + heading   Ctrl+R: human/robot   Ctrl+S: save\n"
            "Heading: type it, drag the dial / slider, or Ctrl+Left/Right (+-5 deg, +Shift +-1 deg)\n"
            "Ctrl+P: capture the LED layout (hat still, facing the dial heading) -> live yaw ('fit')\n"
            "Enter on a captured group: its CURRENT facing becomes the dial heading")).pack(fill="x", padx=8)
        self.status = tk.Label(left, anchor="w", fg="#225")
        self.status.pack(fill="x", padx=8, pady=4)

        tk.Label(right, text="Top view, grid 1 m.  solid: measured  dashed: dial").pack()
        self.preview = tk.Canvas(right, width=self.PREVIEW, height=self.PREVIEW, bg="white",
                                 highlightthickness=1, highlightbackground="#999")
        self.preview.pack()
        dial_row = tk.Frame(right)
        dial_row.pack(fill="x", pady=6)
        self.dial = tk.Canvas(dial_row, width=self.DIAL, height=self.DIAL, highlightthickness=0)
        self.dial.pack(side="left")
        self.dial.bind("<Button-1>", self.on_dial)
        self.dial.bind("<B1-Motion>", self.on_dial)
        side = tk.Frame(dial_row)
        side.pack(side="left", fill="x", expand=True, padx=8)
        tk.Label(side, text="Heading (deg)").pack(anchor="w")
        self.slider = tk.Scale(side, from_=-180, to=180, orient="horizontal", resolution=1, length=250,
                               command=lambda v: self.sync_heading(float(v), "slider"))
        self.slider.pack(fill="x")
        tk.Label(side, justify="left", fg="#555", text=(
            "dial: click/drag (snaps to 5 deg, Shift: free)\n0 = +x, 90 = +y (CCW)")).pack(anchor="w")

        self.root.bind("<Up>", lambda e: self.move(-1))
        self.root.bind("<Down>", lambda e: self.move(1))
        self.root.bind("<Return>", lambda e: self.apply())
        self.root.bind("<Control-r>", lambda e: self.toggle_role())
        self.root.bind("<Control-p>", lambda e: self.capture())
        self.root.bind("<Control-s>", lambda e: self.node.save())
        self.root.bind("<Control-Left>", lambda e: self.nudge(5))
        self.root.bind("<Control-Right>", lambda e: self.nudge(-5))
        self.root.bind("<Control-Shift-Left>", lambda e: self.nudge(1))
        self.root.bind("<Control-Shift-Right>", lambda e: self.nudge(-1))
        self.tree.bind("<<TreeviewSelect>>", lambda e: self.load_fields())
        self.name.focus_set()
        self.draw_dial(None)
        self.refresh()

    # ── heading widgets ──────────────────────────────────────────────────
    def heading_value(self):
        try:
            return float(self.heading.get())
        except ValueError:
            return None

    def sync_heading(self, deg, source):
        """Keep entry, slider and dial showing the same heading (deg, wrapped to [-180, 180))."""
        if self.syncing:
            return
        self.syncing = True
        try:
            if deg is not None:
                deg = (deg + 180.0) % 360.0 - 180.0
                if source != "entry":
                    self.heading.delete(0, "end")
                    self.heading.insert(0, f"{deg:g}")
                if source != "slider":
                    self.slider.set(round(deg))
            self.draw_dial(deg)
        finally:
            self.syncing = False

    def nudge(self, step):
        cur = self.heading_value()
        self.sync_heading((cur if cur is not None else self.measured_deg() or 0.0) + step, "key")
        return "break"

    def on_dial(self, ev):
        c = self.DIAL / 2
        deg = math.degrees(math.atan2(c - ev.y, ev.x - c))
        if not ev.state & 0x1:  # no Shift: snap to 5 deg
            deg = 5 * round(deg / 5)
        self.sync_heading(deg, "dial")

    def measured_deg(self):
        st = self.node.state.get(self.selected() or "", {})
        return None if st.get("yaw") is None else math.degrees(st["yaw"])

    def draw_dial(self, deg):
        d, c, r = self.dial, self.DIAL / 2, self.DIAL / 2 - 14
        d.delete("all")
        d.create_oval(c - r, c - r, c + r, c + r, outline="#888", width=2)
        for a in range(0, 360, 30):
            t = math.radians(a)
            k = 0.85 if a % 90 else 0.75
            d.create_line(c + k * r * math.cos(t), c - k * r * math.sin(t),
                          c + r * math.cos(t), c - r * math.sin(t), fill="#888")
        for a, txt in ((0, "+x"), (90, "+y"), (180, "-x"), (270, "-y")):
            t = math.radians(a)
            d.create_text(c + (r + 8) * math.cos(t), c - (r + 8) * math.sin(t), text=txt, font=("TkDefaultFont", 7))
        meas = self.measured_deg()
        if meas is not None:
            t = math.radians(meas)
            d.create_line(c, c, c + 0.7 * r * math.cos(t), c - 0.7 * r * math.sin(t), fill="#444", width=2,
                          arrow="last")
        if deg is not None:
            t = math.radians(deg)
            d.create_line(c, c, c + r * math.cos(t), c - r * math.sin(t), fill=self.PEND_C, width=4, arrow="last")
            d.create_text(c, c + 18, text=f"{deg:.0f}", fill=self.PEND_C, font=("TkDefaultFont", 10, "bold"))

    # ── 2D preview ───────────────────────────────────────────────────────
    def draw_preview(self, state):
        cv, W = self.preview, self.PREVIEW
        cv.delete("all")
        pts = [(st["x"], st["y"]) for st in state.values() if st["x"] is not None]
        pts += [p for st in state.values() for p in st["led_xy"]]
        if not pts:
            cv.create_text(W / 2, W / 2, text="no PhaseSpace groups visible", fill="#888")
            return
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        span = max(4.0, max(xs) - min(xs) + 3.0, max(ys) - min(ys) + 3.0)  # room for name labels
        k = W / span  # px per m

        def px(x, y):
            return W / 2 + (x - cx) * k, W / 2 - (y - cy) * k

        for g in range(math.floor(cx - span / 2), math.ceil(cx + span / 2) + 1):
            cv.create_line(*px(g, cy - span), *px(g, cy + span), fill="#bbb" if g == 0 else "#eee")
        for g in range(math.floor(cy - span / 2), math.ceil(cy + span / 2) + 1):
            cv.create_line(*px(cx - span, g), *px(cx + span, g), fill="#bbb" if g == 0 else "#eee")
        ox, oy = px(0, 0)
        cv.create_line(ox, oy, ox + 30, oy, fill="#c33", arrow="last")
        cv.create_line(ox, oy, ox, oy - 30, fill="#3a3", arrow="last")
        cv.create_text(ox + 38, oy, text="x", fill="#c33")
        cv.create_text(ox, oy - 38, text="y", fill="#3a3")

        sel, pending = self.selected(), self.heading_value()
        for hat, st in sorted(state.items()):
            for lx, ly in st["led_xy"]:
                x, y = px(lx, ly)
                cv.create_oval(x - 2, y - 2, x + 2, y + 2, fill="#999", outline="")
            if st["x"] is None:
                continue
            x, y = px(st["x"], st["y"])
            col = self.ROBOT_C if st["role"] == "robot" else self.HUMAN_C
            r = 9
            if hat == sel:
                cv.create_oval(x - r - 6, y - r - 6, x + r + 6, y + r + 6, outline=self.SEL_C, width=3)
            if st["role"] == "robot":
                cv.create_rectangle(x - r, y - r, x + r, y + r, fill=col, outline="")
            else:
                cv.create_oval(x - r, y - r, x + r, y + r, fill=col, outline="")
            if st["yaw"] is not None:
                cv.create_line(x, y, x + 34 * math.cos(st["yaw"]), y - 34 * math.sin(st["yaw"]),
                               fill=col, width=3, arrow="last")
            if hat == sel and pending is not None:
                t = math.radians(pending)
                cv.create_line(x, y, x + 46 * math.cos(t), y - 46 * math.sin(t),
                               fill=self.PEND_C, width=2, dash=(5, 3), arrow="last")
            yaw = "" if st["yaw"] is None else f" {math.degrees(st['yaw']):.0f}° {st['src']}"
            cv.create_text(x + 14, y + 14, anchor="nw", text=f"{st['name'] or hat}{yaw}",
                           fill="#000" if st["name"] else "#888")

    # ── selection / actions ──────────────────────────────────────────────
    def selected(self):
        sel = self.tree.selection()
        return sel[0] if sel else None

    def move(self, step):
        rows = self.tree.get_children()
        if not rows:
            return "break"
        cur = self.selected()
        i = (rows.index(cur) + step) % len(rows) if cur in rows else 0
        self.tree.selection_set(rows[i])
        self.tree.see(rows[i])
        self.load_fields()
        return "break"

    def load_fields(self):
        hat = self.selected()
        if hat is None:
            return
        lab = self.node.labels.get(hat, {})
        self.name.delete(0, "end")
        self.name.insert(0, lab.get("name", ""))
        self.heading.delete(0, "end")
        # start the dial at the live heading if there is one, else the stored one
        deg = self.measured_deg()
        deg = lab.get("heading_deg") if deg is None else round(deg)
        if deg is not None:
            self.heading.insert(0, f"{deg:g}")
        self.sync_heading(deg, "entry")
        self.role.config(text=f"role: {lab.get('role', 'human')}")
        self.name.focus_set()
        self.name.icursor("end")

    def apply(self):
        hat = self.selected()
        if hat:
            self.node.apply(hat, self.name.get(), self.heading.get())
        return "break"

    def toggle_role(self):
        hat = self.selected()
        if hat:
            self.node.toggle_role(hat)
            self.role.config(text=f"role: {self.node.labels[hat]['role']}")
        return "break"

    def capture(self):
        hat = self.selected()
        if hat:
            self.node.start_capture(hat, self.heading.get())
        return "break"

    def refresh(self):
        with self.node.lock:
            state = dict(self.node.state)
        fmt = lambda v, k=1.0: "-" if v is None else f"{v * k:.2f}"  # noqa: E731
        for hat, st in state.items():
            vals = (hat, st["leds"], st["name"] or "(unnamed)", st["role"], fmt(st["x"]), fmt(st["y"]),
                    "-" if st["yaw"] is None else f"{math.degrees(st['yaw']):.0f}", st["src"])
            if self.tree.exists(hat):
                self.tree.item(hat, values=vals)
            else:
                self.tree.insert("", "end", iid=hat, values=vals)
        if not self.selected() and self.tree.get_children():
            self.move(0)
        self.draw_preview(state)
        self.draw_dial(self.heading_value())
        self.status.config(text=self.node.status)
        self.root.after(100, self.refresh)


def spin_quietly(node):
    try:
        rclpy.spin(node)
    except ExternalShutdownException:  # Ctrl-C / launch shutdown
        pass


def main():
    rclpy.init()
    node = Labeler()
    spin = threading.Thread(target=spin_quietly, args=(node,), daemon=True)
    spin.start()
    try:
        Gui(node).root.mainloop()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
