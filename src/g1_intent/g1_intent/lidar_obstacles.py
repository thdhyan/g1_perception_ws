"""Walls and tables from the robot's Mid-360 as axis-aligned boxes for the intent planner.

Each /livox/lidar sweep is transformed into fixed_frame (default phasespace, through
the G1 model hung under the head hat: phasespace -> ps_robot_head -> ps/pelvis ->
ps/mid360_link). Points in the z_min..z_max band (floor and ceiling out), away from
every /mocap/agents agent (people and the robot itself are not obstacles), vote into
a decaying 2D grid. Occupied cells merge into rectangles (row runs stacked into
boxes), published latched as JSON for intent_planner / web_console, which add them
to the intent-sim scene's `obstacles` (prompt + Nav/VO cost maps + A* waypoints).

sensor_frame overrides the cloud's frame_id: the robot's own TF names it mid360_link
(parent torso_link, no route to phasespace), the laptop's G1 model names it
ps/mid360_link. With fixed_frame:=lio_map (or map) and sensor_frame:="" it maps in
the SLAM frame instead, without PhaseSpace.

In:  /livox/lidar, /mocap/agents, TF
Out: /intent/obstacles (String JSON, latched): {"frame", "stamp", "boxes": [[x0, y0, x1, y1], ...]}
     /intent/obstacle_grid (OccupancyGrid, for RViz)
"""

import json

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def quat_to_rot(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mask_to_boxes(mask):
    """Occupied cells -> rectangles (i0, j0, i1, j1), inclusive: runs along each row,
    a run identical to one in the row above extends that box downward."""
    boxes, open_ = [], {}  # (j0, j1) -> box index, for runs in the previous row
    for i in range(mask.shape[0]):
        row = np.concatenate(([0], mask[i].astype(np.int8), [0]))
        d = np.diff(row)
        runs = list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))
        nxt = {}
        for j0, j1 in runs:
            k = open_.get((j0, j1))
            if k is None:
                boxes.append([i, j0, i, j1])
                k = len(boxes) - 1
            else:
                boxes[k][2] = i
            nxt[(j0, j1)] = k
        open_ = nxt
    return boxes


class LidarObstacles(Node):
    def __init__(self):
        super().__init__("lidar_obstacles")
        p = lambda n, v: self.declare_parameter(n, v).value  # noqa: E731
        self.fixed_frame = p("fixed_frame", "phasespace")
        self.sensor_frame = p("sensor_frame", "ps/mid360_link")
        # mapped area in fixed_frame: the intent-sim canvas (its top-left at canvas_origin, y down)
        ox, oy = p("canvas_origin_x", -5.0), p("canvas_origin_y", 3.5)
        self.x0, self.x1 = ox, ox + p("canvas_w", 10.0)
        self.y0, self.y1 = oy - p("canvas_h", 7.0), oy
        self.res = p("resolution", 0.1)
        self.z_min, self.z_max = p("z_min", 0.15), p("z_max", 1.6)
        self.max_range = p("max_range", 8.0)
        self.agent_radius = p("agent_radius", 0.5)  # points this close to a tracked agent are the agent
        self.decay = p("decay", 0.97)  # per sweep (10 Hz: ~3 s half-life)
        self.occupied = p("occupied_score", 3.0)
        self.min_cells = p("min_box_cells", 2)
        self.max_boxes = p("max_boxes", 25)

        self.nx = int(round((self.x1 - self.x0) / self.res))
        self.ny = int(round((self.y1 - self.y0) / self.res))
        self.score = np.zeros((self.ny, self.nx), np.float32)
        self.agents = []

        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.create_subscription(PointCloud2, p("cloud_topic", "/livox/lidar"), self.on_cloud, qos_profile_sensor_data)
        self.create_subscription(String, "/mocap/agents", self.on_agents, 10)
        self.box_pub = self.create_publisher(String, "/intent/obstacles", LATCHED)
        self.grid_pub = self.create_publisher(OccupancyGrid, "/intent/obstacle_grid", 1)
        self.create_timer(1.0 / p("publish_hz", 1.0), self.publish)
        self.warned = False
        self.get_logger().info(f"LiDAR obstacles in {self.fixed_frame} via {self.sensor_frame or '<cloud frame>'}, "
                               f"{self.nx}x{self.ny} cells of {self.res} m, z {self.z_min}..{self.z_max} m")

    def on_agents(self, msg):
        self.agents = [(a["x"], a["y"]) for a in json.loads(msg.data).get("agents", [])]

    def on_cloud(self, msg):
        src = self.sensor_frame or msg.header.frame_id
        try:  # latest, not msg stamp: the laptop G1 model and the robot clocks are not synced
            t = self.tf.lookup_transform(self.fixed_frame, src, Time(), timeout=Duration(seconds=0.05))
        except Exception as e:  # noqa: BLE001  (tf2 raises several unrelated types)
            if not self.warned:
                self.get_logger().warn(f"no TF {self.fixed_frame} <- {src} yet: {e}")
                self.warned = True
            return
        self.warned = False
        pts = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True).astype(np.float32)
        pts = pts[np.einsum("ij,ij->i", pts, pts) < self.max_range ** 2]
        tr = t.transform.translation
        w = pts @ quat_to_rot(t.transform.rotation).T.astype(np.float32) + np.array([tr.x, tr.y, tr.z], np.float32)
        w = w[(w[:, 2] > self.z_min) & (w[:, 2] < self.z_max)]
        for ax, ay in self.agents + [(tr.x, tr.y)]:
            w = w[(w[:, 0] - ax) ** 2 + (w[:, 1] - ay) ** 2 > self.agent_radius ** 2]
        j = ((w[:, 0] - self.x0) / self.res).astype(int)
        i = ((w[:, 1] - self.y0) / self.res).astype(int)
        ok = (i >= 0) & (i < self.ny) & (j >= 0) & (j < self.nx)
        hits = np.zeros_like(self.score)
        hits[i[ok], j[ok]] = 1.0  # one vote per cell per sweep
        self.score = self.score * self.decay + hits

    def publish(self):
        mask = self.score >= self.occupied
        boxes = [b for b in mask_to_boxes(mask) if (b[2] - b[0] + 1) * (b[3] - b[1] + 1) >= self.min_cells]
        boxes.sort(key=lambda b: -(b[2] - b[0] + 1) * (b[3] - b[1] + 1))
        out = [[round(self.x0 + j0 * self.res, 2), round(self.y0 + i0 * self.res, 2),
                round(self.x0 + (j1 + 1) * self.res, 2), round(self.y0 + (i1 + 1) * self.res, 2)]
               for i0, j0, i1, j1 in boxes[:self.max_boxes]]
        now = self.get_clock().now()
        self.box_pub.publish(String(data=json.dumps({"frame": self.fixed_frame, "stamp": now.nanoseconds * 1e-9,
                                                     "boxes": out})))
        g = OccupancyGrid()
        g.header.frame_id, g.header.stamp = self.fixed_frame, now.to_msg()
        g.info.resolution, g.info.width, g.info.height = self.res, self.nx, self.ny
        g.info.origin.position.x, g.info.origin.position.y = self.x0, self.y0
        g.info.origin.orientation.w = 1.0
        g.data = (np.clip(self.score / self.occupied, 0, 1) * 100).astype(np.int8).ravel().tolist()
        self.grid_pub.publish(g)


def main():
    rclpy.init()
    node = LidarObstacles()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
