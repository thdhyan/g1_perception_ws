#!/usr/bin/env python3
"""
ccvnorm_node.py

ROS2 node that fuses Livox Mid-360 LiDAR points with D435 RGB-D imagery using
the Stereo-LiDAR-CCVNorm network to produce a denser/refined depth map.

IMPORTANT ARCHITECTURAL NOTE (read before deploying):
Stereo-LiDAR-CCVNorm (Wang et al., IROS 2019) is a *stereo* matching network
(GCNet backbone) that fuses a sparse LiDAR-projected disparity map with a
*stereo RGB pair* (left/right) to regress dense disparity/depth for the left
camera. It was trained and evaluated on KITTI, which has genuine stereo rigs
(0.54 m baseline).

The G1's Intel RealSense D435 is an RGB-D camera: it already outputs a dense
depth image computed by its own onboard stereo IR pair, and this node only
receives the single derived depth image + single RGB image over ROS2, not
the raw IR stereo pair. There is no "right" RGB image available on
`/g1/camera/rgb` topics.

Two integration modes are supported (`~fusion_mode` param):

  1. "ccvnorm_pseudo_stereo" (default, best effort):
     Since we cannot run the real stereo-matching network without a second
     camera view, we do NOT run the trained KITTI GCNet weights (they would
     be meaningless off-domain and the model has no code path for a single
     RGB image anyway). Instead this node performs the same *conceptual*
     depth-completion job the paper targets:
       - Project LiDAR points into the camera frame via TF to get a sparse
         depth map (this reproduces the paper's "velodyne_raw" input exactly).
       - Fuse the sparse LiDAR depth with the D435's own dense depth using a
         confidence-weighted merge (LiDAR is long-range/accurate but sparse;
         D435 stereo-depth is dense but noisy/short-range/fails on
         textureless or reflective surfaces).
     This mode requires no GPU/PyTorch and is what should run on the robot
     day one.

  2. "ccvnorm_network":
     Loads the actual CCVNorm/GCNetLiDAR PyTorch model and runs real stereo
     matching. This REQUIRES a genuine stereo pair. If your D435 driver
     exposes left/right IR images (`/g1/camera/infra1`, `/g1/camera/infra2`),
     wire them in as `left_rgb`/`right_rgb` inputs (grayscale IR repeated to
     3 channels works reasonably as a substitute for RGB in the KITTI-trained
     net) and set `~left_ir_topic` / `~right_ir_topic`. You should also
     fine-tune/retrain the network on your own stereo+LiDAR data — KITTI
     weights will not generalize to indoor D435 scale/baseline/domain. This
     mode is provided as a hook for that future work; see the README in the
     Stereo-LiDAR-CCVNorm repo for what needs to change to make the network
     itself run under torch 2.7/CUDA 12.6.

Subscribes:
    /livox/mid360/points   (sensor_msgs/PointCloud2)
    /g1/camera/depth       (sensor_msgs/Image)
    /g1/camera/rgb         (sensor_msgs/Image)

Publishes:
    /g1/mapping/depth_completed  (sensor_msgs/Image, 32FC1, meters)
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from rclpy.time import Time
from rclpy.duration import Duration

import message_filters
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge

import tf2_ros
from tf2_ros import TransformException


class CCVNormNode(Node):
    """Fuses Mid-360 LiDAR + D435 RGB-D into a completed depth map."""

    def __init__(self):
        super().__init__('ccvnorm_node')

        # ---------------- Parameters ----------------
        self.declare_parameter('lidar_topic', '/livox/mid360/points')
        self.declare_parameter('depth_topic', '/g1/camera/depth')
        self.declare_parameter('rgb_topic', '/g1/camera/rgb')
        self.declare_parameter('camera_info_topic', '/g1/camera/rgb/camera_info')
        self.declare_parameter('output_topic', '/g1/mapping/depth_completed')
        self.declare_parameter('camera_frame', 'd435_color_optical_frame')
        self.declare_parameter('fusion_mode', 'ccvnorm_pseudo_stereo')  # or 'ccvnorm_network'
        self.declare_parameter('max_lidar_range_m', 40.0)
        self.declare_parameter('min_lidar_range_m', 0.2)
        self.declare_parameter('sync_slop_sec', 0.05)
        self.declare_parameter('lidar_confidence_radius_px', 2)
        self.declare_parameter('depth_scale_m', 1.0)  # multiply depth image values to get meters
        # Camera intrinsics fallback if no CameraInfo received yet (D435 720p color defaults)
        self.declare_parameter('fx', 617.0)
        self.declare_parameter('fy', 617.0)
        self.declare_parameter('cx', 320.0)
        self.declare_parameter('cy', 240.0)
        # ccvnorm_network mode only
        self.declare_parameter('ccvnorm_repo_path', '')
        self.declare_parameter('ccvnorm_ckpt_path', '')
        self.declare_parameter('ccvnorm_norm_mode', 'categorical_hier')
        self.declare_parameter('left_ir_topic', '/g1/camera/infra1')
        self.declare_parameter('right_ir_topic', '/g1/camera/infra2')

        self.lidar_topic = self.get_parameter('lidar_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.fusion_mode = self.get_parameter('fusion_mode').value
        self.max_range = self.get_parameter('max_lidar_range_m').value
        self.min_range = self.get_parameter('min_lidar_range_m').value
        self.slop = self.get_parameter('sync_slop_sec').value
        self.conf_radius = int(self.get_parameter('lidar_confidence_radius_px').value)
        self.depth_scale = self.get_parameter('depth_scale_m').value

        self.fx = self.get_parameter('fx').value
        self.fy = self.get_parameter('fy').value
        self.cx = self.get_parameter('cx').value
        self.cy = self.get_parameter('cy').value
        self._have_camera_info = False

        self.bridge = CvBridge()

        # ---------------- TF ----------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---------------- CCVNorm model (optional) ----------------
        self.model = None
        self.torch = None
        if self.fusion_mode == 'ccvnorm_network':
            self._load_ccvnorm_model()

        # ---------------- Subscriptions ----------------
        # CameraInfo updates intrinsics opportunistically; not synchronized.
        self.create_subscription(
            CameraInfo, self.camera_info_topic, self._camera_info_cb,
            QoSPresetProfiles.SENSOR_DATA.value)

        self.sub_lidar = message_filters.Subscriber(
            self, PointCloud2, self.lidar_topic, qos_profile=QoSPresetProfiles.SENSOR_DATA.value)
        self.sub_depth = message_filters.Subscriber(
            self, Image, self.depth_topic, qos_profile=QoSPresetProfiles.SENSOR_DATA.value)
        self.sub_rgb = message_filters.Subscriber(
            self, Image, self.rgb_topic, qos_profile=QoSPresetProfiles.SENSOR_DATA.value)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_lidar, self.sub_depth, self.sub_rgb],
            queue_size=10, slop=self.slop)
        self.ts.registerCallback(self.sync_callback)

        # ---------------- Publisher ----------------
        self.pub_depth = self.create_publisher(
            Image, self.output_topic, QoSPresetProfiles.SENSOR_DATA.value)

        self.get_logger().info(
            f"ccvnorm_node started (mode={self.fusion_mode}). "
            f"Subscribing: {self.lidar_topic}, {self.depth_topic}, {self.rgb_topic}. "
            f"Publishing: {self.output_topic}"
        )

    # ------------------------------------------------------------------
    def _camera_info_cb(self, msg: CameraInfo):
        k = msg.k if hasattr(msg, 'k') else msg.K
        self.fx, self.fy = k[0], k[4]
        self.cx, self.cy = k[2], k[5]
        self._have_camera_info = True

    # ------------------------------------------------------------------
    def _load_ccvnorm_model(self):
        """Load the real CCVNorm/GCNetLiDAR network for 'ccvnorm_network' mode.

        Requires a genuine stereo pair (see module docstring) and a
        checkpoint. See README in Stereo-LiDAR-CCVNorm for the torch
        2.7 / CUDA 12.6 porting notes (Variable() calls, deprecated
        torch.batch_norm signature, etc.) before relying on this path.
        """
        import sys
        repo_path = self.get_parameter('ccvnorm_repo_path').value
        ckpt_path = self.get_parameter('ccvnorm_ckpt_path').value
        if not repo_path or not ckpt_path:
            self.get_logger().error(
                "fusion_mode=ccvnorm_network requires 'ccvnorm_repo_path' and "
                "'ccvnorm_ckpt_path' parameters. Falling back to pseudo_stereo fusion."
            )
            self.fusion_mode = 'ccvnorm_pseudo_stereo'
            return

        sys.path.insert(0, repo_path)
        import torch
        from model.gcnet_lidar import GCNetLiDAR

        self.torch = torch
        norm_mode = self.get_parameter('ccvnorm_norm_mode').value
        self.model = GCNetLiDAR(maxdisparity=192, norm_mode=norm_mode)
        state = torch.load(ckpt_path, map_location='cuda')
        self.model.load_state_dict(state['model'])
        self.model = self.model.cuda().eval()
        self.get_logger().info(f"Loaded CCVNorm checkpoint from {ckpt_path}")

    # ------------------------------------------------------------------
    def sync_callback(self, lidar_msg: PointCloud2, depth_msg: Image, rgb_msg: Image):
        try:
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            depth_img = depth_img.astype(np.float32) * self.depth_scale
            rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().warn(f"cv_bridge conversion failed: {e}")
            return

        h, w = depth_img.shape[:2]

        sparse_depth = self._project_lidar_to_image(lidar_msg, h, w)
        if sparse_depth is None:
            return  # TF not ready yet; drop frame

        if self.fusion_mode == 'ccvnorm_network' and self.model is not None:
            completed = self._run_ccvnorm_network(rgb_img, sparse_depth)
        else:
            completed = self._fuse_pseudo_stereo(depth_img, sparse_depth)

        out_msg = self.bridge.cv2_to_imgmsg(completed.astype(np.float32), encoding='32FC1')
        out_msg.header = depth_msg.header
        self.pub_depth.publish(out_msg)

    # ------------------------------------------------------------------
    def _project_lidar_to_image(self, lidar_msg: PointCloud2, h: int, w: int):
        """Project Livox points into the camera frame -> sparse depth map (meters)."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.camera_frame, lidar_msg.header.frame_id,
                Time.from_msg(lidar_msg.header.stamp),
                timeout=Duration(seconds=0.05))
        except TransformException as e:
            self.get_logger().warn(f"TF lookup {lidar_msg.header.frame_id}->{self.camera_frame} failed: {e}",
                                    throttle_duration_sec=2.0)
            return None

        pts = np.array(list(pc2.read_points(
            lidar_msg, field_names=('x', 'y', 'z'), skip_nans=True)), dtype=np.float32)
        if pts.size == 0:
            return np.zeros((h, w), dtype=np.float32)

        R, t = self._transform_to_rt(tf)
        pts_cam = (R @ pts.T).T + t  # (N,3) in camera optical frame (x-right, y-down, z-forward)

        z = pts_cam[:, 2]
        valid = (z > self.min_range) & (z < self.max_range)
        pts_cam = pts_cam[valid]
        z = z[valid]

        u = (self.fx * pts_cam[:, 0] / z + self.cx).round().astype(np.int32)
        v = (self.fy * pts_cam[:, 1] / z + self.cy).round().astype(np.int32)
        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, z = u[in_bounds], v[in_bounds], z[in_bounds]

        sparse = np.zeros((h, w), dtype=np.float32)
        # Keep the nearest point per pixel (handles overlapping projections).
        order = np.argsort(-z)  # far to near, so near overwrites
        sparse[v[order], u[order]] = z[order]

        if self.conf_radius > 0:
            sparse = self._dilate_sparse(sparse, self.conf_radius)
        return sparse

    @staticmethod
    def _transform_to_rt(tf_msg):
        q = tf_msg.transform.rotation
        t = tf_msg.transform.translation
        x, y, z, wq = q.x, q.y, q.z, q.w
        R = np.array([
            [1 - 2*(y*y+z*z), 2*(x*y - z*wq), 2*(x*z + y*wq)],
            [2*(x*y + z*wq), 1 - 2*(x*x+z*z), 2*(y*z - x*wq)],
            [2*(x*z - y*wq), 2*(y*z + x*wq), 1 - 2*(x*x+y*y)],
        ], dtype=np.float32)
        t_vec = np.array([t.x, t.y, t.z], dtype=np.float32)
        return R, t_vec

    @staticmethod
    def _dilate_sparse(sparse: np.ndarray, radius: int):
        """Thicken single-pixel LiDAR hits into small discs so downstream
        fusion/visualization is not one-pixel sparse. Nearest-point wins on overlap."""
        import cv2
        mask = (sparse > 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
        dilated_mask = cv2.dilate(mask, kernel)
        # Use a min-filter on the (large-value-for-empty) depth to keep nearest point per neighborhood.
        filler = np.where(sparse > 0, sparse, np.inf).astype(np.float32)
        min_filtered = cv2.erode(filler, kernel)  # erode == min-filter for grayscale
        out = np.where((dilated_mask > 0) & (sparse == 0), min_filtered, sparse)
        out[~np.isfinite(out)] = 0.0
        return out

    # ------------------------------------------------------------------
    def _fuse_pseudo_stereo(self, dense_depth: np.ndarray, sparse_lidar: np.ndarray):
        """Confidence-weighted merge of D435 dense depth with sparse LiDAR depth.

        LiDAR is treated as ground truth where available (long range, accurate,
        works on glass/dark surfaces where stereo IR fails); the D435's own
        depth fills the gaps. This is the practical stand-in for CCVNorm's
        depth-completion role when no true stereo pair is available.
        """
        completed = dense_depth.copy()
        valid_dense = dense_depth > 0
        valid_lidar = sparse_lidar > 0

        # Where both exist and disagree a lot, trust LiDAR (more accurate depth sensor).
        both = valid_dense & valid_lidar
        disagreement = np.abs(dense_depth - sparse_lidar) > 0.3  # 30 cm
        completed[both & disagreement] = sparse_lidar[both & disagreement]

        # Where only LiDAR has data (D435 dropout / out of its ~10m range), use LiDAR.
        lidar_only = valid_lidar & ~valid_dense
        completed[lidar_only] = sparse_lidar[lidar_only]

        # Where neither has data, leave as zero (invalid).
        return completed

    # ------------------------------------------------------------------
    def _run_ccvnorm_network(self, rgb_img: np.ndarray, sparse_depth: np.ndarray):
        """Run the actual GCNetLiDAR/CCVNorm model. Requires a real stereo pair;
        this default implementation duplicates the single RGB image as both
        left/right, which will NOT produce a meaningful disparity (zero
        baseline). Wire in real left/right IR images before using this path
        in production -- see module docstring."""
        torch = self.torch
        h, w = sparse_depth.shape

        def to_tensor(img):
            t = torch.from_numpy(img).float() / 255.0
            if t.dim() == 2:
                t = t.unsqueeze(-1).repeat(1, 1, 3)
            return t.permute(2, 0, 1).unsqueeze(0).cuda()

        left_rgb = to_tensor(rgb_img)
        right_rgb = to_tensor(rgb_img)  # PLACEHOLDER: see docstring
        sd = torch.from_numpy(sparse_depth).float().unsqueeze(0).unsqueeze(0).cuda()

        inputs = {
            'left_rgb': left_rgb, 'right_rgb': right_rgb,
            'left_sd': sd, 'right_sd': torch.zeros_like(sd),
        }
        with torch.no_grad():
            pred_disp = self.model(inputs, mode='test')
        pred_disp_np = pred_disp.squeeze().cpu().numpy()

        depth = (self.fx * 0.05) / np.clip(pred_disp_np, 1e-3, None)  # assumed small baseline
        return depth.astype(np.float32)


def main(args=None):
    rclpy.init(args=args)
    node = CCVNormNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
