#!/usr/bin/env python3
"""Runs INSIDE the g1_isaac_slam Docker image (see Dockerfile.g1_isaac_slam).

cuVSLAM (isaac_ros_visual_slam, RGBD mode) + nvblox, remapped to the sim's
D435i topics. Adapted from NVIDIA's own
isaac_ros_visual_slam_realsense_rgbd.launch.py example (same RGBD mode,
num_cameras=1, depth_camera_id=0) with frames/topics swapped for this sim:
  - image/camera_info: /camera/color/image_raw, /camera/color/camera_info
  - depth: /camera/depth/image_16uc1, produced by this file's depth_to_uint16
    node from the sim's /camera/depth/image_rect_raw. The sim (Gazebo
    rgbd_camera) publishes 32FC1 metres, but cuVSLAM's RGBD path does not
    honour the ROS encoding field — it reinterprets the buffer's raw bytes as
    uint16 unconditionally. Proven 2026-08-18 with enable_debug_mode dumps
    while feeding it 32FC1 directly: every /tmp/cuvslam_debug/depths/*.npy was
    614480 bytes = 640*480*2 + npy header, dtype uint16, min 0 / max 65535 /
    mean ~24540, adjacent pixels like [16633 16342 29888 16342 15846] —
    float32 mantissa bytes read as integers, not depth. depth_scale_factor
    cannot fix that (it scales *after* the wrong read), hence the converter
    node + depth_scale_factor=0.001 below.
    Those same dumps confirmed intrinsics/extrinsics reach cuVSLAM correctly
    (stereo.edex: focal 337.22, principal 320/240, size 640x480, valid
    rotation+translation) — depth encoding was the only broken input.)
  - base_frame: pelvis (this sim's real base link, not camera_link)
  - camera_optical_frames: camera_color_optical_frame — NVIDIA's own RGBD example
    uses a REP-103 optical-convention frame (Z-forward/X-right/Y-down), not the
    raw body-convention link frame. Image headers carry frame_id=d435_link (the
    URDF gz_frame_id), but that's a *different* param from camera_optical_frames
    — this one tells cuVSLAM which TF frame's extrinsics (relative to base_frame)
    to use for the optical axis convention. Passing d435_link here was the root
    cause of a per-frame [CUDA] error invalid argument(1) failure (fixed
    2026-08-17 second session) — this repo's sim_teleop.launch.py already
    publishes the needed d435_link -> camera_color_optical_frame static TF.
  - map/odom frame names changed to vslam_map/vslam_odom so this SLAM stack
    never claims the same TF parent/child pair as plain_slam_ros2 (repo rule
    from the earlier multi-parent TF fix).

nvblox subscribes to the same color+depth+camera_info and cuVSLAM's pose
output, reconstructing a mesh into vslam_map. people_segmentation:=true adds
isaac_ros_unet (PeopleSemSegNet, apt-installed via ros-jazzy-isaac-ros-unet)
feeding nvblox's mask input so humans are excluded from the static mesh —
least-verified part of this package, expect to iterate.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


SIM_DEPTH_TOPIC = '/camera/depth/image_rect_raw'   # 32FC1 metres, straight from Gazebo
DEPTH_TOPIC = '/camera/depth/image_16uc1'          # 16UC1 mm, what cuVSLAM/nvblox actually read


def generate_launch_description():
    # Must come up before the consumers below -- see docstring; without it
    # cuVSLAM silently tracks on noise instead of erroring.
    depth_to_uint16_node = Node(
        package='g1_isaac_slam',
        executable='depth_to_uint16',
        name='depth_to_uint16',
        parameters=[{
            'use_sim_time': True,
            'input_topic': SIM_DEPTH_TOPIC,
            'output_topic': DEPTH_TOPIC,
        }],
        output='screen',
    )

    visual_slam_node = ComposableNode(
        package='isaac_ros_visual_slam',
        plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
        name='visual_slam_node',
        parameters=[{
            'use_sim_time': True,
            'tracking_mode': 2,             # RGBD
            'depth_scale_factor': 0.001,    # depth_to_uint16 feeds 16UC1 millimetres (see docstring)
            'enable_image_denoising': False,
            'rectified_images': False,  # NOTE: True breaks single-camera RGBD mode entirely --
                                         # ("Rectified stereo camera mode only works with 1+ stereo
                                         # cameras. Number of cameras must be even.") -- it forces
                                         # cuVSLAM's stereo path regardless of num_cameras=1. Left
                                         # False despite sim's zero-distortion camera_info; the
                                         # per-frame [CUDA] error invalid argument(1) under False
                                         # is a separate, still-open issue -- see docstring above.
            'image_jitter_threshold_ms': 20.00,
            'sync_matching_threshold_ms': 10.0,
            'enable_debug_mode': True,       # DIAGNOSTIC: dump frames cuVSLAM actually receives
            'debug_dump_path': '/tmp/cuvslam_debug',
            'base_frame': 'pelvis',
            'map_frame': 'vslam_map',
            'odom_frame': 'vslam_odom',
            'enable_slam_visualization': True,
            'enable_landmarks_view': True,
            'enable_observations_view': True,
            'enable_ground_constraint_in_odometry': False,
            'enable_ground_constraint_in_slam': False,
            'enable_localization_n_mapping': True,
            'min_num_images': 1,
            'num_cameras': 1,
            'depth_camera_id': 0,
            # NVIDIA's own RGBD example uses the REP-103 *optical*-convention frame
            # (camera_color_optical_frame: Z-forward/X-right/Y-down), not the raw body-
            # convention link frame. 'd435_link' here was the likely cause of the
            # per-frame [CUDA] error invalid argument(1) failure -- cuVSLAM's projection
            # math assumes the optical axis convention. This repo's sim_teleop.launch.py
            # already publishes d435_link -> camera_color_optical_frame statically.
            'camera_optical_frames': ['camera_color_optical_frame'],
        }],
        remappings=[
            ('visual_slam/image_0', '/camera/color/image_raw'),
            ('visual_slam/camera_info_0', '/camera/color/camera_info'),
            ('visual_slam/depth_0', DEPTH_TOPIC),
        ],
    )

    visual_slam_container = ComposableNodeContainer(
        name='visual_slam_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[visual_slam_node],
        output='screen',
    )

    # nvblox: reconstructs a mesh from the same color+depth, using cuVSLAM's
    # pose. global_frame matches visual_slam's map_frame above.
    nvblox_node = Node(
        package='nvblox_ros',
        executable='nvblox_node',
        name='nvblox_node',
        parameters=[{
            'use_sim_time': True,
            'global_frame': 'vslam_map',
            # nvblox_ros has no 'base_frame' param (that name is silently ignored --
            # verified via `ros2 param list /nvblox_node`); the actual param is
            # map_clearing_frame_id, and it also defaults to 'base_link', which
            # doesn't exist in this sim's TF tree.
            'map_clearing_frame_id': 'pelvis',
            'use_depth': True,
            'use_color': True,
            'use_lidar': False,
        }],
        remappings=[
            ('color/image', '/camera/color/image_raw'),
            ('color/camera_info', '/camera/color/camera_info'),
            ('depth/image', DEPTH_TOPIC),
            ('depth/camera_info', '/camera/color/camera_info'),
        ],
        output='screen',
    )

    # people_segmentation arg accepted but NOT wired yet — isaac_ros_unet
    # (PeopleSemSegNet) bring-up + nvblox mask-input feed is Phase 2 future
    # work per the plan (least-verified part, deliberately out of this smoke
    # test). Model is already apt-installed (ros-jazzy-isaac-ros-unet dep),
    # just no launch wiring here yet.
    return LaunchDescription([
        DeclareLaunchArgument('people_segmentation', default_value='false'),
        depth_to_uint16_node,
        visual_slam_container,
        nvblox_node,
    ])
