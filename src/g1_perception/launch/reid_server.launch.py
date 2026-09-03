"""
reid_server.launch.py — Live ReID identity lookup table as a ROS2 node.

Starts reid_server_node which:
  • Subscribes to /g1/smpl/tracks (beta vectors from smpl_hmr_node)
  • Maintains a BetaReIDTable (EMA-updated, cosine-matched)
  • Publishes /g1/reid/table  (full table snapshot at publish_rate_hz)
  • Publishes /g1/reid/matches (per-frame track→identity map)
  • Accepts trigger topics: /g1/reid/enroll, /g1/reid/remove, /g1/reid/clear

Parameters (all keyword args to BetaReIDTable):
  cos_thresh         cosine sim threshold for match            (default 0.85)
  ema_alpha          EMA weight on β update                    (default 0.15)
  delta_thresh       L2 drift threshold to flag identity shift (default 0.25)
  max_table_size     LRU eviction cap                         (default 30)
  auto_enroll        auto-add stable unmatched tracks          (default false)
  min_stable_frames  frames before auto-enroll fires           (default 4)
  publish_rate_hz    /g1/reid/table publish rate               (default 10.0)

Usage (alongside live HMR stack):
    ros2 launch g1_perception smpl_full_stack.launch.py &
    ros2 launch g1_perception reid_server.launch.py

    # Override:
    ros2 launch g1_perception reid_server.launch.py \
        cos_thresh:=0.80 auto_enroll:=true

Trigger examples:
    # Enroll a person by sending their beta + label
    ros2 topic pub --once /g1/reid/enroll std_msgs/String \
      '{"data": "{\"beta\": [0.1,-0.2,...], \"label\": \"alice\"}"}'

    # Remove identity 3
    ros2 topic pub --once /g1/reid/remove std_msgs/Int32 '{"data": 3}'

    # Clear all
    ros2 topic pub --once /g1/reid/clear std_msgs/Empty '{}'
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument("cos_thresh",         default_value="0.85"),
        DeclareLaunchArgument("ema_alpha",           default_value="0.15"),
        DeclareLaunchArgument("delta_thresh",        default_value="0.25"),
        DeclareLaunchArgument("max_table_size",      default_value="30"),
        DeclareLaunchArgument("auto_enroll",         default_value="false"),
        DeclareLaunchArgument("min_stable_frames",   default_value="4"),
        DeclareLaunchArgument("publish_rate_hz",     default_value="10.0"),
    ]

    reid_server = Node(
        package="g1_perception",
        executable="reid_server_node",
        name="reid_server_node",
        output="screen",
        parameters=[{
            "cos_thresh":        LaunchConfiguration("cos_thresh"),
            "ema_alpha":         LaunchConfiguration("ema_alpha"),
            "delta_thresh":      LaunchConfiguration("delta_thresh"),
            "max_table_size":    LaunchConfiguration("max_table_size"),
            "auto_enroll":       LaunchConfiguration("auto_enroll"),
            "min_stable_frames": LaunchConfiguration("min_stable_frames"),
            "publish_rate_hz":   LaunchConfiguration("publish_rate_hz"),
        }],
    )

    return LaunchDescription(args + [reid_server])
