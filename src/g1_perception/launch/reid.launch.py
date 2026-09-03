"""
reid.launch.py — Launch ReID enrollment + matcher nodes together.

Usage:
    ros2 launch g1_perception reid.launch.py
    ros2 launch g1_perception reid.launch.py sim_threshold:=0.60

Then to enroll:
    ros2 service call /reid_enroll/enroll std_srvs/srv/Trigger {}

To clear:
    ros2 service call /reid_enroll/clear std_srvs/srv/Trigger {}

Watch the target:
    ros2 topic echo /g1/reid_target
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_WS_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..')
)
_REID_DATA = os.path.join(_WS_ROOT, 'reid_data')


def generate_launch_description():
    args = [
        DeclareLaunchArgument('model_path',      default_value=os.path.join(_REID_DATA, 'model_identity.pt')),
        DeclareLaunchArgument('enrolled_path',   default_value=os.path.join(_REID_DATA, 'enrolled_target.npy')),
        DeclareLaunchArgument('n_classes',       default_value='2'),
        DeclareLaunchArgument('emb_dim',         default_value='128'),
        DeclareLaunchArgument('n_enroll',        default_value='30'),
        DeclareLaunchArgument('min_score',       default_value='0.15'),
        DeclareLaunchArgument('max_range',       default_value='5.0'),
        DeclareLaunchArgument('sim_threshold',   default_value='0.55'),
        DeclareLaunchArgument('detection_topic', default_value='/g1/detections/livox'),
        DeclareLaunchArgument('lidar_topic',     default_value='/livox/lidar'),
        DeclareLaunchArgument('target_frame',    default_value='livox_frame'),
    ]

    shared_params = {
        'model_path':      LaunchConfiguration('model_path'),
        'enrolled_path':   LaunchConfiguration('enrolled_path'),
        'n_classes':       LaunchConfiguration('n_classes'),
        'emb_dim':         LaunchConfiguration('emb_dim'),
        'min_score':       LaunchConfiguration('min_score'),
        'max_range':       LaunchConfiguration('max_range'),
        'detection_topic': LaunchConfiguration('detection_topic'),
        'lidar_topic':     LaunchConfiguration('lidar_topic'),
    }

    enroll_node = Node(
        package='g1_perception',
        executable='reid_enroll_node',
        name='reid_enroll',
        parameters=[{**shared_params, 'n_enroll': LaunchConfiguration('n_enroll')}],
        output='screen',
        emulate_tty=True,
    )

    matcher_node = Node(
        package='g1_perception',
        executable='reid_matcher_node',
        name='reid_matcher',
        parameters=[{**shared_params,
                     'sim_threshold': LaunchConfiguration('sim_threshold'),
                     'target_frame':  LaunchConfiguration('target_frame')}],
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription(args + [enroll_node, matcher_node])
