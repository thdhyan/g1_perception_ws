from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # NOTE: Nav2 must be separately launched (nav2_bringup) before this node.
    # This node sends goals to Nav2's /navigate_to_pose action server.
    return LaunchDescription([
        Node(
            package='g1_perception',
            executable='nav_goal_node',
            name='nav_goal_node',
            output='screen',
            parameters=[{
                'map_frame': 'map',
                'robot_base_frame': 'base_link',
                'goal_timeout_sec': 60.0,
            }],
        ),
    ])
