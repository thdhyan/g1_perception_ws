from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='g1_perception',
            executable='ccvnorm_node',
            name='ccvnorm_node',
            output='screen',
            parameters=[{
                'lidar_topic': '/livox/mid360/points',
                'depth_topic': '/g1/camera/depth',
                'rgb_topic': '/g1/camera/rgb',
                'output_topic': '/g1/mapping/depth_completed',
                'camera_frame': 'd435_color_optical_frame',
                'fusion_mode': 'ccvnorm_pseudo_stereo',
            }],
        ),
    ])
