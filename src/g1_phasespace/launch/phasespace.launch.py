"""PhaseSpace tracker + SMPL-X humans + G1 head + G1 model + RViz.

ros2 launch g1_phasespace phasespace.launch.py [robot_group:=dhyan-hat-3] [rviz:=false]

G1 model: the hat is fixed on the robot's head, so a static TF ps_robot_head -> ps/pelvis
(hat_to_pelvis_x/y/z, hat frame = robot heading) hangs the 29-DOF URDF (robot_state_publisher,
frame_prefix ps/) under the tracked head. Joints are zero (joint_state_publisher) unless
model_joints:=robot, which uses the robot's own /joint_states. Default z: head top is 0.525 m
above the pelvis in the URDF, the LED centroid sits ~2.5 cm above it; tune by eye in RViz.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot_group = LaunchConfiguration("robot_group")
    return LaunchDescription([
        DeclareLaunchArgument("device", default_value="cs-phasespace.cs.umn.edu"),
        DeclareLaunchArgument("profile", default_value="", description="session profile (empty: server default)"),
        DeclareLaunchArgument("robot_group", default_value="dhyan-hat-3", description="LED group on the G1 head"),
        DeclareLaunchArgument("smplx_model", default_value="~/Projects/HOVER/body_models/smplx/SMPLX_NEUTRAL.npz"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("tracker", default_value="true",
                              description="false: no phasespace_ros2 (fake_phasespace publishes the groups)"),
        DeclareLaunchArgument("robot_model", default_value="true", description="G1 URDF under the tracked head"),
        DeclareLaunchArgument("model_joints", default_value="zero", description="zero | robot (/joint_states)"),
        DeclareLaunchArgument("hat_to_pelvis_x", default_value="0.0"),
        DeclareLaunchArgument("hat_to_pelvis_y", default_value="0.0"),
        DeclareLaunchArgument("hat_to_pelvis_z", default_value="-0.55"),
        GroupAction(scoped=True, condition=IfCondition(LaunchConfiguration("tracker")), actions=[IncludeLaunchDescription(  # scoped: its rviz:=false must not leak to ours
            PathJoinSubstitution([FindPackageShare("phasespace_ros2"), "launch", "phasespace.launch.py"]),
            launch_arguments={"device": LaunchConfiguration("device"), "profile": LaunchConfiguration("profile"),
                              "rviz": "false"}.items(),
        )]),
        Node(package="g1_phasespace", executable="humans", output="screen",
             parameters=[{"robot_group": robot_group, "smplx_model": LaunchConfiguration("smplx_model")}]),
        Node(package="g1_phasespace", executable="robot_head", output="screen",
             parameters=[{"robot_group": robot_group}]),
        Node(package="tf2_ros", executable="static_transform_publisher", name="hat_to_pelvis", output="log",
             condition=IfCondition(LaunchConfiguration("robot_model")),
             arguments=["--x", LaunchConfiguration("hat_to_pelvis_x"), "--y", LaunchConfiguration("hat_to_pelvis_y"),
                        "--z", LaunchConfiguration("hat_to_pelvis_z"),
                        "--frame-id", "ps_robot_head", "--child-frame-id", "ps/pelvis"]),
        Node(package="robot_state_publisher", executable="robot_state_publisher", namespace="ps_model",
             output="log", condition=IfCondition(LaunchConfiguration("robot_model")),
             parameters=[{"frame_prefix": "ps/", "robot_description": ParameterValue(
                 Command(["cat ", PathJoinSubstitution([FindPackageShare("g1_description"), "urdf", "g1_29dof.urdf"])]),
                 value_type=str)}],
             remappings=[("joint_states", PythonExpression(
                 ["'/joint_states' if '", LaunchConfiguration("model_joints"), "' == 'robot' else 'joint_states'"]))]),
        Node(package="joint_state_publisher", executable="joint_state_publisher", namespace="ps_model",
             output="log", condition=IfCondition(PythonExpression(
                 ["'", LaunchConfiguration("robot_model"), "' == 'true' and '",
                  LaunchConfiguration("model_joints"), "' != 'robot'"]))),
        Node(
            package="rviz2", executable="rviz2", output="log", condition=IfCondition(LaunchConfiguration("rviz")),
            arguments=["-d", PathJoinSubstitution([FindPackageShare("g1_phasespace"), "rviz", "phasespace.rviz"])],
        ),
    ])
