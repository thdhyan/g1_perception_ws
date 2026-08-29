cmake_minimum_required(VERSION 3.8)
project(g1_livox_pose_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PersonPose3D.msg"
  "msg/PersonPose3DArray.msg"
  "msg/SkeletonSequence.msg"
  "msg/SkeletonSequenceArray.msg"
  DEPENDENCIES std_msgs builtin_interfaces
)

ament_package()
