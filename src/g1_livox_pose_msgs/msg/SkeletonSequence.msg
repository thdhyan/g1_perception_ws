# Time-continuous pose history of ONE tracked person.
# All frames share track_id and coordinate frame; stamps strictly increase.
std_msgs/Header header

# Stamp of the newest frame; frame_id is the stable reference frame
# (e.g. odom), NOT a robot body frame.
uint32 track_id
PersonPose3D[] frames

# Convenience: newest stamp - oldest stamp.
builtin_interfaces/Duration window
