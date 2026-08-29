# One person's 3D skeleton at a single time instant.
# Joint coordinates follow JOINT_ORDER documented in the package README
# (Waymo 14-joint convention): flattened as [x0,y0,z0, x1,y1,z1, ...].
std_msgs/Header header

# Persistent identity across frames; 0 means "unassigned" and is filled in
# by the sequence assembler.
uint32 track_id

# K*3 floats, K = number of joints (14).
float32[] joints

# Per-joint validity: 1 = estimated, 0 = occluded/missing (coords zeroed).
uint8[] valid

float32 pose_score
