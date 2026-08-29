"""Pose backend registry.

A backend estimates a single person's 3D skeleton from the point cloud
cropped to that person's detection box:

    keypoints, valid, score = backend.infer(points, box7)

where `points` is (N,4) float32 [x,y,z,intensity] in the cloud frame,
`box7` is [x,y,z,dx,dy,dz,yaw], `keypoints` is (K,3) and `valid` is (K,)
uint8 in {0,1}, both expressed in the same frame as the input points.
"""

from .debug_backend import DebugBackend

_BACKENDS = {
    "debug": DebugBackend,
}


def get_backend(name: str):
    """Return an instantiated backend by name."""
    cls = _BACKENDS.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown pose backend '{name}'. Available: {sorted(_BACKENDS)}"
        )
    return cls()
