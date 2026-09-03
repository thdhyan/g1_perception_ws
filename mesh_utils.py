#!/usr/bin/env python3
"""
mesh_utils.py — lightweight SMPL forward (beta, theta) -> mesh vertices.

Uses only the SMPL_layer from LiDAR-HMR (no Point-Transformer backbone),
so this is cheap: CPU-only, no CUDA needed.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HMR_DIR = os.path.join(HERE, "LiDAR-HMR")


class SMPLMeshDecoder:
    """theta: (72,) = [global_orient(3), pose_axis_angle(69)]; beta: (10,)."""

    def __init__(self, device="cpu"):
        if HMR_DIR not in sys.path:
            sys.path.insert(0, HMR_DIR)
        import torch
        self.torch = torch
        self.device = torch.device(device)

        cwd = os.getcwd()
        os.chdir(HMR_DIR)  # SMPL_layer loads files via relative paths
        try:
            from models.smpl_hybrik.SMPL import SMPL_layer
            j36m = np.load("models/data/J_regressor_h36m_correct.npy")
            self.layer = SMPL_layer(
                model_path="smplx_models/smpl/SMPL_NEUTRAL.pkl",
                indx_vert_path="models/indx_vert.pkl",
                num_joints=24, h36m_jregressor=j36m)
        finally:
            os.chdir(cwd)
        self.layer.to(self.device)
        self.layer.eval()
        self.faces = self.layer.faces_tensor.cpu().numpy().astype(np.int32)  # (F, 3)

    def vertices(self, beta, theta, trans=None):
        """beta: (10,) or (B,10); theta: (72,) or (B,72); trans: (3,) or (B,3) or None.
        Returns (verts, joints) as numpy, shape (B, 6890, 3) / (B, 29, 3)."""
        torch = self.torch
        beta = np.atleast_2d(np.asarray(beta, dtype=np.float32))
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float32))
        with torch.no_grad():
            b = torch.from_numpy(beta).to(self.device)
            th = torch.from_numpy(theta).to(self.device)
            global_orient = th[:, :3]
            pose = th[:, 3:]
            tr = None
            if trans is not None:
                trans = np.atleast_2d(np.asarray(trans, dtype=np.float32))
                tr = torch.from_numpy(trans).to(self.device)
            out = self.layer.forward(pose, b, global_orient, transl=tr)
        return out.vertices.cpu().numpy(), out.joints.cpu().numpy()


_singleton = None


def get_decoder():
    global _singleton
    if _singleton is None:
        _singleton = SMPLMeshDecoder()
    return _singleton
