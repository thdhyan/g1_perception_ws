"""
reid_pcr.py — Point-cloud-ReID (WACV 2024, bentherien/point-cloud-reid) backbone loader.

Goal
----
Extract a per-observation ReID embedding (128-d, L2-normalised) from a LiDAR crop
using point-cloud-reid's *pre-trained* point backbones (Point-Transformer / PointNet /
DGCNN on nuScenes ReID).  These models were trained as Siamese matchers, but the single
observation feature the pipeline needs is the **pooled backbone feature** (max+avg pool
over points -> 128-d), which is exactly what the model's own classification / matching
heads consume (`get_pooled_feats`).  We reuse that feature directly.

Light by design
----------------
point-cloud-reid is built on lamtk / mmdet / mmcv / pytorch3d / torchpack (heavy, and
they are NOT installed in the laptop venv).  We only need a *backbone forward pass*, so
this module loads just the three self-contained, **pure-PyTorch** sub-modules straight
from the cloned repo, bypassing the failing `mmdet3d/__init__.py`:

    pointnet2_utils.py  -> SA/FP layers  (used by Pointnet_Backbone, the point-transformer)
    backbone_net.py     -> Pointnet_Backbone
    pointnet.py         -> PointNet
    dgcnn_orig.py       -> DGCNN
    lanegcn_nets.py     -> LinearRes (the `downsample` head for PointNet/DGCNN)

None of those import mmdet/mmcv/pytorch3d.  So this runs with **only torch + numpy**.

Public API
----------
    PCRFeats(name, backbone, downsample, pool_type)   small nn.Module: (B,N,3) -> (B,128)
    load_pcr(name, ckpt_path, device) -> (PCRFeats.eval(), report)
    extract_embeddings(ckpt, crops_np, name, device, batch, tag) -> (n,128) float32
"""
import importlib.util
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PCREID_DIR = os.path.join(HERE, "point-cloud-reid")
PCR_MODELS_DIR = os.path.join(PCREID_DIR, "mmdet3d", "models")

# Model name -> (checkpoint basename, short tag used in emb_<session>_<tag>_*.npy)
PCR_MODELS = {
    "point-transformer": (
        "pts_point-transformer_r_nus_det_500e.pth",
        "pcr_point-transformer_nus",
    ),
    "pointnet": (
        "pts_pointnet_r_nus_det_500e.pth",
        "pcr_pointnet_nus",
    ),
    "dgcnn": (
        "pts_dgcnn_r_nus_det_500e.pth",
        "pcr_dgcnn_nus",
    ),
}

# Default checkpoint search dirs (root-disk is often full -> prefer data disks).
DEFAULT_CKPT_DIRS = [
    os.path.join(PCREID_DIR, "pretrained", "nuscenes"),
    os.path.expanduser("~/reid_pcr_models"),
    "/generalSSD/reid_pcr_models",
]


# ────────────────────────────────────────────────────────────────────────────
# Load the self-contained backbone modules from the repo (no package __init__ run)
# ────────────────────────────────────────────────────────────────────────────
_REPO_MODULES = None


def _load_repo_modules():
    """Import the pure-torch sub-modules from the cloned repo, WITHOUT executing
    mmdet3d/__init__.py (that pulls in mmcv/mmdet/pytorch3d which are absent)."""
    global _REPO_MODULES
    if _REPO_MODULES is not None:
        return _REPO_MODULES

    if not os.path.isdir(PCR_MODELS_DIR):
        raise FileNotFoundError(
            f"point-cloud-reid not found at {PCREID_DIR}. "
            "Run:  git clone https://github.com/bentherien/point-cloud-reid.git "
            f"{PCREID_DIR}"
        )

    pkg_name = "__pcr_models__"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [PCR_MODELS_DIR]          # lets `from .x import y` resolve
        sys.modules[pkg_name] = pkg

    def _mod(name, fname):
        full = f"{pkg_name}.{name}"
        if full in sys.modules:
            return sys.modules[full]
        spec = importlib.util.spec_from_file_location(full, os.path.join(PCR_MODELS_DIR, fname))
        m = importlib.util.module_from_spec(spec)
        m.__package__ = pkg_name
        sys.modules[full] = m
        spec.loader.exec_module(m)
        return m

    # pointnet2_utils is a relative dependency of backbone_net -> must exist first.
    _mod("pointnet2_utils", "pointnet2_utils.py")
    Pointnet_Backbone = _mod("backbone_net", "backbone_net.py").Pointnet_Backbone
    PointNet = _mod("pointnet", "pointnet.py").PointNet
    DGCNN = _mod("dgcnn_orig", "dgcnn_orig.py").DGCNN
    LinearRes = _mod("lanegcn_nets", "lanegcn_nets.py").LinearRes

    _REPO_MODULES = (Pointnet_Backbone, PointNet, DGCNN, LinearRes)
    return _REPO_MODULES


# ────────────────────────────────────────────────────────────────────────────
# Single-observation feature extractor
# ────────────────────────────────────────────────────────────────────────────
class PCRFeats(nn.Module):
    """(B, N, 3) -> (B, 128) pooled (max+avg) backbone feature, pool_type-agnostic."""

    def __init__(self, name: str, backbone, downsample, pool_type="both"):
        super().__init__()
        self.name = name
        self.pool_type = pool_type
        self.use_down = downsample is not None      # PointNet / DGCNN path
        self.backbone = backbone
        self.downsample = downsample                # None for point-transformer
        self.backbone_list = [128, 64, 32]          # matches configs (numpoints for SA stages)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """pts: (B, N, 3) float32 point cloud (centre / de-yaw already applied)."""
        if self.use_down:
            # PointNet / DGCNN: input layout (B, C=3, N), output (B, 1024, N)
            # -> per-point flatten -> downsample -> (B, 64, N)
            _, h = self.backbone(pts.permute(0, 2, 1), self.backbone_list)
            B, N = pts.shape[0], pts.shape[1]
            C = h.shape[-1]
            h = h.permute(0, 2, 1).reshape(-1, C)
            h = self.downsample(h)
            h = h.reshape(B, N, -1).permute(0, 2, 1)                 # (B, 64, N)
        else:
            # Point-Transformer (Pointnet_Backbone / pointnet2): input (B, N, 3)
            _, h = self.backbone(pts, self.backbone_list)            # (B, 64, N)

        if self.pool_type == "both":
            x1 = F.adaptive_max_pool1d(h, 1).reshape(h.size(0), -1)
            x2 = F.adaptive_avg_pool1d(h, 1).reshape(h.size(0), -1)
            return torch.cat((x1, x2), dim=1)
        elif self.pool_type == "max":
            return F.adaptive_max_pool1d(h, 1).reshape(h.size(0), -1)
        elif self.pool_type == "avg":
            return F.adaptive_avg_pool1d(h, 1).reshape(h.size(0), -1)
        raise NotImplementedError(self.pool_type)


def build_pcr(name: str) -> PCRFeats:
    """Construct (untrained) backbone + optional downsample for a given model name."""
    if not os.path.isdir(PCR_MODELS_DIR):
        raise FileNotFoundError(f"point-cloud-reid missing at {PCREID_DIR}")
    Pointnet_Backbone, PointNet, DGCNN, LinearRes = _load_repo_modules()
    name = name.lower()

    if name == "point-transformer":
        backbone = Pointnet_Backbone(input_channels=0, use_xyz=True, conv_out=64, mul=1)
        return PCRFeats(name, backbone=None, downsample=None, pool_type="both") \
            if False else PCRFeats(name, backbone, None, "both")
    if name == "pointnet":
        backbone = PointNet(k=40, normal_channel=False)
        down = nn.Sequential(
            LinearRes(1024, 512, norm="GN", ng=64),
            LinearRes(512, 128, norm="GN", ng=16),
            nn.Linear(128, 64),
        )
        return PCRFeats(name, backbone, down, "both")
    if name == "dgcnn":
        backbone = DGCNN(dropout=0.5, emb_dims=1024, k=20, output_channels=40)
        down = nn.Sequential(
            LinearRes(1024, 512, norm="GN", ng=64),
            LinearRes(512, 128, norm="GN", ng=16),
            nn.Linear(128, 64),
        )
        return PCRFeats(name, backbone, down, "both")

    raise ValueError(f"unknown PCR model '{name}' (expected point-transformer|pointnet|dgcnn)")


def find_checkpoint(name: str, extra_dirs=None) -> str:
    """Locate a pretrained checkpoint in the default + extra dirs."""
    base = PCR_MODELS[name][0]
    dirs = list(extra_dirs or []) + DEFAULT_CKPT_DIRS
    for d in dirs:
        p = os.path.join(d, base)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Pretrained ckpt '{base}' not found. Searched:\n"
        + "\n".join(f"  - {os.path.join(d, base)}" for d in dirs)
        + "\nDownload:  cd point-cloud-reid && ./tools/download_pretrained.sh"
    )


def load_pcr(name: str, ckpt_path: str, device="cpu"):
    """Build the extractor and load the checkpoint's backbone (+ downsample) weights.

    Returns: (model.eval() on `device`, report dict with missing/unexpected keys).
    Only `backbone.*` (and `downsample.*` for PointNet/DGCNN) are loaded; the matcher
    / cross-attention / classifier / shape / fp heads are intentionally ignored — we only
    consume the per-observation pooled backbone feature.
    """
    want = (name.lower() in ("pointnet", "dgcnn"))
    allow = {"backbone", "downsample" if want else "backbone"}

    model = build_pcr(name)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {k: v for k, v in sd.items() if k.split(".")[0] in allow}

    report = {"loaded": 0, "missing": [], "unexpected": []}
    new_sd = model.state_dict()
    matched = {k: sd[k] for k in new_sd if k in sd}
    model.load_state_dict(matched, strict=False)
    report["loaded"] = len(matched)
    report["missing"] = [k for k in new_sd if k not in sd]       # in model, not in ckpt
    report["unexpected"] = [k for k in new_sd if k not in matched and k in sd]

    model.eval()
    model.to(device)
    return model, report


# ────────────────────────────────────────────────────────────────────────────
# Convenience: crop batch -> embeddings (L2-normalised), same as reid_model.py output
# ────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def embed_crops(model: "PCRFeats", crops_np: np.ndarray, device="cpu", batch=32):
    """crops_np: (M, N, 3) float32 -> (M, D) float32, L2-normalised rows."""
    out = []
    dev = torch.device(device)
    for i in range(0, len(crops_np), batch):
        xb = torch.from_numpy(np.ascontiguousarray(crops_np[i:i + batch], dtype=np.float32)).to(dev)
        e = model(xb).cpu().numpy().astype(np.float32)
        n = np.linalg.norm(e, axis=1, keepdims=True)
        out.append(np.divide(e, n, out=np.zeros_like(e), where=n > 1e-8))
    return np.concatenate(out, axis=0)


if __name__ == "__main__":
    # Tiny self-test: two clearly different shapes -> different embeddings.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="point-transformer")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = args.ckpt or find_checkpoint(args.name)
    model, rep = load_pcr(args.name, ck, args.device)
    print(f"[{args.name}] ckpt={ck}")
    print(f"  param tensors loaded: {rep['loaded']}  missing: {len(rep['missing'])}  "
          f"unexpected: {len(rep['unexpected'])}")
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.3, (1, 128, 3)).astype(np.float32)
    b = (rng.normal(0, 0.3, (1, 128, 3)) + np.array([0.0, 0.0, 0.9])).astype(np.float32)
    ea, eb = embed_crops(model, a, args.device), embed_crops(model, b, args.device)
    print(f"  emb dim: {ea.shape[1]}  ||ea||={np.linalg.norm(ea[0]):.4f}  "
          f"cos(a,a)={float(ea@ea.T):.4f}  cos(a,b)={float((ea @ eb.T)[0, 0]):.4f}")
