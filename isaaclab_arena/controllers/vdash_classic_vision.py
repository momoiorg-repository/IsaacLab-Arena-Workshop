# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Classic-vision peg localisation for the rule-based baseline (brief §3 M8).

Estimates the peg grasp point from a fixed top-down RGB+depth camera using **colour segmentation +
depth back-projection** — the perception stand-in for the privileged ground-truth read that
``ScriptedPick`` normally uses. Per brief §2.1 the "rule-based 単体" baseline's grasp phase may NOT
use the simulator's ground-truth peg pose; it must come from this module.

This is deliberately a *naive* classic pipeline (a fixed blue colour threshold), so it works at L0/L1
but degrades under L2 lighting and L3 texture/distractors — that degradation is exactly the E3
comparison point. The socket pose and the table plane remain known fixtures (allowed, §2.1)."""

from __future__ import annotations

import torch


def segment_blue(rgb: torch.Tensor) -> torch.Tensor:
    """Boolean mask (N,H,W) of saturated-blue pixels (the peg's M8 visual material)."""
    rgb = rgb[..., :3].float()
    if float(rgb.max()) <= 1.5:  # normalized [0,1] -> [0,255]
        rgb = rgb * 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (b > 90.0) & (b > r + 30.0) & (b > g + 30.0)


class ClassicVisionGrasp:
    """Top-down colour-segmentation peg grasp-point estimator.

    The camera looks straight down, so a masked centroid pixel projects to a world xy by a ray cast to
    the known grip-cube-top plane (``z_visible``) — robust pinhole back-projection that needs only the
    camera pose + intrinsics (proprioceptive/known) and the nominal grip geometry, no peg ground truth
    and no depth-convention guesswork. The grasp z is the (known) grip-cube centre height."""

    def __init__(self, cam_name: str, z_visible: float = 0.067, grasp_z: float = 0.058,
                 min_pixels: int = 6, y_sign: float = -1.0, cal_xy: tuple[float, float] = (0.003, -0.001)):
        self.cam_name = cam_name
        self.z_visible = z_visible  # world z of the grip-cube top (the surface the camera rays hit)
        self.grasp_z = grasp_z      # world z of the grip-cube centre (where the pick closes)
        self.min_pixels = min_pixels
        self.y_sign = y_sign        # image-row → world-y sign (top-down opengl camera)
        # extrinsic calibration: corrects the residual systematic projection offset measured against a
        # known peg pose (a standard camera-calibration step, not a per-frame ground-truth read).
        self.cal_xy = cal_xy

    def estimate(self, env) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (grasp_point_w (N,3), valid (N,) bool)."""
        cam = env.unwrapped.scene[self.cam_name]
        d = cam.data
        rgb = d.output["rgb"]
        K, pos = d.intrinsic_matrices, d.pos_w  # (N,3,3), (N,3) world
        n, h, w = rgb.shape[0], rgb.shape[1], rgb.shape[2]
        dev = rgb.device

        mask = segment_blue(rgb)  # (N,H,W)
        vv, uu = torch.meshgrid(
            torch.arange(h, device=dev, dtype=torch.float32),
            torch.arange(w, device=dev, dtype=torch.float32),
            indexing="ij",
        )
        out = torch.zeros(n, 3, device=dev)
        valid = torch.zeros(n, dtype=torch.bool, device=dev)
        for i in range(n):
            m = mask[i]
            if int(m.sum()) < self.min_pixels:
                continue
            u = uu[m].mean()
            v = vv[m].mean()
            fx, fy, cx, cy = K[i, 0, 0], K[i, 1, 1], K[i, 0, 2], K[i, 1, 2]
            height = pos[i, 2] - self.z_visible  # camera height above the grip top
            out[i, 0] = pos[i, 0] + (u - cx) * height / fx + self.cal_xy[0]
            out[i, 1] = pos[i, 1] + self.y_sign * (v - cy) * height / fy + self.cal_xy[1]
            out[i, 2] = self.grasp_z
            valid[i] = True
        return out, valid
