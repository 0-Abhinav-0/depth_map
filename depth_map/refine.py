from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from .depth_model import disparity_to_depth

# Pixels closer than this many pixels apart in a Huber loss are treated as
# "agree" rather than penalized further; keeps a few bad correspondences
# from dominating the gradient.
HUBER_DELTA_WORLD = 0.1
MIN_ANCHORS_FOR_CALIBRATION = 8


def _grid_sample_at(field, uv):
    """field: (H,W); uv: (N,2) pixel coords (x,y), float. Returns (N,1)."""
    field = field.unsqueeze(0)
    _, h, w = field.shape
    grid_x = 2.0 * uv[:, 0] / max(w - 1, 1) - 1.0
    grid_y = 2.0 * uv[:, 1] / max(h - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        field.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return sampled[0, :, :, 0].T  # (N,1)


def backproject(uv, depth, fx, fy, cx, cy):
    x = (uv[:, 0] - cx) / fx * depth
    y = (uv[:, 1] - cy) / fy * depth
    return torch.stack([x, y, depth], dim=-1)


def fit_disparity_to_depth_affine(disparity, anchor_uv, anchor_depth):
    """Closed-form least squares: find (scale, shift) so that
    scale*disparity + shift ~= 1/depth at COLMAP's sparse anchor points.
    This ties DA2's arbitrary per-frame disparity scale to COLMAP's
    reconstruction scale before any fine-tuning happens."""
    disp_at_anchors = _grid_sample_at(disparity, anchor_uv)[:, 0]
    target = 1.0 / anchor_depth.clamp(min=1e-6)
    A = torch.stack([disp_at_anchors, torch.ones_like(disp_at_anchors)], dim=1)
    solution = torch.linalg.lstsq(A, target.unsqueeze(1)).solution
    scale, shift = solution[0, 0].item(), solution[1, 0].item()
    if scale <= 0:
        # Degenerate anchor set (e.g. near-collinear disparities); fall back
        # to identity affine and let the anchor loss pull it into shape
        # during fine-tuning instead.
        scale, shift = 1.0, 0.0
    return scale, shift


def _sample_valid_correspondences(flow_ij, mask_ij, max_samples):
    ys, xs = torch.nonzero(mask_ij, as_tuple=True)
    if ys.numel() == 0:
        return None
    if ys.numel() > max_samples:
        sel = torch.randperm(ys.numel(), device=ys.device)[:max_samples]
        ys, xs = ys[sel], xs[sel]
    uv_i = torch.stack([xs.float(), ys.float()], dim=1)
    uv_j = uv_i + flow_ij[:, ys, xs].T
    return uv_i, uv_j


def refine_depth_sequence(
    depth_model,
    frame_paths,
    geometry,
    flow_results,
    num_epochs=100,
    lr=1e-4,
    anchor_weight=1.0,
    consistency_weight=1.0,
    max_correspondences_per_pair=2048,
    device="cuda",
):
    """Test-time fine-tune `depth_model`'s neck+head, per video, so that:
      - each frame's calibrated depth agrees with COLMAP's sparse 3D points
        (anchor_loss -- keeps absolute scale fixed, prevents collapse)
      - for every optical-flow correspondence between nearby frames, the two
        frames' calibrated depth backproject to the SAME 3D world point
        (consistency_loss -- this is the actual cross-frame geometric
        constraint; it is what removes flicker/drift, not smoothing)
    Returns a list of (H,W) numpy depth maps, one per input frame, in the
    same units/scale as the COLMAP reconstruction.
    """
    names = [Path(p).name for p in frame_paths]
    n = len(frame_paths)

    out_sizes, feature_cache = [], []
    for p in frame_paths:
        img = Image.open(p).convert("RGB")
        out_sizes.append((img.height, img.width))
        pixel_values = depth_model.preprocess(img)
        feature_cache.append(depth_model.precompute_backbone_features(pixel_values))

    depth_model.eval_mode()
    scale_shift, anchors, poses = [], [], []
    with torch.no_grad():
        for idx in range(n):
            fmaps, patch_hw = feature_cache[idx]
            disp = depth_model.forward_head(fmaps, patch_hw, out_sizes[idx])[0]
            geo = geometry.get(names[idx])
            if geo is None or geo.anchor_uv.shape[0] < MIN_ANCHORS_FOR_CALIBRATION:
                scale_shift.append((1.0, 0.0))
                anchors.append(None)
                poses.append(None)
                continue
            anchor_uv = torch.tensor(geo.anchor_uv, device=device)
            anchor_depth = torch.tensor(geo.anchor_depth, device=device)
            scale_shift.append(fit_disparity_to_depth_affine(disp, anchor_uv, anchor_depth))
            anchors.append((anchor_uv, anchor_depth))
            R = torch.tensor(geo.R, dtype=torch.float32, device=device)
            t = torch.tensor(geo.t, dtype=torch.float32, device=device)
            poses.append((R, t, geo.fx, geo.fy, geo.cx, geo.cy))

    # Frames COLMAP couldn't register have no anchors, so they were left at
    # the identity (1.0, 0.0) affine above -- an arbitrary, uncalibrated
    # scale completely unrelated to the other frames' COLMAP-tied scale.
    # Left as-is, those frames' depth values land nowhere near the
    # calibrated frames', which blows out the whole clip's dynamic range
    # once export.py normalizes against the global min/max. They still
    # don't get real geometric correction (no pose -> can't join the
    # consistency loss), but borrowing the nearest registered frame's
    # (scale, shift) at least puts them on the right order of magnitude so
    # they don't wreck the shared scale every other frame depends on.
    calibrated_idxs = [idx for idx in range(n) if anchors[idx] is not None]
    if calibrated_idxs:
        for idx in range(n):
            if anchors[idx] is None:
                nearest = min(calibrated_idxs, key=lambda c: abs(c - idx))
                scale_shift[idx] = scale_shift[nearest]

    correspondences = []
    for (i, j), (flow_ij, mask_ij) in flow_results.items():
        if poses[i] is None or poses[j] is None:
            continue
        sampled = _sample_valid_correspondences(flow_ij, mask_ij, max_correspondences_per_pair)
        if sampled is None:
            continue
        uv_i, uv_j = sampled
        correspondences.append((i, j, uv_i.to(device), uv_j.to(device)))

    if not correspondences:
        print(
            "[refine] WARNING: no usable cross-frame correspondences "
            "(COLMAP registered too few frames, or flow masks rejected everything) "
            "-- falling back to per-frame calibrated depth with no geometric fine-tuning."
        )
    if not any(a is not None for a in anchors):
        print(
            "[refine] WARNING: no frame had enough COLMAP anchors for scale calibration "
            "-- depth scale/shift left at identity (1.0, 0.0), likely visibly wrong."
        )

    depth_model.train_mode()
    optimizer = torch.optim.Adam(depth_model.trainable_parameters(), lr=lr)
    huber = torch.nn.HuberLoss(delta=HUBER_DELTA_WORLD)

    num_anchor_terms = sum(1 for a in anchors if a is not None)
    num_correspondence_terms = len(correspondences)

    # Every term is backward()'d individually and its (tiny, single-frame or
    # single-pair) graph freed immediately, instead of keeping all n frames'
    # graphs alive for one combined backward() at epoch end. Peak memory this
    # way is independent of clip length; keeping every frame's graph resident
    # simultaneously (the more obvious full-batch formulation) scales
    # linearly with frame count and reliably OOMs a 6GB GPU past a couple
    # dozen frames. Each term is pre-divided by its group's total count so
    # the accumulated gradient matches what one combined mean-loss backward
    # would have produced.
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        anchor_loss_total = 0.0
        consistency_loss_total = 0.0

        for idx in range(n):
            if anchors[idx] is None:
                continue
            disp = depth_model.forward_head(feature_cache[idx][0], feature_cache[idx][1], out_sizes[idx])[0]
            anchor_uv, anchor_depth = anchors[idx]
            scale, shift = scale_shift[idx]
            disp_at_anchor = _grid_sample_at(disp, anchor_uv)[:, 0]
            depth_at_anchor = disparity_to_depth(disp_at_anchor, scale, shift)
            term_loss = anchor_weight * huber(depth_at_anchor, anchor_depth) / num_anchor_terms
            term_loss.backward()
            anchor_loss_total += term_loss.item()

        for i, j, uv_i, uv_j in correspondences:
            disp_i = depth_model.forward_head(feature_cache[i][0], feature_cache[i][1], out_sizes[i])[0]
            disp_j = depth_model.forward_head(feature_cache[j][0], feature_cache[j][1], out_sizes[j])[0]
            scale_i, shift_i = scale_shift[i]
            scale_j, shift_j = scale_shift[j]
            R_i, t_i, fx_i, fy_i, cx_i, cy_i = poses[i]
            R_j, t_j, fx_j, fy_j, cx_j, cy_j = poses[j]

            depth_i_vals = disparity_to_depth(_grid_sample_at(disp_i, uv_i)[:, 0], scale_i, shift_i)
            depth_j_vals = disparity_to_depth(_grid_sample_at(disp_j, uv_j)[:, 0], scale_j, shift_j)

            p_i_cam = backproject(uv_i, depth_i_vals, fx_i, fy_i, cx_i, cy_i)
            p_j_cam = backproject(uv_j, depth_j_vals, fx_j, fy_j, cx_j, cy_j)
            p_i_world = (p_i_cam - t_i) @ R_i
            p_j_world = (p_j_cam - t_j) @ R_j
            term_loss = consistency_weight * huber(p_i_world, p_j_world) / num_correspondence_terms
            term_loss.backward()
            consistency_loss_total += term_loss.item()

        optimizer.step()

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(
                f"[refine] epoch {epoch:4d}  anchor={anchor_loss_total:.4f}  "
                f"consistency={consistency_loss_total:.4f}"
            )

    depth_model.eval_mode()
    final_depths = []
    with torch.no_grad():
        for idx in range(n):
            disp = depth_model.forward_head(feature_cache[idx][0], feature_cache[idx][1], out_sizes[idx])[0]
            scale, shift = scale_shift[idx]
            depth = disparity_to_depth(disp, scale, shift)
            final_depths.append(depth.cpu().numpy())
    return final_depths
