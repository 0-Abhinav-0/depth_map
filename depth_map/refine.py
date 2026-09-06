from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from .depth_model import disparity_to_depth
from .motion import MAX_EPIPOLAR_ERROR_PX, epipolar_error_px, essential_matrix

# Pixels closer than this many pixels apart in a Huber loss are treated as
# "agree" rather than penalized further; keeps a few bad correspondences
# from dominating the gradient.
HUBER_DELTA_WORLD = 0.1
MIN_ANCHORS_FOR_CALIBRATION = 8
MIN_INLIER_CORRESPONDENCES_PER_PAIR = 16
# Only judge a frame's reliability once it has enough epipolar-checked
# samples to be statistically meaningful; a frame touched by only a
# handful of correspondences gets the benefit of the doubt.
MIN_SAMPLES_FOR_RELIABILITY_CHECK = 200
MIN_EPIPOLAR_INLIER_RATE = 0.5
# Fraction of each pair's correspondences permanently held out as a
# validation set (never used for a gradient step) to actually detect
# overfitting, rather than trusting the training loss curve alone.
VAL_FRACTION = 0.2
EARLY_STOP_PATIENCE = 5


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

    # Every flow correspondence between two POSED frames is checked against
    # the epipolar constraint implied by their recovered relative pose: any
    # STATIC point, at any depth, must satisfy it under rigid camera motion.
    # A correspondence that violates it isn't an unusual-depth static point
    # (parallax alone never breaks this constraint) -- it's evidence of a
    # moving object, or a bad correspondence/pose. Filtering these out here
    # keeps the consistency loss from being handed contradictory
    # constraints by e.g. a person moving in the foreground.
    #
    # For each surviving pair we also carve out a fixed validation subset
    # (never trained on) and a training mask that excludes those exact
    # pixels. A first version of this trained on ONE fixed batch of
    # correspondences for all epochs; the training loss converged nicely,
    # but disagreement measured on an independently-resampled set of
    # correspondences from the SAME flow fields got worse, not better --
    # classic overfitting to the fixed batch rather than learning a
    # genuinely more self-consistent depth field. Resampling fresh training
    # points every epoch (below) plus this held-out validation set (used
    # for early stopping) is the fix.
    pair_setup = {}
    inlier_counts = [0] * n
    total_counts = [0] * n
    for (i, j), (flow_ij, mask_ij) in flow_results.items():
        if poses[i] is None or poses[j] is None:
            continue
        pool = _sample_valid_correspondences(flow_ij, mask_ij, max_correspondences_per_pair * 2)
        if pool is None:
            continue
        uv_i, uv_j = pool[0].to(device), pool[1].to(device)

        R_i, t_i, fx_i, fy_i, cx_i, cy_i = poses[i]
        R_j, t_j, fx_j, fy_j, cx_j, cy_j = poses[j]
        E = essential_matrix(R_i, t_i, R_j, t_j)
        epi_error = epipolar_error_px(uv_i, uv_j, E, fx_i, fy_i, cx_i, cy_i, fx_j, fy_j, cx_j, cy_j)
        inlier = epi_error < MAX_EPIPOLAR_ERROR_PX

        total_counts[i] += inlier.numel()
        total_counts[j] += inlier.numel()
        inlier_counts[i] += inlier.sum().item()
        inlier_counts[j] += inlier.sum().item()

        uv_i, uv_j = uv_i[inlier], uv_j[inlier]
        if uv_i.shape[0] < MIN_INLIER_CORRESPONDENCES_PER_PAIR * 2:
            continue

        m = uv_i.shape[0]
        n_val = max(MIN_INLIER_CORRESPONDENCES_PER_PAIR, int(m * VAL_FRACTION))
        perm = torch.randperm(m, device=device)
        val_idx = perm[:n_val]
        val_uv_i, val_uv_j = uv_i[val_idx], uv_j[val_idx]

        train_mask = mask_ij.clone()
        val_y = val_uv_i[:, 1].round().long().cpu()
        val_x = val_uv_i[:, 0].round().long().cpu()
        train_mask[val_y, val_x] = False

        pair_setup[(i, j)] = {
            "E": E,
            "flow_ij": flow_ij,
            "train_mask": train_mask,
            "val_uv_i": val_uv_i,
            "val_uv_j": val_uv_j,
        }

    # A frame whose correspondences are mostly epipolar outliers is either
    # dominated by a moving subject or was itself badly posed by COLMAP
    # (e.g. registered via a weak fallback rather than normal PnP) --
    # either way its own anchors/pose aren't trustworthy either, so it's
    # pulled out of the anchor loss and every pair touching it, the same as
    # a frame COLMAP never registered at all.
    unreliable_frames = {
        idx
        for idx in range(n)
        if total_counts[idx] >= MIN_SAMPLES_FOR_RELIABILITY_CHECK
        and inlier_counts[idx] / total_counts[idx] < MIN_EPIPOLAR_INLIER_RATE
    }
    if unreliable_frames:
        print(
            f"[refine] flagged {len(unreliable_frames)} frame(s) as unreliable "
            f"(likely dominated by a moving subject, or a bad COLMAP pose): "
            f"{[names[idx] for idx in sorted(unreliable_frames)]}"
        )
        for idx in unreliable_frames:
            anchors[idx] = None
        pair_setup = {
            (i, j): setup for (i, j), setup in pair_setup.items()
            if i not in unreliable_frames and j not in unreliable_frames
        }

    # Frames with no anchors -- either COLMAP never registered them, or they
    # were just flagged unreliable above -- were left at the identity
    # (1.0, 0.0) affine, an arbitrary scale unrelated to the other frames'
    # COLMAP-tied scale. Left as-is, their depth values land nowhere near
    # the calibrated frames', which blows out the whole clip's dynamic
    # range once export.py normalizes against the global min/max. They
    # still don't get real geometric correction (no trustworthy pose ->
    # can't join the consistency loss), but borrowing the nearest reliable
    # frame's (scale, shift) at least puts them on the right order of
    # magnitude so they don't wreck the shared scale every other frame
    # depends on.
    calibrated_idxs = [idx for idx in range(n) if anchors[idx] is not None]
    if calibrated_idxs:
        for idx in range(n):
            if anchors[idx] is None:
                nearest = min(calibrated_idxs, key=lambda c: abs(c - idx))
                scale_shift[idx] = scale_shift[nearest]

    if not pair_setup:
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
    num_pairs = len(pair_setup)
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

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

        # Fresh random training sample every epoch (drawn from train_mask,
        # which already excludes this pair's held-out validation pixels) --
        # not the same fixed batch reused every epoch, which is what
        # previously let the network overfit to particular pixels instead
        # of learning a genuinely more self-consistent depth field.
        for (i, j), setup in pair_setup.items():
            sampled = _sample_valid_correspondences(setup["flow_ij"], setup["train_mask"], max_correspondences_per_pair)
            if sampled is None:
                continue
            uv_i, uv_j = sampled[0].to(device), sampled[1].to(device)
            epi_error = epipolar_error_px(uv_i, uv_j, setup["E"], *poses[i][2:], *poses[j][2:])
            inlier = epi_error < MAX_EPIPOLAR_ERROR_PX
            if inlier.sum().item() < MIN_INLIER_CORRESPONDENCES_PER_PAIR:
                continue
            uv_i, uv_j = uv_i[inlier], uv_j[inlier]

            disp_i = depth_model.forward_head(feature_cache[i][0], feature_cache[i][1], out_sizes[i])[0]
            disp_j = depth_model.forward_head(feature_cache[j][0], feature_cache[j][1], out_sizes[j])[0]
            scale_i, shift_i = scale_shift[i]
            scale_j, shift_j = scale_shift[j]
            R_i, t_i = poses[i][0], poses[i][1]
            R_j, t_j = poses[j][0], poses[j][1]

            depth_i_vals = disparity_to_depth(_grid_sample_at(disp_i, uv_i)[:, 0], scale_i, shift_i)
            depth_j_vals = disparity_to_depth(_grid_sample_at(disp_j, uv_j)[:, 0], scale_j, shift_j)

            p_i_cam = backproject(uv_i, depth_i_vals, *poses[i][2:])
            p_j_cam = backproject(uv_j, depth_j_vals, *poses[j][2:])
            p_i_world = (p_i_cam - t_i) @ R_i
            p_j_world = (p_j_cam - t_j) @ R_j
            term_loss = consistency_weight * huber(p_i_world, p_j_world) / num_pairs
            term_loss.backward()
            consistency_loss_total += term_loss.item()

        optimizer.step()

        # Validation pass on the permanently held-out pixels, never used for
        # a gradient step -- this is what actually detects overfitting,
        # since the training loss above will happily keep decreasing on
        # whatever it's currently being fed regardless.
        depth_model.eval_mode()
        val_loss_total = 0.0
        with torch.no_grad():
            for (i, j), setup in pair_setup.items():
                disp_i = depth_model.forward_head(feature_cache[i][0], feature_cache[i][1], out_sizes[i])[0]
                disp_j = depth_model.forward_head(feature_cache[j][0], feature_cache[j][1], out_sizes[j])[0]
                scale_i, shift_i = scale_shift[i]
                scale_j, shift_j = scale_shift[j]
                R_i, t_i = poses[i][0], poses[i][1]
                R_j, t_j = poses[j][0], poses[j][1]

                depth_i_vals = disparity_to_depth(_grid_sample_at(disp_i, setup["val_uv_i"])[:, 0], scale_i, shift_i)
                depth_j_vals = disparity_to_depth(_grid_sample_at(disp_j, setup["val_uv_j"])[:, 0], scale_j, shift_j)
                p_i_cam = backproject(setup["val_uv_i"], depth_i_vals, *poses[i][2:])
                p_j_cam = backproject(setup["val_uv_j"], depth_j_vals, *poses[j][2:])
                p_i_world = (p_i_cam - t_i) @ R_i
                p_j_world = (p_j_cam - t_j) @ R_j
                val_loss_total += huber(p_i_world, p_j_world).item()
        depth_model.train_mode()

        if val_loss_total < best_val_loss - 1e-6:
            best_val_loss = val_loss_total
            best_state = (
                {k: v.detach().clone() for k, v in depth_model.model.neck.state_dict().items()},
                {k: v.detach().clone() for k, v in depth_model.model.head.state_dict().items()},
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"[refine] epoch {epoch:4d}  anchor={anchor_loss_total:.4f}  "
            f"consistency(train)={consistency_loss_total:.4f}  consistency(val)={val_loss_total:.4f}"
        )

        if num_pairs and epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"[refine] early stopping at epoch {epoch} (no validation improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    if best_state is not None:
        depth_model.model.neck.load_state_dict(best_state[0])
        depth_model.model.head.load_state_dict(best_state[1])

    depth_model.eval_mode()
    final_depths = []
    with torch.no_grad():
        for idx in range(n):
            disp = depth_model.forward_head(feature_cache[idx][0], feature_cache[idx][1], out_sizes[idx])[0]
            scale, shift = scale_shift[idx]
            depth = disparity_to_depth(disp, scale, shift)
            final_depths.append(depth.cpu().numpy())
    return final_depths
