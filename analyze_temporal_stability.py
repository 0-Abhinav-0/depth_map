"""Compare the pipeline's geometrically-refined depth against a baseline of
independent per-frame depth (frozen DA2 + per-frame COLMAP-anchor
calibration only, no cross-frame consistency fine-tuning) -- the exact
"do NOT do this" strawman from the project's own spec. Both are computed
fresh in this script, in the same raw COLMAP-scale units (not the lossy,
independently-percentile-normalized 16-bit export), so the comparison is
apples-to-apples. Renders both as colorized videos and reports two
quantitative stability metrics:

1. Anchor-point CV: for each COLMAP sparse point visible across multiple
   frames, coefficient of variation (std/mean) of predicted depth at that
   point across the frames it appears in. Both methods calibrate directly
   against these points, so this mostly measures calibration quality, not
   the cross-frame consistency loss's marginal effect.
2. Flow-correspondence world-point disagreement: for dense optical-flow
   correspondences (epipolar-filtered, the same ones the consistency loss
   actually optimizes), the 3D distance between the two frames' backprojected
   points. This is the actual quantity the refinement targets, evaluated on
   points that are NOT explicitly anchored -- the more meaningful test.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch
from PIL import Image

from depth_map.depth_model import DepthRefinerModel, disparity_to_depth
from depth_map.flow import compute_windowed_flow
from depth_map.motion import MAX_EPIPOLAR_ERROR_PX, epipolar_error_px, essential_matrix
from depth_map.poses import geometry_from_reconstruction
from depth_map.refine import (
    _grid_sample_at,
    _sample_valid_correspondences,
    backproject,
    fit_disparity_to_depth_affine,
    refine_depth_sequence,
)


def build_baseline_depth(depth_model, frame_paths, geometry, device="cuda"):
    names = [Path(p).name for p in frame_paths]
    n = len(frame_paths)
    depth_model.eval_mode()

    scale_shift, calibrated, disparities = [], [], []
    with torch.no_grad():
        for idx in range(n):
            img = Image.open(frame_paths[idx]).convert("RGB")
            pixel_values = depth_model.preprocess(img)
            disp = depth_model.forward(pixel_values, (img.height, img.width))[0]
            geo = geometry.get(names[idx])
            if geo is None or geo.anchor_uv.shape[0] < 8:
                scale_shift.append((1.0, 0.0))
                calibrated.append(False)
            else:
                anchor_uv = torch.tensor(geo.anchor_uv, device=device)
                anchor_depth = torch.tensor(geo.anchor_depth, device=device)
                scale_shift.append(fit_disparity_to_depth_affine(disp, anchor_uv, anchor_depth))
                calibrated.append(True)
            disparities.append(disp)

    calibrated_idxs = [i for i in range(n) if calibrated[i]]
    for idx in range(n):
        if not calibrated[idx] and calibrated_idxs:
            nearest = min(calibrated_idxs, key=lambda c: abs(c - idx))
            scale_shift[idx] = scale_shift[nearest]

    final = []
    with torch.no_grad():
        for idx in range(n):
            scale, shift = scale_shift[idx]
            depth = disparity_to_depth(disparities[idx], scale, shift)
            final.append(depth.cpu().numpy())
    return final


def colorize_and_write_video(depth_maps, out_path, fps):
    all_values = np.concatenate([d.reshape(-1) for d in depth_maps])
    lo, hi = np.percentile(all_values, [0.5, 99.5])
    span = max(hi - lo, 1e-6)

    h, w = depth_maps[0].shape
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for depth in depth_maps:
        normalized = np.clip((depth - lo) / span, 0, 1)
        vis = (normalized * 255).astype(np.uint8)
        color = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
        writer.write(color)
    writer.release()


def anchor_point_cv_report(recon, frame_names, baseline_depths, refined_depths, min_track_len=5):
    name_to_idx = {name: i for i, name in enumerate(frame_names)}
    baseline_cvs, refined_cvs = [], []

    for point3D in recon.points3D.values():
        obs = []
        for track_el in point3D.track.elements:
            image = recon.image(track_el.image_id)
            if image.name not in name_to_idx:
                continue
            px, py = image.point2D(track_el.point2D_idx).xy
            obs.append((name_to_idx[image.name], int(round(px)), int(round(py))))
        if len(obs) < min_track_len:
            continue

        h, w = baseline_depths[0].shape
        b_vals, r_vals = [], []
        for idx, x, y in obs:
            if 0 <= y < h and 0 <= x < w:
                b_vals.append(baseline_depths[idx][y, x])
                r_vals.append(refined_depths[idx][y, x])
        if len(b_vals) < min_track_len:
            continue
        b_vals, r_vals = np.array(b_vals), np.array(r_vals)
        if b_vals.mean() > 0:
            baseline_cvs.append(b_vals.std() / b_vals.mean())
        if r_vals.mean() > 0:
            refined_cvs.append(r_vals.std() / r_vals.mean())

    print(f"\n--- Anchor-point CV (measures calibration; both methods anchor here) ---")
    print(f"Static points tracked across >= {min_track_len} frames: {len(baseline_cvs)}")
    print(f"Baseline median CV: {np.median(baseline_cvs):.4f}")
    print(f"Refined  median CV: {np.median(refined_cvs):.4f}")


def flow_correspondence_disagreement_report(geometry, frame_names, flow_results, baseline_depths, refined_depths, device="cuda"):
    poses = []
    for name in frame_names:
        geo = geometry.get(name)
        if geo is None:
            poses.append(None)
            continue
        R = torch.tensor(geo.R, dtype=torch.float32, device=device)
        t = torch.tensor(geo.t, dtype=torch.float32, device=device)
        poses.append((R, t, geo.fx, geo.fy, geo.cx, geo.cy))

    baseline_dists, refined_dists = [], []
    for (i, j), (flow_ij, mask_ij) in flow_results.items():
        if poses[i] is None or poses[j] is None:
            continue
        sampled = _sample_valid_correspondences(flow_ij, mask_ij, 2048)
        if sampled is None:
            continue
        uv_i, uv_j = sampled[0].to(device), sampled[1].to(device)

        R_i, t_i, fx_i, fy_i, cx_i, cy_i = poses[i]
        R_j, t_j, fx_j, fy_j, cx_j, cy_j = poses[j]
        E = essential_matrix(R_i, t_i, R_j, t_j)
        epi_error = epipolar_error_px(uv_i, uv_j, E, fx_i, fy_i, cx_i, cy_i, fx_j, fy_j, cx_j, cy_j)
        inlier = epi_error < MAX_EPIPOLAR_ERROR_PX
        if inlier.sum().item() < 16:
            continue
        uv_i, uv_j = uv_i[inlier], uv_j[inlier]

        for depths, dists in [(baseline_depths, baseline_dists), (refined_depths, refined_dists)]:
            depth_i = torch.tensor(depths[i], device=device)
            depth_j = torch.tensor(depths[j], device=device)
            d_i = _grid_sample_at(depth_i, uv_i)[:, 0]
            d_j = _grid_sample_at(depth_j, uv_j)[:, 0]
            p_i_cam = backproject(uv_i, d_i, fx_i, fy_i, cx_i, cy_i)
            p_j_cam = backproject(uv_j, d_j, fx_j, fy_j, cx_j, cy_j)
            p_i_world = (p_i_cam - t_i) @ R_i
            p_j_world = (p_j_cam - t_j) @ R_j
            dist = (p_i_world - p_j_world).norm(dim=1)
            dists.append(dist.cpu().numpy())

    baseline_dists = np.concatenate(baseline_dists)
    refined_dists = np.concatenate(refined_dists)
    print(f"\n--- Flow-correspondence world-point disagreement (the consistency loss's actual target) ---")
    print(f"Correspondences evaluated: {len(baseline_dists)}")
    print(f"Baseline median 3D disagreement: {np.median(baseline_dists):.4f}")
    print(f"Refined  median 3D disagreement: {np.median(refined_dists):.4f}")
    improvement = 1 - np.median(refined_dists) / np.median(baseline_dists)
    print(f"Relative improvement: {improvement * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="workdir_test")
    parser.add_argument("--recon", default="1", help="which sparse/<n> component to use")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    frames_dir = workdir / "frames"
    recon_path = workdir / "colmap" / "sparse" / args.recon

    recon = pycolmap.Reconstruction(str(recon_path))
    geometry = geometry_from_reconstruction(recon)

    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    frame_names = [p.name for p in frame_paths]

    print(f"Loaded reconstruction: {recon.num_reg_images()} registered / {len(frame_paths)} frames")

    print("Recomputing optical flow (needed for both the refinement and the evaluation)...")
    frame_images = [np.array(Image.open(p).convert("RGB")) for p in frame_paths]
    flow_results = compute_windowed_flow(frame_images, offsets=(1, 2, 3), device=args.device)

    depth_model = DepthRefinerModel(device=args.device)

    print("Building baseline (frozen DA2 + per-frame calibration, no refinement)...")
    baseline_depths = build_baseline_depth(depth_model, frame_paths, geometry, device=args.device)

    print(f"Running the actual refinement ({args.epochs} epochs) on the same model...")
    refined_depths = refine_depth_sequence(
        depth_model, frame_paths, geometry, flow_results, num_epochs=args.epochs, device=args.device
    )

    print("Rendering comparison videos...")
    colorize_and_write_video(baseline_depths, workdir / "baseline_depth.mp4", args.fps)
    colorize_and_write_video(refined_depths, workdir / "refined_depth.mp4", args.fps)

    anchor_point_cv_report(recon, frame_names, baseline_depths, refined_depths)
    flow_correspondence_disagreement_report(geometry, frame_names, flow_results, baseline_depths, refined_depths, device=args.device)


if __name__ == "__main__":
    main()
