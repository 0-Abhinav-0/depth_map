import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from depth_map.depth_model import DepthRefinerModel
from depth_map.export import export_depth_sequence
from depth_map.flow import compute_windowed_flow
from depth_map.frames import extract_frames
from depth_map.poses import recover_scene_geometry
from depth_map.refine import refine_depth_sequence


def main():
    parser = argparse.ArgumentParser(description="Video -> temporally consistent depth map sequence")
    parser.add_argument("video", help="input video file")
    parser.add_argument("--workdir", default="workdir")
    parser.add_argument("--fps", type=float, default=None, help="resample video to this fps before extraction")
    parser.add_argument("--scale-width", type=int, default=None, help="downscale frames to this width before extraction")
    parser.add_argument("--flow-offsets", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--export-format", choices=["png16", "exr"], default="png16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    frames_dir = workdir / "frames"
    colmap_dir = workdir / "colmap"
    depth_dir = workdir / "depth"

    print("[1/5] extracting frames...")
    frame_paths = extract_frames(args.video, frames_dir, fps=args.fps, scale_width=args.scale_width)
    print(f"  {len(frame_paths)} frames extracted to {frames_dir}")

    print("[2/5] recovering camera poses (COLMAP)...")
    geometry = recover_scene_geometry(frames_dir, colmap_dir)
    print(f"  {len(geometry)}/{len(frame_paths)} frames registered")

    print("[3/5] computing optical flow (RAFT)...")
    frame_images = [np.array(Image.open(p).convert("RGB")) for p in frame_paths]
    flow_results = compute_windowed_flow(frame_images, offsets=tuple(args.flow_offsets), device=args.device)
    print(f"  {len(flow_results)} frame pairs")

    print("[4/5] refining depth (Depth Anything V2 + geometric consistency)...")
    depth_model = DepthRefinerModel(device=args.device)
    depth_maps = refine_depth_sequence(
        depth_model,
        frame_paths,
        geometry,
        flow_results,
        num_epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )

    print("[5/5] exporting depth sequence...")
    frame_names = [p.name for p in frame_paths]
    export_depth_sequence(depth_maps, depth_dir, frame_names, fmt=args.export_format)
    print(f"  depth sequence written to {depth_dir}")


if __name__ == "__main__":
    main()
