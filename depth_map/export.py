import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

from pathlib import Path

import cv2
import numpy as np


def export_depth_sequence(depth_maps, out_dir, frame_names, fmt="png16"):
    """depth_maps: list of (H,W) float32 numpy arrays, one per frame, already
    in a shared/consistent scale (as returned by refine.refine_depth_sequence).
    fmt: "png16" (16-bit grayscale PNG, normalized against the WHOLE video's
    min/max so the quantization itself doesn't introduce per-frame scale
    jumps) or "exr" (32-bit float, no quantization/normalization needed --
    preferred when precision matters more than universal tool support).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "exr":
        for depth, name in zip(depth_maps, frame_names):
            path = out_dir / f"{Path(name).stem}.exr"
            cv2.imwrite(str(path), depth.astype(np.float32))
        return

    if fmt != "png16":
        raise ValueError(f"unknown export format: {fmt}")

    # Percentile rather than true min/max: a single stray pixel (or an
    # uncalibrated frame that slipped through) can otherwise claim most of
    # the 16-bit range and wash out the contrast for every well-behaved
    # frame. Values outside [0.5, 99.5] just clip instead of shifting the
    # shared scale everything else is encoded against.
    all_values = np.concatenate([d.reshape(-1) for d in depth_maps])
    global_min, global_max = np.percentile(all_values, [0.5, 99.5])
    span = max(global_max - global_min, 1e-6)
    for depth, name in zip(depth_maps, frame_names):
        normalized = (depth - global_min) / span
        as_uint16 = (normalized.clip(0, 1) * 65535).astype(np.uint16)
        path = out_dir / f"{Path(name).stem}.png"
        cv2.imwrite(str(path), as_uint16)
