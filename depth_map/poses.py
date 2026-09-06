from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap

# Reprojection error (px) above which a COLMAP sparse point is dropped as an
# anchor for disparity-to-depth calibration in refine.py.
MAX_ANCHOR_REPROJ_ERROR = 4.0


@dataclass
class FrameGeometry:
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R: np.ndarray  # (3,3) world -> camera rotation
    t: np.ndarray  # (3,) world -> camera translation
    anchor_uv: np.ndarray  # (K,2) pixel coords with a triangulated 3D point
    anchor_depth: np.ndarray  # (K,) camera-space depth (z) at those pixels


def recover_scene_geometry(image_dir, workspace_dir, sequential_overlap=10):
    """Run COLMAP SfM over the extracted frames and return per-frame intrinsics,
    pose, and sparse anchor depths. Frames COLMAP fails to register (e.g. from
    severe motion blur or textureless content) are simply absent from the
    returned dict — callers must handle missing frames.
    """
    image_dir = Path(image_dir)
    workspace_dir = Path(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    db_path = workspace_dir / "database.db"
    if db_path.exists():
        db_path.unlink()

    # Force an undistorted pinhole camera model so downstream flow/depth/refine
    # math never has to deal with radial distortion.
    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = "PINHOLE"
    pycolmap.extract_features(db_path, image_dir, reader_options=reader_options)

    pairing_options = pycolmap.SequentialPairingOptions()
    pairing_options.overlap = sequential_overlap
    pycolmap.match_sequential(db_path, pairing_options=pairing_options)

    sparse_dir = workspace_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    reconstructions = pycolmap.incremental_mapping(db_path, image_dir, sparse_dir)
    if not reconstructions:
        raise RuntimeError(
            "COLMAP could not reconstruct a scene from these frames "
            "(insufficient texture, parallax, or overlap between frames)"
        )
    # Video should yield one connected reconstruction; if COLMAP split it into
    # multiple disjoint components, keep the one covering the most frames.
    recon = max(reconstructions.values(), key=lambda r: r.num_reg_images())
    return geometry_from_reconstruction(recon)


def geometry_from_reconstruction(recon):
    """Extract the same per-frame FrameGeometry dict `recover_scene_geometry`
    returns, from an already-built pycolmap.Reconstruction. Split out so a
    saved reconstruction (e.g. workdir/colmap/sparse/<n>) can be reloaded and
    reused without rerunning SfM -- useful for analysis/comparison scripts."""
    geometry = {}
    for image in recon.images.values():
        if not image.has_pose:
            continue
        camera = recon.camera(image.camera_id)
        pose = image.cam_from_world()

        anchor_uv, anchor_depth = [], []
        for p2d in image.points2D:
            if not p2d.has_point3D():
                continue
            point3D = recon.point3D(p2d.point3D_id)
            if point3D.error > MAX_ANCHOR_REPROJ_ERROR:
                continue
            cam_pt = pose * point3D.xyz
            if cam_pt[2] <= 1e-4:
                continue
            anchor_uv.append(p2d.xy)
            anchor_depth.append(cam_pt[2])

        geometry[image.name] = FrameGeometry(
            name=image.name,
            width=camera.width,
            height=camera.height,
            fx=camera.focal_length_x,
            fy=camera.focal_length_y,
            cx=camera.principal_point_x,
            cy=camera.principal_point_y,
            R=pose.rotation.matrix(),
            t=np.asarray(pose.translation),
            anchor_uv=np.asarray(anchor_uv, dtype=np.float32).reshape(-1, 2),
            anchor_depth=np.asarray(anchor_depth, dtype=np.float32).reshape(-1),
        )
    return geometry
