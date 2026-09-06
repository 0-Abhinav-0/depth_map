import torch

# Any static 3D point, at any depth, seen by two frames under the SAME
# rigid camera motion must satisfy the epipolar constraint. It does not
# matter how far away the point is -- parallax alone never violates it. So
# a correspondence that violates it by more than a few pixels isn't just
# "a point at an unusual depth" -- it's evidence the point moved between
# the two frames (or the correspondence/pose is wrong). This is what lets
# us flag moving-object correspondences using only the poses we already
# recovered from COLMAP, without a separate object detector.
MAX_EPIPOLAR_ERROR_PX = 4.0


def _skew(t):
    zero = t.new_zeros(())
    return torch.stack(
        [
            torch.stack([zero, -t[2], t[1]]),
            torch.stack([t[2], zero, -t[0]]),
            torch.stack([-t[1], t[0], zero]),
        ]
    )


def essential_matrix(R_i, t_i, R_j, t_j):
    """E such that x_j_norm^T @ E @ x_i_norm ~= 0 for any static point,
    given world->cam poses (R_i,t_i) and (R_j,t_j)."""
    R_rel = R_j @ R_i.T
    t_rel = t_j - R_rel @ t_i
    return _skew(t_rel) @ R_rel


def _normalize(uv, fx, fy, cx, cy):
    x = (uv[:, 0] - cx) / fx
    y = (uv[:, 1] - cy) / fy
    return torch.stack([x, y, torch.ones_like(x)], dim=1)


def epipolar_error_px(uv_i, uv_j, E, fx_i, fy_i, cx_i, cy_i, fx_j, fy_j, cx_j, cy_j):
    """Sampson distance (first-order approximation of the true geometric
    distance to the epipolar line), converted to approximate pixel units
    for an interpretable threshold. Large values: (uv_i, uv_j) is
    inconsistent with ANY static point under the recovered camera motion."""
    x_i = _normalize(uv_i, fx_i, fy_i, cx_i, cy_i)
    x_j = _normalize(uv_j, fx_j, fy_j, cx_j, cy_j)

    Ex_i = x_i @ E.T  # row-form of E @ x_i
    Etx_j = x_j @ E  # row-form of E.T @ x_j
    numerator = (x_j * Ex_i).sum(dim=1) ** 2
    denom = Ex_i[:, 0] ** 2 + Ex_i[:, 1] ** 2 + Etx_j[:, 0] ** 2 + Etx_j[:, 1] ** 2
    sampson_sq = numerator / denom.clamp(min=1e-12)

    focal = 0.5 * (fx_i + fy_i)
    return sampson_sq.clamp(min=0).sqrt() * focal
