import numpy as np
import torch
import torch.nn.functional as F
import ptlflow
from ptlflow.utils.io_adapter import IOAdapter

# Sundaram et al. (2010) adaptive forward-backward consistency threshold:
# a correspondence is kept only if the round-trip flow error is small
# relative to the local flow magnitude. Larger ALPHA_1/ALPHA_2 keep more
# (noisier) correspondences; these defaults favor precision over recall,
# since bad correspondences corrupt the geometric loss more than a sparser
# but cleaner set does.
FB_ALPHA_1 = 0.01
FB_ALPHA_2 = 0.5


def load_raft(device="cuda"):
    model = ptlflow.get_model("raft", ckpt_path="things")
    return model.eval().to(device)


def _flow_field(model, io_adapter, img1, img2):
    inputs = io_adapter.prepare_inputs([img1, img2])
    with torch.no_grad():
        out = model(inputs)
    # (1, 1, 2, H, W) -> (2, H, W), channel 0 = dx, channel 1 = dy
    return out["flows"][0, 0]


def _warp(field, flow):
    # Bilinearly sample `field` (C,H,W) at each pixel's flow-displaced
    # location, i.e. field_warped(x) = field(x + flow(x)).
    _, h, w = field.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=flow.device, dtype=flow.dtype),
        torch.arange(w, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    sample_x = xs + flow[0]
    sample_y = ys + flow[1]
    grid_x = 2.0 * sample_x / max(w - 1, 1) - 1.0
    grid_y = 2.0 * sample_y / max(h - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    warped = F.grid_sample(
        field.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return warped[0]


def forward_backward_consistency_mask(flow_fwd, flow_bwd):
    """flow_fwd: i->j, flow_bwd: j->i, both (2,H,W). Returns a (H,W) bool
    mask over frame i marking pixels whose correspondence in j round-trips
    back close to where it started (i.e. not occluded/disoccluded in j)."""
    flow_bwd_warped = _warp(flow_bwd, flow_fwd)
    round_trip = flow_fwd + flow_bwd_warped
    round_trip_err = (round_trip ** 2).sum(dim=0)
    mag = (flow_fwd ** 2).sum(dim=0) + (flow_bwd_warped ** 2).sum(dim=0)
    threshold = FB_ALPHA_1 * mag + FB_ALPHA_2
    return round_trip_err < threshold


def compute_windowed_flow(frame_images, offsets=(1, 2, 3), device="cuda"):
    """frame_images: list of HxWx3 uint8 numpy arrays, in temporal order.
    Returns {(i, j): (flow_ij, mask_ij)} for every ordered pair (i, j=i+off)
    within `offsets` of each other, both directions computed so a
    forward-backward consistency mask can be derived.
    flow_ij is a (2,H,W) float32 CPU tensor (pixel displacement i->j).
    mask_ij is a (H,W) bool CPU tensor: True where the i->j correspondence is
    trustworthy (not occluded/disoccluded).

    Every flow field is moved to CPU as soon as it's computed. A short clip
    at moderate resolution can easily need hundreds of (2,H,W) flow fields
    (n_frames * len(offsets) * 2 directions) -- keeping all of them resident
    on a 6GB GPU alongside RAFT's own correlation-volume working memory
    reliably OOMs, since nothing here needs more than one pair's fields on
    GPU at a time.
    """
    model = load_raft(device)
    h, w = frame_images[0].shape[:2]
    io_adapter = IOAdapter(model, (h, w), cuda=(device == "cuda"))

    n = len(frame_images)
    raw_flow_cpu = {}
    results = {}
    for i in range(n):
        for off in offsets:
            j = i + off
            if j >= n:
                continue
            if (i, j) not in raw_flow_cpu:
                raw_flow_cpu[(i, j)] = _flow_field(model, io_adapter, frame_images[i], frame_images[j]).cpu()
            if (j, i) not in raw_flow_cpu:
                raw_flow_cpu[(j, i)] = _flow_field(model, io_adapter, frame_images[j], frame_images[i]).cpu()

            flow_ij_gpu = raw_flow_cpu[(i, j)].to(device)
            flow_ji_gpu = raw_flow_cpu[(j, i)].to(device)
            mask_ij = forward_backward_consistency_mask(flow_ij_gpu, flow_ji_gpu).cpu()
            results[(i, j)] = (flow_ij_gpu.cpu(), mask_ij)
            del flow_ij_gpu, flow_ji_gpu
    return results
