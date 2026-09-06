import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

CHECKPOINT = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthRefinerModel:
    """Wraps Depth Anything V2 with its ViT backbone frozen and only the
    DPT-style neck (multi-scale feature fusion) + head (final conv layers)
    left trainable. The backbone is the expensive, general-purpose feature
    extractor; the neck+head is what actually turns those features into a
    per-pixel disparity map, so backprop-ing the geometric consistency loss
    into just the neck+head can still correct real per-pixel shape errors
    (not just a uniform per-frame scale/shift) at a fraction of the compute
    and VRAM cost of fine-tuning the whole network.

    Note: the raw model output is *disparity* (larger value = closer),
    per Depth Anything's `depth_estimation_type: relative` training convention
    -- not depth. Callers must convert via `disparity_to_depth` before doing
    any 3D geometry with it.
    """

    def __init__(self, checkpoint=CHECKPOINT, device="cuda"):
        self.device = device
        self.model = AutoModelForDepthEstimation.from_pretrained(checkpoint).to(device)
        self.processor = AutoImageProcessor.from_pretrained(checkpoint)
        self.patch_size = self.model.config.patch_size

        for p in self.model.backbone.parameters():
            p.requires_grad_(False)
        self.model.backbone.eval()

    def trainable_parameters(self):
        return list(self.model.neck.parameters()) + list(self.model.head.parameters())

    def train_mode(self):
        self.model.neck.train()
        self.model.head.train()

    def eval_mode(self):
        self.model.neck.eval()
        self.model.head.eval()

    def preprocess(self, pil_image):
        return self.processor(images=pil_image, return_tensors="pt")["pixel_values"].to(self.device)

    def precompute_backbone_features(self, pixel_values):
        """Run the frozen backbone once; cache the result on CPU. Call this
        upfront for every frame so the optimization loop only ever runs the
        cheap trainable neck+head forward, not the full ViT, per step."""
        with torch.no_grad():
            outputs = self.model.backbone.forward_with_filtered_kwargs(pixel_values)
        _, _, h, w = pixel_values.shape
        patch_hw = (h // self.patch_size, w // self.patch_size)
        feature_maps = [f.detach().cpu() for f in outputs.feature_maps]
        return feature_maps, patch_hw

    def forward_head(self, feature_maps, patch_hw, out_size):
        """Trainable forward: cached backbone features -> disparity map,
        resized (differentiably) to `out_size` = (height, width) of the
        original frame."""
        feature_maps = [f.to(self.device) for f in feature_maps]
        patch_h, patch_w = patch_hw
        hidden_states = self.model.neck(feature_maps, patch_h, patch_w)
        disparity = self.model.head(hidden_states, patch_h, patch_w)  # (1, h, w)
        disparity = F.interpolate(
            disparity.unsqueeze(1), size=out_size, mode="bilinear", align_corners=False
        )
        return disparity[:, 0]  # (1, H, W)

    def forward(self, pixel_values, out_size):
        feature_maps, patch_hw = self.precompute_backbone_features(pixel_values)
        return self.forward_head(feature_maps, patch_hw, out_size)


def disparity_to_depth(disparity, scale, shift, eps=1e-6):
    """depth = 1 / (scale * disparity + shift), the standard affine-invariant
    inverse-depth relationship used by MiDaS-family relative depth models.
    scale/shift are fit per-video against COLMAP's sparse anchor depths
    (see refine.py) to tie DA2's arbitrary disparity scale to COLMAP's
    reconstruction scale."""
    denom = scale * disparity + shift
    return 1.0 / denom.clamp(min=eps)
