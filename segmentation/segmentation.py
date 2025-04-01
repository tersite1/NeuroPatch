import torch.nn as nn
from segmentation.segmentation_module import SegmentationHead
from backbone.attention_utils import lif_to_attention, apply_attention_weights

class NeuroPatchSegmentor(nn.Module):
    def __init__(self, num_classes=21, backbone=None):
        super().__init__()
        assert backbone is not None, "Backbone must be provided"

        self.backbone = backbone
        patch_grid = (backbone.img_size // backbone.patch_size,
                      backbone.img_size // backbone.patch_size)

        self.seg_head = SegmentationHead(
            embed_dim=backbone.embed_dim,
            num_classes=num_classes,
            patch_grid=patch_grid
        )

    def forward(self, x):
        patch_feats = self.backbone(x)  # (B, N, D)

        # 조건부 masking: PatchVGGWithLIF인 경우에만 attention mask 적용
        if self.backbone.__class__.__name__ == "PatchVGGWithLIF":
            binary_attention = lif_to_binary_attention(patch_feats)       # (B, N)
            patch_feats = apply_attention_mask(patch_feats, binary_attention)  # (B, N, D)

        seg_logits = self.seg_head(patch_feats)  # (B, C, H, W)
        return seg_logits

import torch.nn as nn

class SimpleSegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        return self.criterion(pred, target)
