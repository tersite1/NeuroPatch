import torch.nn as nn
from backbone.vit_patch_vgg_lif import PatchVGGWithLIF
from segmentation.segmentation_module import SegmentationHead
from backbone.attention_utils import lif_to_binary_attention, apply_attention_mask

class NeuroPatchSegmentor(nn.Module):
    def __init__(self, num_classes=21, img_size=224, patch_size=16):
        super().__init__()
        self.backbone = PatchVGGWithLIF(img_size=img_size, patch_size=patch_size)
        self.seg_head = SegmentationHead(embed_dim=self.backbone.embed_dim, num_classes=num_classes, patch_grid=(img_size // patch_size, img_size // patch_size))

    def forward(self, x):
        patch_feats = self.backbone(x)                              # (B, N, D)
        binary_attention = lif_to_binary_attention(patch_feats)     # (B, N)
        masked_feats = apply_attention_mask(patch_feats, binary_attention)  # (B, N, D)
        seg_logits = self.seg_head(masked_feats)                    # (B, C, H, W)
        return seg_logits
    



import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSegmentationLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred, target):
        """
        pred: (B, C, H, W)
        target: (B, H, W) or (B, 1, H, W)
        """
        if target.ndim == 4:
            target = target.squeeze(1)  # (B, H, W)
        return self.loss_fn(pred, target)
