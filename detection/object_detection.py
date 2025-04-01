# object_detection.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.vit_patch_vgg_lif import PatchVGGWithLIF
from detection.object_detection_module import DetectionHead
from backbone.attention_utils import lif_to_attention, apply_attention_weights

class NeuroPatchDetector(nn.Module):
    def __init__(self, num_classes=10, img_size=224, patch_size=16):
        super().__init__()
        self.backbone = PatchVGGWithLIF(img_size=img_size, patch_size=patch_size)
        self.det_head = DetectionHead(embed_dim=self.backbone.embed_dim, num_classes=num_classes)
        # 손실은 여기에 안 둬도 됨 (외부에서 SimpleDetectionLoss 사용 예정)

    def forward(self, x):
        """
        x: (B, 3, H, W)
        Return: (cls_logits, bboxes)
        """
        # 1) Backbone => (B, N, D)
        patch_feats = self.backbone(x)

        # 2) LIF-based attention
        bin_attn = lif_to_binary_attention(patch_feats)  # (B, N)
        masked_feats = apply_attention_mask(patch_feats, bin_attn)  # (B, N, D)

        # 3) detection head => (cls_logits, bboxes) only
        cls_logits, bboxes = self.det_head(masked_feats)
        return cls_logits, bboxes

class SimpleDetectionLoss(nn.Module):
    def __init__(self, cls_weight=1.0, bbox_weight=1.0):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss()
        self.bbox_loss = nn.SmoothL1Loss()
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight

    def forward(self, cls_pred, bbox_pred, cls_target, bbox_target):
        """
        cls_pred: (B, Q, C) or flatten needed
        bbox_pred: (B, Q, 4)
        cls_target: (B, Q) or (B,) if Q=1
        bbox_target: (B, Q, 4)
        """
        # flatten if needed
        B, Q, C = cls_pred.shape
        cls_pred = cls_pred.view(-1, C)         # (B*Q, C)
        cls_target = cls_target.view(-1)        # (B*Q)

        c_loss = self.cls_loss(cls_pred, cls_target)

        # do similarly for bbox if needed
        # e.g. bbox_pred = bbox_pred.view(-1, 4)
        # bbox_target = bbox_target.view(-1, 4)
        b_loss = self.bbox_loss(bbox_pred, bbox_target)

        return self.cls_weight * c_loss + self.bbox_weight * b_loss
