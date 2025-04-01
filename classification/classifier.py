import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.vit_patch_vgg_lif import PatchVGGWithLIF
from backbone.attention_utils import lif_to_binary_attention, apply_attention_mask

class NeuroPatchClassifier(nn.Module):
    def __init__(self, num_classes=10, img_size=224, patch_size=16):
        super().__init__()
        # 1) 백본
        self.backbone = PatchVGGWithLIF(img_size=img_size, patch_size=patch_size)

        # 2) 최종 분류기
        #    (B, N, D) → (B, D) [global pooling] → (B, num_classes)
        self.pool = nn.AdaptiveAvgPool1d(1)  # patch dimension pooling
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.embed_dim, num_classes)
        )

    def forward(self, x):
        # 1) backbone → (B, N, D)
        patch_feats = self.backbone(x)

        # 2) LIF-based attention
        binary_attention = lif_to_binary_attention(patch_feats) 
        masked_feats = apply_attention_mask(patch_feats, binary_attention)  # (B, N, D)

        # 3) pooling: (B, N, D) → (B, D)
        #    transpose => (B, D, N) → adaptive pool => (B, D, 1) => squeeze => (B, D)
        masked_feats = masked_feats.transpose(1,2)   # (B, D, N)
        pooled = self.pool(masked_feats)             # (B, D, 1)
        pooled = pooled.squeeze(-1)                  # (B, D)

        # 4) classification
        logits = self.classifier(pooled)             # (B, num_classes)
        return logits
