import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.attention_utils import lif_to_binary_attention, apply_attention_mask

class NeuroPatchClassifier(nn.Module):
    def __init__(self, num_classes=10, backbone=None):
        super().__init__()
        assert backbone is not None, "Backbone model must be provided"
        self.backbone = backbone  # 예: PatchVGGWithLIF or ResNet 기반 등

        # Fully connected FFNN after masked patch features
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        patch_feats = self.backbone(x)                            # (B, N, D)
        binary_attention = lif_to_binary_attention(patch_feats)  # (B, N)
        masked_feats = apply_attention_mask(patch_feats, binary_attention)  # (B, N, D)

        masked_feats = masked_feats.transpose(1, 2)  # (B, D, N)
        pooled = self.pool(masked_feats).squeeze(-1) # (B, D)
        logits = self.classifier(pooled)              # (B, num_classes)
        return logits