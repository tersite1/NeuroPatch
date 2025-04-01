import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.attention_utils import lif_to_attention, apply_attention_mask

class NeuroPatchClassifier(nn.Module):
    def __init__(self, num_classes=10, backbone=None):
        super().__init__()
        assert backbone is not None, "Backbone model must be provided"
        self.backbone = backbone

        # Dynamically calculate the embedding dimension based on backbone output.
        self.embed_dim = self._get_embedding_size(backbone)

        # Fully connected FFNN after masked patch features or direct backbone features.
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def _get_embedding_size(self, model):
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)  # Adjust input size as needed
            output = model(dummy_input)
            # 지원하는 출력 차원에 따라 embed_dim을 결정합니다.
            if output.dim() == 3:
                # backbone이 (B, C, N) 형태로 출력하면, C가 embedding dimension입니다.
                return output.size(1)
            elif output.dim() == 2:
                # backbone이 (B, D) 형태로 출력하면, D가 embedding dimension입니다.
                return output.size(1)
            else:
                raise ValueError("Unsupported backbone output dimensions: {}".format(output.dim()))

    def forward(self, x):
        backbone_out = self.backbone(x)
        if backbone_out.dim() == 3:
            # 백본 출력이 (B, C, N)인 경우,
            # transpose를 통해 (B, N, C)로 변경 후 attention 및 pooling 적용
            patch_feats = backbone_out.transpose(1, 2)  # (B, N, D)
            binary_attention = lif_to_attention(patch_feats)  # (B, N)
            masked_feats = apply_attention_mask(patch_feats, binary_attention)  # (B, N, D)
            pooled = masked_feats.mean(dim=1)  # (B, D)
            logits = self.classifier(pooled)  # (B, num_classes)
        elif backbone_out.dim() == 2:
            # 백본 출력이 (B, D)인 경우, 바로 분류기로 전달
            logits = self.classifier(backbone_out)
        else:
            raise ValueError("Unsupported backbone output dimensions: {}".format(backbone_out.dim()))
        return logits
