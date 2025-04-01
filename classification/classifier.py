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

        # Classification Head: Gated Linear Unit (GLU) for feature selection and classification.
        self.classifier = nn.Sequential(
            GatedLinearUnit(self.embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            GatedLinearUnit(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def _get_embedding_size(self, model):
        """Automatically determine the embedding dimension based on backbone output."""
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)  # Adjust input size as needed
            output = model(dummy_input)
            if output.dim() == 4:  # Example: (B, C, H, W)
                return output.size(1) * output.size(2) * output.size(3)  # Flatten dimensions
            elif output.dim() == 3:  # Example: (B, C, N)
                return output.size(-1)
            elif output.dim() == 2:  # Example: (B, D)
                return output.size(-1)
            else:
                raise ValueError("Unsupported backbone output dimensions: {}".format(output.dim()))

    def attention_weighted_pooling(self, masked_feats, attn_scores):
        """Attention-weighted pooling to emphasize important features."""
        weighted_feats = masked_feats * attn_scores.unsqueeze(-1)  # (B, N, D)
        pooled_feats = weighted_feats.sum(dim=1) / attn_scores.sum(dim=1).unsqueeze(-1)  # (B, D)
        return pooled_feats

    def forward(self, x):
        """Forward pass through the classifier."""
        backbone_out = self.backbone(x)

        if backbone_out.dim() == 3:
            # Backbone outputs patch-wise features: (B, C, N)
            patch_feats = backbone_out.transpose(1, 2)  # Transpose to (B, N, C)
            binary_attention = lif_to_attention(patch_feats)  # Compute attention scores: (B, N)
            masked_feats = apply_attention_mask(patch_feats, binary_attention)  # Apply attention mask: (B, N, D)

            # Use attention-weighted pooling instead of mean pooling
            pooled = self.attention_weighted_pooling(masked_feats, binary_attention)  # (B, D)

            logits = self.classifier(pooled)  # Pass pooled features to classifier head: (B, num_classes)

        elif backbone_out.dim() == 2:
            # Backbone outputs global features directly: (B, D)
            logits = self.classifier(backbone_out)

        else:
            raise ValueError("Unsupported backbone output dimensions: {}".format(backbone_out.dim()))

        return logits


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit for feature selection and activation control."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x) * torch.sigmoid(self.gate(x))
