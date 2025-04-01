# attention_utils.py

import torch
import torch.nn.functional as F


def lif_to_binary_attention(lif_outputs, threshold=0.5):
    """
    lif_outputs: Tensor of shape (B, N, D) - output from PatchVGGWithLIF
    threshold: scalar - threshold above which attention is active (spike)

    Returns:
        binary_attention: Tensor of shape (B, N), where each value is 0 or 1
    """
    # Reduce over embedding dimension to get scalar per patch
    patch_scores = lif_outputs.mean(dim=-1)  # (B, N)
    binary_attention = (patch_scores > threshold).float()  # (B, N)
    return binary_attention


def apply_attention_mask(features, binary_attention):
    """
    features: Tensor of shape (B, N, D)
    binary_attention: Tensor of shape (B, N)

    Returns:
        masked_features: Tensor of shape (B, N, D) where inactive patches are zeroed out
    """
    B, N, D = features.shape
    binary_attention = binary_attention.unsqueeze(-1)  # (B, N, 1)
    masked_features = features * binary_attention  # broadcasting
    return masked_features
