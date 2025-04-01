# attention_utils.py

import torch
import torch.nn.functional as F


def lif_to_attention(lif_outputs, mode='binary', threshold=0.5, temperature=1.0):
    """
    Convert LIF outputs to attention scores.

    Args:
        lif_outputs (Tensor): (B, N, D), output from PatchVGGWithLIF
        mode (str): 'binary', 'soft', or 'none'
        threshold (float): threshold value for binary mode
        temperature (float): scaling factor for soft mode

    Returns:
        attention_scores (Tensor): (B, N)
    """
    patch_scores = lif_outputs.mean(dim=-1)  # (B, N)

    if mode == 'binary':
        return (patch_scores > threshold).float()
    elif mode == 'soft':
        return torch.sigmoid(patch_scores / temperature)
    elif mode == 'none':
        return torch.ones_like(patch_scores)
    else:
        raise ValueError(f"[lif_to_attention] Unknown mode: {mode}")


def apply_attention_mask(features, attention_scores, mode='mul'):
    """
    Apply attention scores to patch features.

    Args:
        features (Tensor): (B, N, D), feature vectors for each patch
        attention_scores (Tensor): (B, N), attention weights per patch
        mode (str): 'mul' | 'zero' | 'scale'

    Returns:
        masked_features (Tensor): (B, N, D)
    """
    if mode == 'mul':
        return features * attention_scores.unsqueeze(-1)  # broadcasting
    elif mode == 'zero':
        mask = (attention_scores > 0).float().unsqueeze(-1)
        return features * mask
    elif mode == 'scale':
        min_val = attention_scores.min(dim=1, keepdim=True)[0]
        max_val = attention_scores.max(dim=1, keepdim=True)[0]
        norm_scores = (attention_scores - min_val) / (max_val - min_val + 1e-6)
        return features * norm_scores.unsqueeze(-1)
    else:
        raise ValueError(f"[apply_attention_mask] Unknown mode: {mode}")

def apply_attention_weights(features, attention_scores):
    """
    features: Tensor of shape (B, N, D)
    attention_scores: Tensor of shape (B, N) or (B, N, 1)

    Returns:
        Weighted features: (B, N, D)
    """
    if attention_scores.dim() == 2:
        attention_scores = attention_scores.unsqueeze(-1)  # (B, N, 1)
    return features * attention_scores  # broadcasting
