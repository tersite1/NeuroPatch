# metrics.py

import torch

def compute_pixel_accuracy(pred, target):
    """
    pred: (B, C, H, W) - logits
    target: (B, H, W) - int labels
    """
    pred_labels = pred.argmax(dim=1)  # (B, H, W)
    correct = (pred_labels == target).float()
    return correct.sum() / target.numel()

def compute_mIoU(pred, target, num_classes):
    """
    pred: (B, C, H, W) - logits
    target: (B, H, W) - int labels
    """
    pred_labels = pred.argmax(dim=1)  # (B, H, W)
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred_labels == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        if union == 0:
            continue
        ious.append(intersection / union)

    if len(ious) == 0:
        return torch.tensor(1.0)  # 모든 클래스가 공백이면 100% IoU 처리
    return torch.mean(torch.stack(ious))
