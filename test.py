import argparse
import yaml
import torch
import torch.nn as nn
from dataset import get_dataloader
from detection import NeuroPatchDetector
from segmentation import NeuroPatchSegmentor
from utils.metrics import compute_pixel_accuracy, compute_mIoU
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['detection', 'segmentation'], required=True)
    parser.add_argument('--config', type=str, default='config/default_config.yaml')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to checkpoint')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate():
    args = parse_args()
    config = load_config(args.config)

    device = torch.device('cuda' if config.get('cuda', False) and torch.cuda.is_available() else 'cpu')
    img_size = config['resolution']
    batch_size = config['test_batchSize']

    test_loader = get_dataloader(
        task=args.task,
        data_root='./data',
        batch_size=batch_size,
        img_size=img_size,
        num_workers=config.get('workers', 4),
        split='val',
        config=config
    )

    backbone_cfg = config['backbone_config']
    if args.task == 'detection':
        model = NeuroPatchDetector(
            num_classes=backbone_cfg['num_classes'],
            img_size=img_size,
            patch_size=backbone_cfg.get('patch_size', 16)
        ).to(device)
    else:
        model = NeuroPatchSegmentor(
            num_classes=backbone_cfg['num_classes'],
            img_size=img_size,
            patch_size=backbone_cfg.get('patch_size', 16)
        ).to(device)

    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    total_acc, total_miou = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            if args.task == 'detection':
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                cls_logits, _ = model(images)
                preds = cls_logits.argmax(dim=-1)
                acc = (preds == labels).float().mean()
                total_acc += acc.item()
            else:
                images, masks = batch
                images, masks = images.to(device), masks.to(device)
                seg_logits = model(images)
                acc = compute_pixel_accuracy(seg_logits, masks.squeeze(1))
                miou = compute_mIoU(seg_logits, masks.squeeze(1), num_classes=backbone_cfg['num_classes'])
                total_acc += acc.item()
                total_miou += miou.item()

    avg_acc = total_acc / len(test_loader)
    print(f"[Test Result] Accuracy: {avg_acc:.4f}")

    if args.task == 'segmentation':
        avg_miou = total_miou / len(test_loader)
        print(f"mIoU: {avg_miou:.4f}")

if __name__ == '__main__':
    evaluate()