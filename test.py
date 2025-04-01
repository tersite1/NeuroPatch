import argparse
import yaml
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import neptune

from dataset import get_dataloader
from detection import NeuroPatchDetector
from segmentation import NeuroPatchSegmentor
from classification.classifier import NeuroPatchClassifier
from utils.metrics import compute_pixel_accuracy, compute_mIoU
from config.neptune_token import NEPTUNE_API_TOKEN, PROJECT_NAME

def parse_args():
    parser = argparse.ArgumentParser(description="STRAW project test script")
    parser.add_argument('--task', type=str, choices=['classification', 'detection', 'segmentation'], required=True)
    parser.add_argument('--config', type=str, default='config/default_config.yaml')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--neptune', dest='use_neptune', action='store_true')
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
    num_classes = config['backbone_config']['num_classes']

    test_loader = get_dataloader(
        task=args.task,
        data_root=config.get('data_path', './data'),
        batch_size=batch_size,
        img_size=img_size,
        num_workers=config.get('workers', 4),
        split='val',
        config=config
    )

    if args.task == 'classification':
        model = NeuroPatchClassifier(num_classes=num_classes, img_size=img_size)
    elif args.task == 'detection':
        model = NeuroPatchDetector(num_classes=num_classes, img_size=img_size)
    elif args.task == 'segmentation':
        model = NeuroPatchSegmentor(num_classes=num_classes, img_size=img_size)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
    model.to(device)
    model.eval()

    run = None
    if args.use_neptune:
        run = neptune.init_run(project=PROJECT_NAME, api_token=NEPTUNE_API_TOKEN)

    correct, total, total_miou = 0, 0, 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            if args.task == 'classification':
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.numel()
            elif args.task == 'detection':
                images, cls_labels, _ = batch
                images, cls_labels = images.to(device), cls_labels.to(device)
                cls_logits, _ = model(images)
                preds = cls_logits.argmax(dim=-1)
                correct += (preds == cls_labels).float().sum().item()
                total += cls_labels.numel()
            elif args.task == 'segmentation':
                images, masks = batch
                images, masks = images.to(device), masks.to(device)
                seg_logits = model(images)
                acc = compute_pixel_accuracy(seg_logits, masks.squeeze(1))
                miou = compute_mIoU(seg_logits, masks.squeeze(1), num_classes=num_classes)
                correct += acc.item()
                total_miou += miou.item()
                total += 1

    if args.task == 'segmentation':
        avg_acc = correct / total
        avg_miou = total_miou / total
        print(f"[Segmentation] Pixel Acc: {avg_acc:.4f} | mIoU: {avg_miou:.4f}")
        if run:
            run["test/pixel_acc"] = avg_acc
            run["test/miou"] = avg_miou
    else:
        avg_acc = correct / total
        print(f"[Test] Accuracy: {avg_acc:.4f}")
        if run:
            run["test/accuracy"] = avg_acc

    if run:
        run.stop()

if __name__ == '__main__':
    evaluate()