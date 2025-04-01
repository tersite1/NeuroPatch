import argparse
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from dataset import get_dataloader
from detection import NeuroPatchDetector, SimpleDetectionLoss
from segmentation import NeuroPatchSegmentor, SimpleSegmentationLoss
from utils.metrics import compute_pixel_accuracy, compute_mIoU
import os
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['detection', 'segmentation'], required=True)
    parser.add_argument('--config', type=str, default='config/default_config.yaml')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_optimizer(config, model):
    opt_cfg = config['optimizer']
    if opt_cfg['type'] == 'adam':
        return Adam(model.parameters(), **opt_cfg['adam'])
    elif opt_cfg['type'] == 'sgd':
        return SGD(model.parameters(), **opt_cfg['sgd'])
    elif opt_cfg['type'] == 'sam':
        print("[Warning] SAM optimizer is not implemented. Falling back to SGD.")
        return SGD(model.parameters(), lr=opt_cfg['sam']['lr'], momentum=opt_cfg['sam']['momentum'])
    else:
        raise ValueError("Unsupported optimizer type")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get('manualSeed', 42))

    device = torch.device('cuda' if config.get('cuda', False) and torch.cuda.is_available() else 'cpu')
    img_size = config['resolution']
    batch_size = config['train_batchSize']
    n_epochs = config['nEpochs']

    log_path = os.path.join(config['log_dir'], 'train_log.txt')
    os.makedirs(config['log_dir'], exist_ok=True)

    # DataLoader
    train_loader = get_dataloader(
        task=args.task,
        data_root='./data',
        batch_size=batch_size,
        img_size=img_size,
        num_workers=config.get('workers', 4),
        split='train',
        config=config
    )

    # Model
    backbone_cfg = config['backbone_config']
    if args.task == 'detection':
        model = NeuroPatchDetector(
            num_classes=backbone_cfg['num_classes'],
            img_size=img_size,
            patch_size=backbone_cfg.get('patch_size', 16)
        ).to(device)
        criterion = SimpleDetectionLoss()
    else:
        model = NeuroPatchSegmentor(
            num_classes=backbone_cfg['num_classes'],
            img_size=img_size,
            patch_size=backbone_cfg.get('patch_size', 16)
        ).to(device)
        criterion = SimpleSegmentationLoss()

    # Optimizer
    optimizer = build_optimizer(config, model)

    # LR Scheduler (optional)
    scheduler = None
    if config.get('lr_scheduler') == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step'], gamma=config['lr_gamma'])

    # Training
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        total_miou = 0
        for batch in train_loader:
            if args.task == 'detection':
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                cls_logits, bboxes = model(images)
                loss = criterion(cls_logits, bboxes, labels, torch.zeros_like(bboxes))
            else:
                images, masks = batch
                images, masks = images.to(device), masks.to(device)
                seg_logits = model(images)
                loss = criterion(seg_logits, masks.squeeze(1).long())

                # Metrics
                acc = compute_pixel_accuracy(seg_logits, masks.squeeze(1))
                miou = compute_mIoU(seg_logits, masks.squeeze(1), num_classes=backbone_cfg['num_classes'])
                total_acc += acc.item()
                total_miou += miou.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if scheduler:
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        log_msg = f"[Epoch {epoch+1}/{n_epochs}] Loss: {avg_loss:.4f}"
        if args.task == 'segmentation':
            avg_acc = total_acc / len(train_loader)
            avg_miou = total_miou / len(train_loader)
            log_msg += f", Pixel Acc: {avg_acc:.4f}, mIoU: {avg_miou:.4f}"

        print(log_msg)
        with open(log_path, 'a') as f:
            f.write(log_msg + "\n")

        # Save checkpoint
        if config.get('save_ckpt', False) and (epoch + 1) % config.get('save_epoch', 1) == 0:
            ckpt_path = os.path.join(config.get('log_dir', './logs'), f"ckpt_epoch{epoch+1}.pt")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)


if __name__ == '__main__':
    main()