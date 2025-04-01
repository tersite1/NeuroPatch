import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Neptune
import neptune

# dataset / detection / segmentation / classification 모듈
from dataset import get_dataloader
from detection import NeuroPatchDetector, SimpleDetectionLoss
from segmentation import NeuroPatchSegmentor, SimpleSegmentationLoss
from classification.classifier import NeuroPatchClassifier

# config
from config.neptune_token import NEPTUNE_API_TOKEN, PROJECT_NAME  # 토큰, 프로젝트명 별도 파일에서 가져오기

def parse_args():
    parser = argparse.ArgumentParser(description="STRAW project train script")
    parser.add_argument('--task', type=str,
                        choices=['classification','detection','segmentation'],
                        required=True,
                        help="Choose which task to train: classification, detection, or segmentation.")
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help="Path to YAML config file.")
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help="Path to load checkpoint from (if any).")
    parser.add_argument('--neptune', action='store_true',
                        help="Enable Neptune logging.")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def train():
    """
    STRAW - Spiking Transformer with Residual Attention Windows
    Main training script that handles classification/detection/segmentation tasks.
    """
    args = parse_args()
    config = load_config(args.config)

    # Neptune init (if enabled)
    run = None
    if args.use_neptune:
        run = neptune.init_run(
            project=PROJECT_NAME,
            api_token=NEPTUNE_API_TOKEN
        )
        # 기록: config
        run['config'] = config

    # device
    device = torch.device('cuda' if (torch.cuda.is_available() and config.get('cuda', False)) else 'cpu')

    # config에서 필요한 변수 로드
    img_size = config['backbone_config'].get('img_size', 224)
    patch_size = config['backbone_config'].get('patch_size', 16)
    num_classes = config['backbone_config'].get('num_classes', 10)
    batch_size = config.get('train_batchSize', 32)
    n_epochs = config.get('nEpochs', 10)
    lr = config['optimizer']['adam']['lr']

    # dataloader
    train_loader = get_dataloader(
        task=args.task,
        data_root='./data',
        batch_size=batch_size,
        img_size=img_size,
        num_workers=config.get('workers', 4),
        split='train',
        config=config
    )

    # 모델 설정
    if args.task == 'classification':
        model = NeuroPatchClassifier(num_classes=num_classes, img_size=img_size, patch_size=patch_size).to(device)
        criterion = nn.CrossEntropyLoss()

    elif args.task == 'detection':
        model = NeuroPatchDetector(num_classes=num_classes, img_size=img_size, patch_size=patch_size).to(device)
        criterion = SimpleDetectionLoss()  # (cls_pred, bbox_pred, cls_target, bbox_target)

    elif args.task == 'segmentation':
        model = NeuroPatchSegmentor(num_classes=num_classes, img_size=img_size, patch_size=patch_size).to(device)
        criterion = SimpleSegmentationLoss()
    else:
        raise ValueError(f"Unknown task: {args.task}")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # checkpoint 로드 (있다면)
    if args.ckpt_path and os.path.exists(args.ckpt_path):
        print(f"[INFO] Loading checkpoint from {args.ckpt_path}")
        model.load_state_dict(torch.load(args.ckpt_path, map_location=device))

    # 학습 루프
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0

        loop = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{n_epochs}]", leave=False)
        for batch_data in loop:
            optimizer.zero_grad()

            if args.task == 'classification':
                # batch_data = (images, labels)
                images, labels = batch_data
                images, labels = images.to(device), labels.to(device)

                logits = model(images)               # (B, num_classes)
                loss = criterion(logits, labels)

            elif args.task == 'detection':
                # batch_data = (images, cls_labels, bbox_labels) (가정)
                images, cls_labels, bbox_labels = batch_data
                images = images.to(device)
                cls_labels = cls_labels.to(device)
                bbox_labels = bbox_labels.to(device)

                cls_logits, bboxes = model(images)   # (B, Q, C), (B, Q, 4)
                loss = criterion(cls_logits, bboxes, cls_labels, bbox_labels)

            elif args.task == 'segmentation':
                # batch_data = (images, masks)
                images, masks = batch_data
                images, masks = images.to(device), masks.to(device)

                seg_logits = model(images)           # (B, num_classes, H, W)
                loss = criterion(seg_logits, masks)

            else:
                raise ValueError(f"Unknown task: {args.task}")

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{n_epochs}]  Loss: {avg_loss:.4f}")

        # Neptune 로그
        if run is not None:
            run['train/loss'].append(avg_loss)
            run['train/epoch'].log(epoch+1)

    # Neptune 종료
    if run is not None:
        run.stop()

if __name__ == '__main__':
    train()
