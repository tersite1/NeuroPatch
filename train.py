import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import neptune

from dataset import get_dataloader
from detection import NeuroPatchDetector, SimpleDetectionLoss
from segmentation import NeuroPatchSegmentor, SimpleSegmentationLoss
from classification.classifier import NeuroPatchClassifier
from config.neptune_token import NEPTUNE_API_TOKEN, PROJECT_NAME
from utils.backbone_selector import get_backbone_model

def parse_args():
    parser = argparse.ArgumentParser(description="STRAW project train script")
    parser.add_argument('--task', type=str, choices=['classification', 'detection', 'segmentation'], required=True)
    parser.add_argument('--config', type=str, default='config/default_config.yaml')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--neptune', dest='use_neptune', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def train():
    args = parse_args()
    config = load_config(args.config)
    
    # 환경변수 LOCAL_RANK를 우선 사용하여 local_rank 값을 설정합니다.
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    
    # DDP 초기화 및 GPU 할당 (수정됨: local_rank 사용)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    run = None
    if args.use_neptune and local_rank == 0:
        run = neptune.init_run(project=PROJECT_NAME, api_token=NEPTUNE_API_TOKEN)
        run['config'] = str(config)  # Neptune에 기록하기 전에 문자열로 변환

    img_size = config['backbone_config'].get('img_size', 224)
    patch_size = config['backbone_config'].get('patch_size', 16)
    num_classes = config['backbone_config'].get('num_classes', 10)
    batch_size = config.get('train_batchSize', 32)
    n_epochs = config.get('nEpochs', 10)
    lr = config['optimizer']['adam']['lr']
    backbone_type = config['backbone_config'].get('type', 'vgg')

    train_loader = get_dataloader(
        task=args.task,
        data_root=config.get('data_path', './data'),
        batch_size=batch_size,
        img_size=img_size,
        num_workers=config.get('workers', 4),
        split='train',
        config=config
    )

    # 백본 선택 및 모델 구성 (weights 대신 pretrained 사용)
    backbone_model = get_backbone_model(
        model_type=backbone_type,
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=config['backbone_config'].get('embed_dim', 768),
        pretrained=args.pretrained
    )

    if args.task == 'classification':
        model = NeuroPatchClassifier(num_classes=num_classes, backbone=backbone_model)
        criterion = nn.CrossEntropyLoss()
    elif args.task == 'detection':
        model = NeuroPatchDetector(num_classes=num_classes, backbone=backbone_model)
        criterion = SimpleDetectionLoss()
    elif args.task == 'segmentation':
        model = NeuroPatchSegmentor(num_classes=num_classes, backbone=backbone_model)
        criterion = SimpleSegmentationLoss()
    else:
        raise ValueError(f"Unknown task: {args.task}")

    model.to(device)

    # DDP 초기화 시 local_rank를 사용하여 각 프로세스가 고유한 GPU를 사용하도록 변경함
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get('lr_step', 5),
        gamma=config.get('lr_gamma', 0.5)
    )

    start_epoch = 0
    best_loss = float('inf')

    if args.ckpt_path and os.path.exists(args.ckpt_path):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(args.ckpt_path, map_location=map_location)
        model.module.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_loss = checkpoint.get('best_loss', float('inf'))

    for epoch in range(start_epoch, n_epochs):
        model.train()

        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        total_loss = 0.0

        loop = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{n_epochs}]", leave=False) if local_rank == 0 else train_loader

        for batch_data in loop:
            optimizer.zero_grad()

            if args.task == 'classification':
                images, labels = batch_data
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)

            elif args.task == 'detection':
                images, cls_labels, bbox_labels = batch_data
                images, cls_labels, bbox_labels = images.to(device), cls_labels.to(device), bbox_labels.to(device)
                cls_logits, bboxes = model(images)
                loss = criterion(cls_logits, bboxes, cls_labels, bbox_labels)

            elif args.task == 'segmentation':
                images, masks = batch_data
                images, masks = images.to(device), masks.to(device)
                seg_logits = model(images)
                loss = criterion(seg_logits, masks)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if local_rank == 0:
                loop.set_postfix(loss=loss.item())

        scheduler.step()

        if local_rank == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{n_epochs}] Loss: {avg_loss:.4f}")

            if run is not None:
                run['train/loss'].append(avg_loss)

            log_dir = config.get('log_dir', './logs')
            os.makedirs(log_dir, exist_ok=True)

            if config.get('save_ckpt', True):
                save_path = os.path.join(log_dir, f"ckpt_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_loss': best_loss,
                }, save_path)

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_path = os.path.join(log_dir, "best_ckpt.pth")
                    torch.save({
                        'epoch': epoch,
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_loss': best_loss,
                    }, best_path)

    if run is not None and local_rank == 0:
        run.stop()

if __name__ == '__main__':
    train()
