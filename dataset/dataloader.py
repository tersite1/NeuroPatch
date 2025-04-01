import os

from torch.utils.data import DataLoader
from torchvision import transforms
from .segmentation_dataset import SegmentationDataset
from .detection_dataset import DetectionDataset
from .classification_dataset import ClassificationDataset

def build_transforms(config, task='detection'):
    aug_cfg = config.get('data_aug', {})
    use_aug = config.get('use_data_augmentation', False)

    transform_list = []
    if use_aug:
        if aug_cfg.get('flip_prob', 0) > 0:
            transform_list.append(transforms.RandomHorizontalFlip(p=aug_cfg['flip_prob']))
        if aug_cfg.get('rotate_prob', 0) > 0:
            degrees = aug_cfg.get('rotate_limit', [-10, 10])
            transform_list.append(transforms.RandomRotation(degrees))
        if aug_cfg.get('brightness_prob', 0) > 0:
            brightness = aug_cfg.get('brightness_limit', [0.0, 0.0])
            transform_list.append(transforms.ColorJitter(brightness=tuple(map(abs, brightness))))
        if aug_cfg.get('blur_prob', 0) > 0:
            transform_list.append(transforms.GaussianBlur(kernel_size=3))
        if aug_cfg.get('contrast_limit', [0.0, 0.0]) != [0.0, 0.0]:
            contrast = aug_cfg['contrast_limit']
            transform_list.append(transforms.ColorJitter(contrast=tuple(map(abs, contrast))))

    transform_list.extend([
        transforms.Resize((config['resolution'], config['resolution'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std'])
    ])

    return transforms.Compose(transform_list)

def get_dataloader(task, data_root, batch_size, img_size, num_workers, split, config):
    transform = build_transforms(config, task=task)

    if task == 'detection':
        dataset = DetectionDataset(data_root, split=split, transform=transform)
    elif task == 'segmentation':
        dataset = SegmentationDataset(data_root, split=split, transform=transform)
    elif task == 'classification':
        # ★ classification dataset 선택
        dataset = ClassificationDataset(data_root, split=split, transform=transform)
    else:
        raise ValueError(f"Unknown task {task}")

    shuffle = (split == 'train')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader