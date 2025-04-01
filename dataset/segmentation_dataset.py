# segmentation_dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        super().__init__()
        self.img_dir = os.path.join(root_dir, split, 'images')
        self.mask_dir = os.path.join(root_dir, split, 'masks')
        self.transform = transform

        #print(f"[INFO] 이미지 경로: {self.img_dir}")
        print(f"[INFO] 경로 존재 여부: {os.path.exists(self.img_dir)}")

        self.image_paths = sorted([
            os.path.join(self.img_dir, f)
            for f in os.listdir(self.img_dir)
            if f.lower().endswith(('jpg', 'jpeg', 'png'))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = os.path.join(self.mask_dir, os.path.basename(img_path))

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for image: {img_path}")

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # grayscale class IDs

        if self.transform:
            image = self.transform(image)
            mask = TF.to_tensor(mask).long().squeeze(0)  # (H, W)
        else:
            image = transforms.ToTensor()(image)
            mask = transforms.ToTensor()(mask).long().squeeze(0)

        return image, mask
