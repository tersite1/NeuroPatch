# detection_dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset

class DetectionDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        super().__init__()
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform

        # ImageNet / CIFAR-10 구조 가정
        self.samples = []
        for class_name in sorted(os.listdir(self.root_dir)):
            class_path = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('png', 'jpg', 'jpeg')):
                    self.samples.append((os.path.join(class_path, fname), class_name))

        # 클래스 이름 → 인덱스 맵핑
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted({label for _, label in self.samples}))}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_name = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label
