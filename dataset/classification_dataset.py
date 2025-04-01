import os
from PIL import Image
from torch.utils.data import Dataset

class ClassificationDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        super().__init__()
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform

        self.samples = []
        # CIFAR-10/Imagenet-style: root_dir/class_name/xxx.png
        for class_name in sorted(os.listdir(self.root_dir)):
            class_path = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('jpg','jpeg','png')):
                    self.samples.append((
                        os.path.join(class_path, fname), class_name
                    ))
        
        # class → index
        classes = sorted(list({s[1] for s in self.samples}))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_name = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[label_name]

        if self.transform:
            image = self.transform(image)
        return image, label
