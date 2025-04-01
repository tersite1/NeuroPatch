import os
from PIL import Image
from torchvision import datasets
from torchvision.transforms import ToPILImage
import shutil

root_dir = "./data"
splits = ['train', 'val']

for split in splits:
    save_img_dir = os.path.join(root_dir, split, "images")
    save_mask_dir = os.path.join(root_dir, split, "masks")

    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_mask_dir, exist_ok=True)

    split_dir = os.path.join(root_dir, split)
    class_names = os.listdir(split_dir)

    count = 0
    for cls in class_names:
        class_path = os.path.join(split_dir, cls)
        if not os.path.isdir(class_path): continue

        for fname in os.listdir(class_path):
            if not fname.lower().endswith((".jpg", ".png")):
                continue

            src_path = os.path.join(class_path, fname)
            dst_path = os.path.join(save_img_dir, f"{count:06d}.png")
            shutil.copyfile(src_path, dst_path)

            # Create dummy mask (all zeros, same size as image)
            img = Image.open(src_path)
            dummy_mask = Image.new("L", img.size, color=0)
            dummy_mask.save(os.path.join(save_mask_dir, f"{count:06d}.png"))

            count += 1

print("✅ CIFAR classification format → segmentation format 변환 완료")
