import os
import torchvision

def export_cifar10_to_folders(root='./data', split='train'):
    dataset = torchvision.datasets.CIFAR10(root=root, train=(split=='train'), download=True)
    out_dir = os.path.join(root, split)
    os.makedirs(out_dir, exist_ok=True)

    class_names = dataset.classes
    for class_name in class_names:
        os.makedirs(os.path.join(out_dir, class_name), exist_ok=True)

    for i, (img, label) in enumerate(dataset):
        class_name = class_names[label]
        # img는 이미 PIL 이미지이므로 추가 변환이 필요 없습니다.
        save_path = os.path.join(out_dir, class_name, f'{split}_{i:05d}.png')
        img.save(save_path)

    print(f"CIFAR-10 '{split}' split exported to folder structure at {out_dir}")

if __name__ == '__main__':
    export_cifar10_to_folders(root='./data', split='train')
    export_cifar10_to_folders(root='./data', split='val')
