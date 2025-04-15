import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
import numpy as np


def load_annotations(label_path):
    with open(label_path, 'r') as f:
        anns = json.load(f)
    return anns


def crop_image_with_overlap(image, size=1024, overlap=0.2):
    h, w, _ = image.shape
    step = int(size * (1 - overlap))
    patches = []
    for y in range(0, h - size + 1, step):
        for x in range(0, w - size + 1, step):
            patch = image[y:y + size, x:x + size]
            patches.append((patch, (x, y)))
    return patches


class ShipDataset(Dataset):
    def __init__(self, cfg, mode='train'):
        super().__init__()
        self.mode = mode
        self.root = os.path.join(cfg.data_root, cfg.dataset, mode)
        self.image_dir = os.path.join(self.root, 'images')
        self.label_dir = os.path.join(self.root, 'labels')
        self.image_list = sorted(os.listdir(self.image_dir))
        self.input_size = cfg.input_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.image_dir, image_name)
        label_path = os.path.join(self.label_dir, image_name.replace('.jpg', '.json'))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_size, self.input_size))

        anns = load_annotations(label_path)
        boxes = torch.tensor([ann['bbox'] for ann in anns], dtype=torch.float32)  # [x1,y1,x2,y2]
        labels = torch.tensor([ann['category_id'] for ann in anns], dtype=torch.long)

        image = self.transform(image)

        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'image_id': image_name
        }

