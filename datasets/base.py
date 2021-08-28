"""
"""
import os
import glob
import random

import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torchvision.transforms.transforms import Resize


class FaceEmbeddingDataset(Dataset):
    def __init__(self, image_dir_list, same_prob, image_size=256, transform=None):
        self.image_dir_list = image_dir_list
        self.same_prob = same_prob
        self.extensions = ['.jpg', '.jpeg', '.png']
        if transform is None:
            self.transforms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            self.transforms = transforms
        
        # Load image paths
        self.image_paths = self._load_data()
    
    def _load_data(self):
        """
        """
        image_paths = []
        self.N = []
        for image_dir in self.image_dir_list:
            for ext in self.extensions:
                image_list = glob.glob(f'{image_dir}/*{ext}')
                image_paths += image_list
            self.N.append(len(image_list))
        return image_paths

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        Xs = Image.open(image_path).convert('RGB')

        if random.random() < self.same_prob:
            image_path = random.choice(self.image_paths)
            Xt = Image.open(image_path).convert('RGB')
            same_person = 0
        else:
            Xt = Xs.copy()
            same_person = 1
        return self.transforms(Xs), self.transforms(Xt), same_person

    def __len__(self):
        return len(self.image_paths)