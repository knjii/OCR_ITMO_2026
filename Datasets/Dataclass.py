import math
import os
import random as _random

import cv2
from torch.utils.data.dataset import Dataset


class OldBooksDataclass(Dataset):
    """Dataclass for old-books-dataset"""

    def __init__(self, root_dir, images_folder='300dpi', ground_truths_folder='groundtruth',
                 split=None, val_size=0.15, test_size=0.15, seed=42,
                 augmentation=None, preprocessing=None):
        """
        Args:
            root_dir               - path of the dataset
            images_folder          - name of folder with images
            ground_truths_folder   - name of folder with ground_truths
            split                  - one of 'train', 'val', 'test', or None (all data)
            val_size               - fraction of data for validation (used when split is set)
            test_size              - fraction of data for test (used when split is set)
            seed                   - random seed for reproducible splits
            augmentation           - applied augmentations on images
            preprocessing          - applied preprocessing on images
        """

        super().__init__()

        self.root_dir = root_dir
        self.image_folder = images_folder
        self.ground_truth_folder = ground_truths_folder
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        images_dir = os.path.join(self.root_dir, self.image_folder)
        all_images = os.listdir(images_dir)
        self.image_ids = [img.replace('.tiff', '') for img in all_images if img.endswith('.tiff')]

        if split is not None:
            rng = _random.Random(seed)
            ids = sorted(self.image_ids)
            rng.shuffle(ids)
            n = len(ids)
            n_test = math.ceil(n * test_size)
            n_val = math.ceil(n * val_size)
            if split == 'test':
                self.image_ids = ids[:n_test]
            elif split == 'val':
                self.image_ids = ids[n_test:n_test + n_val]
            elif split == 'train':
                self.image_ids = ids[n_test + n_val:]
            else:
                raise ValueError(f"Unknown split: {split!r}. Expected 'train', 'val' or 'test'.")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        image_path = os.path.join(self.root_dir, self.image_folder, f"{image_id}.tiff")
        ground_truth_path = os.path.join(self.root_dir, self.ground_truth_folder, f"{image_id}.txt")

        image = cv2.imread(image_path)
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            text = f.read()

        if self.augmentation:
            image = self.augmentation(image)

        if self.preprocessing:
            image = self.preprocessing(image)

        return image, text
