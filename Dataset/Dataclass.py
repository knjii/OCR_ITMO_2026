from torch.utils.data.dataset import Dataset
import cv2
import os

class OldBooksDataclass(Dataset):
    """Dataclass for old-books-dataset"""

    def __init__(self, root_dir, images_folder='300dpi', ground_truths_folder='groundtruth', 
                 augmentation=None, preprocessing=None):
        """
        Args:
            root_dir - path of the dataset
            images_folder - name of folder with images
            ground_truths_folder - name of folder with ground_truths
            augmentations - applied augmentations on images
            preprocessing - applied preprocessing on images

        """

        super.__init__(self)

        self.root_dir = root_dir
        self.image_folder = images_folder
        self.ground_truth_folder = ground_truths_folder
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        images_dir = os.path.join(self.root_dir, self.image_folder)
        all_images = os.listdir(images_dir)
        self.image_ids = [img.replace('.tiff', '') for img in all_images if img.endswith('.tiff')]

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        image_id = self.image_ids[idx]

        image_path = os.path.join(self.root_dir, self.image_folder, f"{image_id}.tiff")

        ground_truth_id = image_id
        ground_truth_path = os.path.join(self.root_dir, self.ground_truth_folder, f"{ground_truth_id}.txt")

        image = cv2.imread(image_path)
        text = open(ground_truth_path, 'r', encoding='utf-8')

        if self.augmentation:
            image = self.augmentation(image)

        if self.preprocessing:
            image = self.preprocessing(image)

        return image, text
     