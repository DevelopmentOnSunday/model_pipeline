import os
import cv2
import copy
import random
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

file = os.path.join(os.getcwd(), "dataset")
floder = "training"
file_names = []

for file_name in glob(file + "/" + floder + "/*.png"):
    file_names.append(file_name)

class MangaDataset(Dataset):
    def __init__(self, images_filenames, transform=None):
        self.image_filenames = images_filenames
        self.transform = transform

        file = os.path.join(os.getcwd(), "dataset")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image = cv2.imread(image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image

train_transform = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
])

train_dataset = MangaDataset(file_names, transform= train_transform)
print(type(train_dataset))

def visualize_augmentations(dataset, idx=0, samples=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    figure, ax = plt.subplots(nrows=samples, ncols=2, figsize=(6 , 10))
    for i in range(samples):
        image = dataset[idx]
        ax[i, 0].imshow(image)
        ax[i, 0].set_title("Augmented image")
        ax[i, 0].set_axis_off()

    plt.tight_layout()
    plt.show()

random.seed(42)
visualize_augmentations(train_dataset, idx=55)