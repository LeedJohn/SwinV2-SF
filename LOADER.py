import glob
import pandas as pd
import torch
import numpy as np
from PIL import Image
from skimage.io import imread
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import cv2
import os


def create_csv_from_folder(folder_path, csv_name):
    IMAGES_PATH = os.path.join(folder_path, 'images/')
    all_images = glob.glob(os.path.join(IMAGES_PATH, "*.png"))
    
    data = {'image_name': [os.path.basename(path) for path in all_images]}
    df = pd.DataFrame(data)
    csv_full_path = os.path.join(folder_path, f"{csv_name}.csv")
    df.to_csv(csv_full_path, index=False)
    print(f"CSV文件已创建于: {csv_full_path}")


def load_data_to_model(img_size, folder_path, csv_path):
    df = pd.read_csv(csv_path)
    images_folder = os.path.join(folder_path, 'images/')
    masks_folder = os.path.join(folder_path, 'masks/')
    all_ids = df['image_name'].tolist()
    X = np.zeros((len(all_ids), img_size, img_size, 3), dtype=np.float32)
    Y = np.zeros((len(all_ids), img_size, img_size), dtype=np.uint8)
    IDs = []

    for n, id_ in tqdm(enumerate(all_ids), desc="加载数据"):
        IDs.append(id_)
        image_path = os.path.join(images_folder, id_)
        mask_path = os.path.join(masks_folder, id_)

        image = imread(image_path)
        mask_ = imread(mask_path)
        
        resized_image = cv2.resize(image, (img_size, img_size))
        resized_mask = cv2.resize(mask_, (img_size, img_size), interpolation=cv2.INTER_NEAREST)  
        
        X[n] = resized_image / 255.0
        
        mask_gray = cv2.cvtColor(resized_mask, cv2.COLOR_BGR2GRAY) if len(resized_mask.shape)==3 else resized_mask
        mask = (mask_gray >= 127).astype(np.uint8)
        Y[n] = mask

    Y = np.expand_dims(Y, axis=-1)  
    return IDs, X, Y


class Polyp_Dataset(Dataset):
    def __init__(self, IDs, X, Y, geo_transform=None, color_transform=None):
        self.IDs = IDs
        self.X = X
        self.Y = Y
        self.geo_transform = geo_transform
        self.color_transform = color_transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        image = self.X[index]
        mask = self.Y[index]

        
        if self.geo_transform:
            augmented = self.geo_transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        if self.color_transform:
            augmented = self.color_transform(image=image)
            image = augmented['image']
        
        
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).permute(2, 0, 1).long() 

        return self.IDs[index], image, mask
