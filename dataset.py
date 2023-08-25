import os
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.utils.checkpoint as checkpoint
import torch.utils.data
from torch.utils.data import random_split

import torchio as tio
import torchio.transforms as transforms

from PIL import Image

import random
import math

from tools import crop_tensor, RandomChannelReplace, RandomCrop3D

"""
About BRATS2021:
0: Background - This class represents the healthy brain tissue and is the majority class in the dataset.
1: Necrosis and Non-Enhancing Tumor - This class represents the non-enhancing tumor and the necrotic core of the enhancing tumor.
2: Edema - This class represents the edema surrounding the tumor.
3: Enhancing Tumor - This class represents the enhancing tumor.

Settings:
One-hot tensor - 
Crop by non-zero - 
Normalization - 

Augmentations:
Random Crop - 
Random Flip - 
Random Axes Transpose - 
"""

class ClearBRATS(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.patient_ids = os.listdir(data_dir)

    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        scan_files = [os.path.join(self.data_dir, patient_id, f'{patient_id}_{modality}.nii.gz')
                      for modality in ['t1', 't2', 't1ce', 'flair']]
        mask_file = os.path.join(self.data_dir, patient_id, f'{patient_id}_seg.nii.gz')
        
        scans = []
        for scan_file in scan_files:
            scan_data = nib.load(scan_file).get_fdata()
            scans.append(scan_data)
        scans_tensor = np.stack(scans, axis=0)
        
        # Load mask file and convert to integer labels
        mask_data = nib.load(mask_file).get_fdata()
        mask_tensor = (mask_data > 0).astype(np.int64) + (mask_data == 4).astype(np.int64) * 2

        return scans_tensor, mask_tensor

class BRATS(Dataset):
    def __init__(self, dataset, normalized=True, add_onehot=True, is_cropped=False, augmentation_config=None):
        self.dataset = dataset
        self.is_cropped = is_cropped
        self.add_onehot = add_onehot
        self.normalized = normalized
        self.augmentation_config = augmentation_config

    def __len__(self):
        return len(self.dataset)

    def montage_display(self, idx, channel = 0, figsize=(10, 10), cmap = 'gray'):
        """
        Displays all slices as one image.
        Last channel is the mask
        """
        scans_tensor, mask_tensor = self.__getitem__(idx)
        scan_mask_merged = torch.cat([scans_tensor, mask_tensor.unsqueeze(dim = 0)], dim=0).numpy()
        input_array = scan_mask_merged[channel]
        input_array = np.transpose(input_array, (2, 0, 1))
        
        z, x, y = input_array.shape

        # Calculate number of plot rows and columns
        rows = int(math.sqrt(z))
        cols = math.ceil(z / rows)

        # Create a new figure
        fig, axs = plt.subplots(rows, cols, figsize=figsize)

        # Iterate through each 2D slice
        for i in range(z):
            # Calculate subplot row and column
            r, c = i // cols, i % cols

            # Plot 2D slice at subplot [r, c]
            axs[r, c].imshow(input_array[i], cmap=cmap)
            axs[r, c].axis('off')

        plt.show()

    def slice_display(self, idx, slice_num = 10, figsize=(15, 5)):
        scan, mask = self.__getitem__(idx)
        slice_idx = slice_num

        num_channels = scan.shape[0] + 1

        # Visualize the scan and mask using matplotlib
        fig, axs = plt.subplots(1, num_channels, figsize=figsize)
        for i in range(num_channels - 1):
            axs[i].imshow(scan[i][:, :, slice_idx], cmap='gray')
            axs[i].set_title(f'Modality {i+1}')
        axs[num_channels - 1].imshow(mask[:, :, slice_idx], cmap='gray')
        axs[num_channels - 1].set_title('Segmentation Mask')
        plt.show()


    def __getitem__(self, idx):
        scans_tensor, mask_tensor = self.dataset[idx]

        # Normalize scans and convert to PyTorch tensor
        if self.normalized == True:
            for channel_num in range(scans_tensor.shape[0]):
                channel_data = scans_tensor[channel_num]
                channel_data = scans_tensor[channel_num]
                non_zero_indices = np.nonzero(channel_data)

                mean = np.mean(channel_data[non_zero_indices])
                std = np.std(channel_data[non_zero_indices])

                channel_data[non_zero_indices] -= mean
                channel_data[non_zero_indices] /= std

                # Min-Max scaling
                min_val = np.min(channel_data[non_zero_indices])
                max_val = np.max(channel_data[non_zero_indices])
                channel_data[non_zero_indices] = (channel_data[non_zero_indices] - min_val) / (max_val - min_val)

                scans_tensor[channel_num] = channel_data

        scans_tensor = torch.from_numpy(scans_tensor)
        mask_tensor = torch.from_numpy(mask_tensor)

        # Add the last one-hot layer
        if self.add_onehot:
            one_hot = (scans_tensor[0]!= 0)*1.0
            one_hot = one_hot.unsqueeze(dim = 0) # Add channel dimensions
            scans_tensor = torch.cat([scans_tensor, one_hot], dim=0)

        if self.is_cropped: 
            scans_tensor = torch.cat([scans_tensor, mask_tensor.unsqueeze(dim = 0)], dim=0)
            scans_tensor = crop_tensor (scans_tensor)
            mask_tensor = scans_tensor[-1][:]
            scans_tensor = scans_tensor[:-1][:]

        # Augmentations
        if self.augmentation_config != None:
            # Merge scans and mask to ensure that augmentations are the same for both
            scan_mask_merged = torch.cat([scans_tensor, mask_tensor.unsqueeze(dim = 0)], dim=0)

            cfg = self.augmentation_config

            if "random_crop" in cfg:
                pr = random.random()
                if pr <= cfg["random_crop"]["p"]:
                    scan_mask_merged = RandomCrop3D(cfg["random_crop"]["size"])(scan_mask_merged)
            
            if "random_flip" in cfg:
                scan_mask_merged = transforms.RandomFlip(flip_probability = cfg["random_flip"]["p"])(scan_mask_merged)

            # Replaces random channel with a Gaussian distribution
            if "random_channel_gauss":
                mean = cfg["random_channel_gauss"]["mean"]
                std = cfg["random_channel_gauss"]["std"]
                channels = cfg["random_channel_gauss"]["channels"]
                prb = cfg["random_channel_gauss"]["p"]
                scan_mask_merged = RandomChannelReplace (mean=mean, std=std, channels=channels, p=prb)(scan_mask_merged)
            

            mask_tensor = scan_mask_merged[-1][:]
            scans_tensor = scan_mask_merged[:-1][:]


        return scans_tensor.float(), mask_tensor.long()

    def update_settings():
        print ("nothing so far")



# To be reworked
def init_train_test_datasets (data_dir, train_ratio=0.8, normalized=True, is_cropped=True, add_onehot=True, augmentation_config=None, add_test_augmentations=False):
    start_dataset = ClearBRATS (data_dir)

    train_size = int(train_ratio * len(start_dataset))
    test_size = len(start_dataset) - train_size

    torch.manual_seed(21)
    train_dataset, test_dataset = torch.utils.data.random_split(start_dataset, [train_size, test_size])
    train_dataset = BRATS (train_dataset, normalized=normalized, is_cropped=is_cropped, add_onehot=add_onehot, augmentation_config=augmentation_config)
    if add_test_augmentations:
        test_dataset = BRATS (test_dataset, normalized=normalized, is_cropped=is_cropped, add_onehot=add_onehot, augmentation_config=augmentation_config)
    else:
        test_dataset = BRATS (test_dataset)

    return train_dataset, test_dataset

