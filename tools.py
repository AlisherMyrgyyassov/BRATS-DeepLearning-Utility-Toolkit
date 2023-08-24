import random
import os
import numpy as np
import torch

# Convert BRATS2021 dataset to HDF5

def convert_brats_folder_to_hdf5(brats_folder, train_file, val_file, test_file, train_size = 0.8, val_size = 0.1):
    """
    This function converts a BRATS folder to a HDF5 file.
    """ 

    patient_ids = [name for name in os.listdir(brats_folder) if os.path.isdir(os.path.join(brats_folder, name))]
    
    # Splitting into 3 subsets
    random.seed(21)
    random.shuffle(patient_ids)

    train_len = int(len(patient_ids) * train_size)
    val_len = int(len(patient_ids) * val_size)
    
    train_ids = patient_ids[:train_len]
    val_ids = patient_ids[train_len:train_len+val_len]
    test_ids = patient_ids[train_len+val_len:]


    # Creating 3 HDF5 files
    for ids, filename in zip([train_ids, val_ids, test_ids], [train_file, val_file, test_file]):
        hdf5_file = h5py.File(filename, mode='w')

        for patient_id in ids:
            group = hdf5_file.create_group(patient_id)

            for root, dirs, files in os.walk(os.path.join(brats_folder, patient_id)):
                if len(files) > 0:
                    for file in files:
                        if file.endswith(('.nii', '.nii.gz')):
                            nifti_file = nib.load(os.path.join(root, file))
                            volume = np.array(nifti_file.get_fdata())

                            group.create_dataset(file.replace('.nii.gz', ''), data=volume)
            
            print (patient_id, "folder created")

        hdf5_file.close()


def crop_tensor(tensor):
    """
    Crop a 4D tensor along the spatial dimensions to remove slices that only contain zeros.
    """
    # Find the indices where the tensor is not zero
    indices = np.where(tensor != 0)

    # Get min and max indices along each axis (ignoring the channel dimension)
    min_height, max_height = np.min(indices[1]), np.max(indices[1])
    min_width, max_width = np.min(indices[2]), np.max(indices[2])
    min_depth, max_depth = np.min(indices[3]), np.max(indices[3])

    # Crop the tensor
    cropped_tensor = tensor[:, min_height:max_height+1, min_width:max_width+1, min_depth:max_depth+1]

    return cropped_tensor

# =========================================================
# ========================================================= Augmentations
# =========================================================
class RandomChannelReplace:
    def __init__(self, mean=0, std=1, channels=4, p=0.5):
        self.mean = mean
        self.std = std
        self.channels = channels
        self.p = p

    def __call__(self, image):
        if np.random.rand() < self.p:
            channel_to_replace = np.random.randint(self.channels) # from 0 to 3 by default 
            noise = np.random.normal(self.mean, self.std, size=image[channel_to_replace].shape)
            image[channel_to_replace] = torch.tensor(noise)
        return image


class RandomCrop3D:
    """
    Custom module for 3D images as torchvision does not support 3D images
    output_size - a tuple (x, y, z)
    """
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        x, y, z = sample.shape[-3:]
        x_start = np.random.randint(0, x - self.output_size[0])
        y_start = np.random.randint(0, y - self.output_size[1])
        z_start = np.random.randint(0, z - self.output_size[2])
        return sample[..., x_start:x_start+self.output_size[0], y_start:y_start+self.output_size[1], z_start:z_start+self.output_size[2]]