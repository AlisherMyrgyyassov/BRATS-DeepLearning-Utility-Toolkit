import random
import os
import numpy as np

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