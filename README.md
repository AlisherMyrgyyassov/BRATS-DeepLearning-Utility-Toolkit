# **BRATS Toolkit**

The BRATS Toolkit is a suite of tools designed to facilitate the processing and analysis of the Brain Tumor Segmentation (BRATS) dataset. It includes multiple data loaders, various augmentations for biomedical images, and useful functions such as cropping and one-hot channel addition.

## About BRATS2021
The BRATS2021 dataset contains MRI scans with the corresponding segmentation of brain tumors. Each voxel in the segmentation is labeled with one of the following classes:

1. **Background:** Represents the healthy brain tissue and is the majority class in the dataset. 
1. **Necrosis and Non-Enhancing Tumor:** Represents the non-enhancing tumor and the necrotic core of the enhancing tumor.
1. **Edema:** Represents the edema surrounding the tumor.
1. **Enhancing Tumor:** Represents the enhancing tumor.

## Datasets and DataLoaders
The datasets and data loaders include the following settings, which can be toggled on or off:

* One-hot tensor: This setting ensures that the model can distinguish between zero and close-to-zero small-value pixels.
* Crop by non-zero: This setting removes all the full-zero slices.
* Normalization: This setting normalizes the data with (X-mean)/std and then scales it from 0 to 1.

## Augmentations
The toolkit includes several augmentations specifically designed for biomedical imaging tasks:

* Random Crop
* Random Flip
* Random Axes Transpose
* Random Channel Replacement With Gaussian Noise

#  How to Use
To use the classes or functions in this toolkit, you need to import them as follows:

```
from dataset import ...
```

To create a clear BRATS dataset with no augmentations, use the ClearBRATS class from dataset.py:
```
dataset = ClearBRATS(data_dir="your/path/to/file")
```

The main BRATS class is a PyTorch dataset that can be used with any PyTorch data loader. It includes the following options:

```
main_dataset = BRATS (dataset, normalized=True, add_onehot=True,
                      is_cropped=False, augmentation_config=default_augmentations)
```

* dataset: A PyTorch dataset.
* normalized: If set to True, the dataset is normalized.
* add_onehot: If set to True, one-hot channel is added.
* is_cropped: If set to True, the dataset is cropped.
* augmentation_config: Configuration for data augmentations.
  
The augmentation_config is a dictionary that specifies the parameters for the augmentations. Here is the default configuration:

```
default_augmentations = {
    "random_crop": {
        "size": (72, 72, 72),
        "p": 1.0
    },
    "random_flip": {
        "p": 0.25
    },
    "random_axes_transpose": {
        "p": 0.1
    },
    "random_channel_gauss": {
        "mean" : 0, 
        "std" : 1, 
        "channels" : 4,
        "p": 0.2
    }
}
```

You can also create train and test datasets using the init_train_test_datasets function from dataset.py:

```
train_dataset, test_dataset = init_train_test_datasets(data_dir, train_ratio=0.8, normalized=True, is_cropped=True, add_onehot=True, augmentation_config=None, add_test_augmentations=False)
```

The BRATS class also supports two image display options: slice_display and montage_display. The slice_display function displays an image for each of the modalities, while the montage_display function displays all slices of one channel as a single image.

### Slice Display
For the slice display, you need to indicate the index of the desired sample and optionally the level where you are willing to slice the image as well as the figure size:
```
your_dataset.slice_display(idx, slice_num = 10, figsize=(15, 5))
```

![image](https://github.com/AlisherMyrgyyassov/BRATS-DeepLearning-Utility-Toolkit/assets/79082361/4abb863b-1189-4dae-8b03-7a0e177df767)

### Montage Display
Similarly to Slice Display, you have to choose the index first, and then optionally what channel (modality) you want to see all slices from (to see the mask, you may choose the last channel or -1).
```
your_dataset.montage_display(idx, channel = 0, figsize=(10, 10), cmap = 'gray')
```

![image](https://github.com/AlisherMyrgyyassov/BRATS-DeepLearning-Utility-Toolkit/assets/79082361/14ccb036-c831-469f-b197-d28835670af8)

## Custom Loss Functions
The losses.py file contains several custom loss functions that are useful for segmentation tasks:

* DiceLoss: Computes the Dice loss, which is commonly used for segmentation tasks.
* softmax_dice: Computes the Dice loss after applying a softmax function to the inputs.
* CombinedLoss: Combines the Dice loss and the cross-entropy loss. This can be useful for multi-class segmentation tasks.

These classes all inherit from nn.Module and can be used like any other PyTorch loss function.

## Segmentation Metrics
The seg_metrics.py file contains several functions for computing segmentation metrics:

* precision: Computes the precision of the segmentation.
* recall: Computes the recall of the segmentation.
* accuracy: Computes the accuracy of the segmentation.
* f1: Computes the F1 score of the segmentation.

These functions can be imported and used to evaluate the quality of the segmentation.

## Importing Functions and Classes
To import any of the functions or classes in this toolkit, use the following syntax:
```
from dataset import ClearBRATS, BRATS, init_train_test_datasets
from losses import DiceLoss, softmax_dice, CombinedLoss
from seg_metrics import precision, recall, accuracy, f1
```
