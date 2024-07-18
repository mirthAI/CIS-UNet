import os
import glob
from pathlib import Path
from random import seed
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    RandRotate90d,
    Spacingd,
    RandAffined
)

def image_and_masks_paths(root_dir):
    # Initialize empty lists to store image and segmentation file paths
    image_paths = []
    segmentation_paths = []

    # Iterate through all subdirectories within the root directory
    for subdir, dirs, files in os.walk(root_dir):
        # Check for the "Volumes" subdirectory (assuming it contains the image files)
        images_subdir = os.path.join(subdir, "Volumes")
        if os.path.isdir(images_subdir):
            # Find the image file within the "Volumes" subdirectory
            for filename in os.listdir(images_subdir):
                if filename.lower().endswith((".nii.gz")):  # Adjust extensions as needed
                    image_path = os.path.join(images_subdir, filename)
                    image_paths.append(image_path)

        # Check for the "Labels" subdirectory (assuming it contains the segmentation files)
        masks_subdir = os.path.join(subdir, "Labels")
        if os.path.isdir(masks_subdir):
            # Find the segmentation file within the "Labels" subdirectory
            for filename in os.listdir(masks_subdir):
                segmentation_path = os.path.join(masks_subdir, filename)
                segmentation_paths.append(segmentation_path)
    
    # Combine image and label paths into a list of dictionaries
    files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(sorted(image_paths), sorted(segmentation_paths))]
    return files


class DatasetProcessor:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        # Get list of image and mask file paths
        self.files = image_and_masks_paths(self.root_dir)
        total_images = len(self.files)
        

    def get_train_transforms(self, patch_size=128, num_samples=4):
        return Compose([
            # Load images and labels and ensure they have channels first
            LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=False),
            # Scale intensity of images to a specified range
            ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            # Crop the foreground of the image
            CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            # Reorient the image to the RAS (Right-Anterior-Superior) orientation
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Resample the image to a specified pixel spacing
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
            # Randomly crop patches from the image, ensuring balance between positive and negative labels
            RandCropByPosNegLabeld(keys=["image", "label"], 
                                   label_key="label", 
                                   spatial_size=(patch_size, patch_size, patch_size),
                                   pos=1, neg=1, num_samples=num_samples, 
                                   image_key="image", image_threshold=0),
            # Randomly flip the image along specified axes
            RandFlipd(keys=["image", "label"], spatial_axis=[0, 1, 2], prob=0.10),
            # Randomly rotate the image by 90 degrees
            RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
            # Randomly shift the intensity of the image
            RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
        ])
    
    def get_val_transforms(self):
        return Compose([
            # Load images and labels and ensure they have channels first
            LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=False),
            # Scale intensity of images to a specified range
            ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            # Crop the foreground of the image
            CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            # Reorient the image to the RAS (Right-Anterior-Superior) orientation
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Resample the image to a specified pixel spacing
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
        ])
