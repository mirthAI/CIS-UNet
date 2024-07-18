
import argparse
import os
import torch
from pathlib import Path  # Import pathlib for handling paths
from monai.data import (DataLoader, CacheDataset, decollate_batch)
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from utils.dataset_processor import DatasetProcessor, image_and_masks_paths
from utils.CIS_UNet import CIS_UNet
from monai.losses import DiceCELoss
from sklearn.model_selection import KFold
from utils.training_validation import Trainer


# Define an argument parser to handle command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True, help="Path to the root directory of the dataset")
parser.add_argument("--saved_model_dir", type=str, required=True, help="Path to the root directory where the best model is saved")
parser.add_argument("--num_classes", type=int, default=24, help="Number of classes for segmentation")
parser.add_argument("--max_iterations", type=int, default=30000, help="Maximum number of training iterations")
parser.add_argument("--eval_num", type=int, default=500, help="Number of validation images for evaluation")
parser.add_argument("--patch_size", type=int, default=128, help="Size of patches for training")
parser.add_argument("--spatial_dims", type=int, default=3, help="For 3D data it is 3 for 2D data it is 2")
parser.add_argument("--feature_size", type=int, default=48, help="Initial Filters for SegResNet Model")
parser.add_argument("--num_samples", type=int, default=4, help="Number of Samples per batch")
parser.add_argument("--num_folds", type=int, default=4, help="K for K-fold Cross-Validation")
parser.add_argument("--in_channels", type=int, default=1, help="Number of input channels")
parser.add_argument("--encoder_channels", nargs="+", type=int, default=[64, 64, 128, 256], help="Number of encoder channels")
parser.add_argument("--norm_name", type=str, default='instance', help="Type of normalization")

args = parser.parse_args()  # Parse command-line arguments


def main():
    print("*"*100)
    print(args)
    print("*"*100)
    skf = KFold(n_splits=args.num_folds, shuffle=True, random_state=92)
    if not os.path.exists(args.saved_model_dir):
        # Create the directory if it doesn't exist
        os.makedirs(args.saved_model_dir, exist_ok=True)
        print(f"Directory '{args.saved_model_dir}' created successfully.")
    files = image_and_masks_paths(args.data_dir)
    processor = DatasetProcessor(args.data_dir)
    for fold, (train_indices, val_indices) in enumerate(skf.split(files)):
        print(f"Processing Fold {fold}")
        train_files = [files[i] for i in train_indices]
        val_files = [files[i] for i in val_indices]
        
        train_images = [item['image'].split('/')[-1].split('.')[0] for item in train_files]
        val_images = [item['image'].split('/')[-1].split('.')[0] for item in val_files]
        
        

        print(f"Training Images Names: {', '.join(train_images)}")
        print('')
        print(f"Validation Images Names: {', '.join(val_images)}")

        train_transforms = processor.get_train_transforms(args.patch_size, args.num_samples)
        val_transforms = processor.get_val_transforms()
        
        num_cpus = torch.get_num_threads()
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_num=len(train_files), cache_rate=1.0, num_workers=num_cpus//2)
        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=len(val_files), cache_rate=1.0, num_workers=num_cpus//2)
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=num_cpus//2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_cpus//2, pin_memory=True)
        print("Dataset is loaded and prepared for training and validation...\n")
        # print(f"Training Dataset: {len(train_loader)} | Validation Dataset: {len(val_loader)}")
        ###################### DEFINING THE MODEL ################################################
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CIS_UNet(spatial_dims=args.spatial_dims, 
                         in_channels=args.in_channels, 
                         num_classes=args.num_classes, 
                         encoder_channels=args.encoder_channels,
                         feature_size=args.feature_size)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model is defined with total paramters: {num_params}\n")
        print(f"Model is loading to the appropriate device...\n")        
        if torch.cuda.device_count() > 1:
            print('There are more than 1 GPUs... Hurray... Parallel Processing\n')
            model = torch.nn.DataParallel(model)  # Wrap the model for multi-GPU training
        elif torch.cuda.device_count() == 1:
            print('There is only 1 GPU... Loading model onto it\n')
        else:
            print("No GPU Detected!!!\n")
            
        model = model.to(device)
        print("Model loaded to the appropriate device...\n")
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)    
        print("Defined loss function and optimizer...\n")
        ###################### DEFINING THE PARAMETERS ################################################
        post_label = AsDiscrete(to_onehot=args.num_classes)
        post_pred = AsDiscrete(argmax=True, to_onehot=args.num_classes)
        trainer = Trainer(model=model,
                          loss_function=loss_function,
                          optimizer=optimizer,
                          max_iterations=args.max_iterations,
                          eval_num=args.eval_num,
                          saved_model_dir=args.saved_model_dir,
                          device=device,
                          patch_size=args.patch_size,
                          num_samples=args.num_samples,  # Include missing argument
                          decollate_batch=decollate_batch,  # Include missing argument
                          post_label=post_label,  # Include missing argument
                          post_pred=post_pred,  # Include missing argument
                          dice_metric=dice_metric)
        global_step = 0
        dice_val_best = 0.0
        global_step_best = 0
        while global_step < args.max_iterations:
            global_step, dice_val_best, global_step_best = trainer.train(global_step, train_loader, val_loader, dice_val_best, global_step_best, fold)            
        print(f"Global Step: {global_step} | Best Dice: {dice_val_best} | Global Best: {global_step_best}")
    
        print('_'*100)
if __name__ == "__main__":
    main()
    
    
