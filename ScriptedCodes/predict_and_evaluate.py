import argparse
import os
import glob
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
from torch.utils.data import DataLoader
from monai.data import CacheDataset
from monai.inferers import sliding_window_inference
import SimpleITK as sitk
import seg_metrics.seg_metrics as sg
from sklearn.model_selection import KFold
from utils.dataset_processor import DatasetProcessor, image_and_masks_paths
from utils.CIS_UNet import CIS_UNet
from utils.save_volumes import save_volumes


# Define an argument parser to handle command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True, help="Path to the root directory of the dataset")
parser.add_argument("--saved_model_dir", type=str, required=True, help="Path to the root directory where the best model is saved")
parser.add_argument("--results_dir", type=str, required=True, help="Path where the results will be stored!")
parser.add_argument("--num_classes", type=int, default=24, help="Number of classes for segmentation")
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
    files = image_and_masks_paths(args.data_dir)
    processor = DatasetProcessor(args.data_dir)
    for fold, (train_indices, val_indices) in enumerate(skf.split(files)):
        print(f"Processing Fold {fold}")
        
        ####################### Preparing the Data ##################################
        val_files = [files[i] for i in val_indices]        
        val_images = [item['image'].split('/')[-1].split('.')[0] for item in val_files]
        print(f"Validation Images Names: {', '.join(val_images)}")
        val_transforms = processor.get_val_transforms()
        
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
        
        print(f"Loading the Weights of the Trained Model for Fold {fold}...\n")
        model_path = os.path.join(args.saved_model_dir,"Fold" + str(fold) + "_" + "best_metric_model.pth")
        print(f"Loading Model: {model_path}\n")
        state_dict = torch.load(model_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:] if k.startswith('module.') else k] = v
        model.load_state_dict(new_state_dict)
        ###########################################################################################        
        num_cpus = torch.get_num_threads()
        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=len(val_files), cache_rate=1.0, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        print("Dataset is loaded and prepared for validation...\n")

        # Create the results directory for the current fold
        result_dir = os.path.join(args.results_dir, f'Fold{fold}')
        os.makedirs(result_dir,exist_ok=True)

        ###################### DEFINING THE PARAMETERS ################################################
        # Evaluate the model
        model.eval()
        individual_dices = {}
        individual_surface_scores = {}
        mean_dice_coeff = []

        # Disable gradient computation
        with torch.no_grad():
            for i, batch1 in enumerate(val_loader):
                test_inputs, test_labels = batch1["image"].to(device), batch1["label"].to(device)
                test_outputs = sliding_window_inference(test_inputs, 
                                                        (args.patch_size, args.patch_size, args.patch_size), 
                                                        args.num_samples, model)

                file_path = val_ds[i]['image_meta_dict']['filename_or_obj']
                vol_name = os.path.basename(file_path).split('.')[0]
                print(f'Processing Volume: {vol_name}')
                
                # Save the volumes
                save_volumes(
                    test_img=test_inputs,
                    test_label=test_labels,
                    test_outputs=test_outputs,
                    vol_name=vol_name,
                    results_dir=result_dir
                )
                
        # Calculate metrics for each fold
        gdth_fpaths = sorted(glob.glob(os.path.join(result_dir, '*original.nii.gz')))
        pred_fpaths = sorted(glob.glob(os.path.join(result_dir, '*predicted.nii.gz')))
        labels_fpaths = [{"gdth_fpath": gdth_label, "pred_fpath": pred_label} for gdth_label, pred_label in zip(gdth_fpaths, pred_fpaths)]

        dice_results = {}
        msd_results = {}
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        segment_names = {
            0: "Aorta", 1: "Left Subclavian Artery", 2: "Celiac Artery",
            3: "SMA", 4: "Left Renal Artery", 5: "Right Renal Artery",
            6: "Left Common Iliac Artery", 7: "Right Common Iliac Artery",
            8: "Innominate Artery", 9: "Left Common Carotid", 10: "Right External Iliac Artery",
            11: "Right Internal Iliac Artery", 12: "Left External Iliac Artery",
            13: "Left Internal Iliac Artery"
        }
        
        # Compute metrics for each volume
        for label_fp in labels_fpaths:
            gdth_fpath = label_fp['gdth_fpath']
            pred_fpath = label_fp['pred_fpath']
            vol_name = os.path.basename(gdth_fpath).split("_")[0]

            # Read images and convert them to numpy arrays
            gdth_img = sitk.ReadImage(gdth_fpath)
            gdth_np = sitk.GetArrayFromImage(gdth_img)
            pred_img = sitk.ReadImage(pred_fpath)
            pred_np = sitk.GetArrayFromImage(pred_img)
            spacing = np.array(list(reversed(pred_img.GetSpacing())))

            print(f"Processing {vol_name} for metrics computation ...")

            # Calculate metrics
            metrics = sg.write_metrics(
                labels=labels,
                gdth_img=gdth_np,
                pred_img=pred_np,
                csv_file=None,
                spacing=spacing,
                metrics=['msd', 'dice']
            )

            dice_results[vol_name] = metrics[0]['dice']
            msd_results[vol_name] = metrics[0]['msd']
            
        # Save the metrics to CSV files
        df_msd = pd.DataFrame(msd_results).T
        df_msd["Labels' Avg"] = df_msd.mean(axis=1)
        df_msd.loc['Volume Avg'] = df_msd.mean(axis=0)
        df_msd = df_msd.rename(index=segment_names)
        df_msd.index.names = ['Segments']
        df_msd.to_csv(os.path.join(result_dir, "test_msd.csv"))

        df_dice = pd.DataFrame(dice_results).T
        df_dice["Labels' Avg"] = df_dice.mean(axis=1)
        df_dice.loc['Volume Avg'] = df_dice.mean(axis=0)
        df_dice = df_dice.rename(index=segment_names)
        df_dice.index.names = ['Segments']
        df_dice.to_csv(os.path.join(result_dir, "test_dice.csv"))

    
        print('_'*100)
if __name__ == "__main__":
    main()
