#!/bin/bash

# Directories
ver="CIS_UNet"
data_dir="../data"
saved_model_dir="./saved_models/$ver"

# Training Parameters
num_classes=15
max_iterations=30000
eval_num=500
num_samples=4
num_folds=4
patch_size=128
spatial_dims=3
in_channels=1
block_inplanes=(64 128 256 512)
encoder_channels=(64 ${block_inplanes[0]} ${block_inplanes[1]} ${block_inplanes[2]})
feature_size=48
norm_name='instance'

# Run the training script with the specified parameters
# export CUDA_LAUNCH_BLOCKING=1
python segmentation_pipeline.py \
    --data_dir "$data_dir" \
    --saved_model_dir "$saved_model_dir" \
    --num_classes "$num_classes" \
    --max_iterations "$max_iterations" \
    --eval_num "$eval_num" \
    --num_samples "$num_samples" \
    --num_folds "$num_folds" \
    --patch_size "$patch_size" \
    --spatial_dims "$spatial_dims" \
    --in_channels "$in_channels" \
    --encoder_channels "${encoder_channels[@]}" \
    --feature_size "$feature_size" \
    --norm_name "$norm_name"