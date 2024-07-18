
import SimpleITK as sitk
from pathlib import Path
import torch
def save_volumes(test_img, test_label, test_outputs, vol_name, results_dir):
    """
    Save the test image, label, and predicted output as NIfTI files.

    Args:
        test_img : The test image tensor.
        test_label : The ground truth label tensor.
        test_outputs : The predicted output tensor.
        vol_name (str): The volume name used for saving the files.
        results_dir (str or Path): The directory where the results will be saved.
    """

    # Convert results_dir to Path if it's not already
    results_dir = Path(results_dir)

    # Ensure results directory exists
    results_dir.mkdir(parents=True, exist_ok=True)

    # Prepare image data for saving
    img = test_img.detach().cpu().squeeze().permute(2, 1, 0)
    img_sitk = sitk.GetImageFromArray(img.numpy())
    img_sitk.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(img_sitk, results_dir / f"{vol_name}.nii.gz")

    # Prepare label data for saving
    label = test_label.detach().cpu().squeeze().permute(2, 1, 0)
    label_sitk = sitk.GetImageFromArray(label.numpy())
    label_sitk.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(label_sitk, results_dir / f"{vol_name}_original.nii.gz")

    # Prepare predicted label data for saving
    pred_label = torch.argmax(test_outputs, dim=1).detach().cpu().squeeze().permute(2, 1, 0)
    pred_sitk = sitk.GetImageFromArray(pred_label.numpy())
    pred_sitk.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(pred_sitk, results_dir / f"{vol_name}_predicted.nii.gz")

    # Confirmation message
    print(f"Results for {vol_name} saved successfully!")
