<div align=center> <h1>
  <img align="left" width="190" height="150" src="assets/OriginalLogo.png" alt="CIS-UNet: Multi-Class Segmentation of the Aorta in  Computed Tomography Angiography via Context-Aware Shifted Window Self-Attention">
  CIS-UNet: Multi-Class Segmentation of the Aorta in  Computed Tomography Angiography via Context-Aware Shifted Window Self-Attention</h1>

Welcome to the repository containing the code and dataset for **CIS-UNet**, a deep learning model designed for accurate 3D segmentation of the aorta and its branches.
  
[![](https://img.shields.io/badge/Imran-gray?logo=github&logoColor=white&label=Muhammad&labelColor=darkgreen&color=red)](https://www.linkedin.com/in/imrannust/) &emsp;
[![](https://img.shields.io/badge/Jonathan-gray?logo=world%20health%20organization&logoColor=white&label=Krebs&labelColor=darkblue&color=limegreen)](https://surgery.med.ufl.edu/profile/krebs-jonathan/) &emsp;
[![](https://img.shields.io/badge/Gopu-gray?logo=linkedin&logoColor=white&label=Veera&labelColor=black&color=yellow)](https://www.linkedin.com/in/veera-rajasekhar-reddy-gopu-3107361a6/?originalSubdomain=in) &emsp;
[![](https://img.shields.io/badge/Fazzone-gray?logo=world%20health%20organization&logoColor=white&label=Brian&labelColor=darkred&color=cyan)](https://surgery.med.ufl.edu/profile/fazzone-brian/) &emsp;
[![](https://img.shields.io/badge/Balaji%20Sivaraman-gray?logo=linkedin&logoColor=white&label=Vishal&labelColor=darkgreen&color=orange)](https://www.linkedin.com/in/vishal-balaji-sivaraman-ab86a7294/) &emsp;
[![](https://img.shields.io/badge/Kumar-gray?logo=linkedin&logoColor=white&label=Amarjeet&labelColor=darkpurple&color=lime)](https://www.linkedin.com/in/amar-ufl/) &emsp;
[![](https://img.shields.io/badge/Viscardi-gray?logo=world%20health%20organization&logoColor=white&label=Chelsea&labelColor=darkslategray&color=fuchsia)](https://surgery.med.ufl.edu/profile/viscardi-chelsea/) &emsp;
[![](https://img.shields.io/badge/Heithaus-gray?logo=world%20health%20organization&logoColor=white&label=Robert&labelColor=darkolivegreen&color=purple)](https://www.orlandohealth.com/physician-finder/robert-e-heithaus-md#/overview) &emsp;
[![](https://img.shields.io/badge/Shickel-gray?logo=linkedin&logoColor=white&label=Benjamin&labelColor=navy&color=orange)](https://www.linkedin.com/in/benjamin-shickel-804976ab/) &emsp;
[![](https://img.shields.io/badge/Zhou-gray?logo=github&logoColor=white&label=Yuyin&labelColor=darkorange&color=blue)](https://yuyinzhou.github.io/) &emsp;
[![](https://img.shields.io/badge/Cooper-gray?logo=world%20health%20organization&logoColor=white&label=Michol&labelColor=darkcyan&color=magenta)](https://surgery.med.ufl.edu/profile/cooper-michol/) &emsp;
[![](https://img.shields.io/badge/Shao-gray?logo=linkedin&logoColor=white&label=Wei&labelColor=darkviolet&color=teal)](https://www.linkedin.com/in/wei-shao-438782115/)

</div>

---

## Repository Contents

<img align="right" width="400" height="350" src="assets/LabelAnnotationDemo_v3.gif" alt="Demo of Aortic Branches">

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Dataset Detail](#dataset-detail)
4. [Scripted Code](#scripted)
    1. [Training From The Shell](#shell_training)
    2. [Prediction and Evaluation From the Shell](#shell_prediction)
6. [Interactive Code](#interactive)
7. [Dependencies](#dependencies)
8. [Citations](#citations)



## Overview <a id="overview"></a>

<div align=justify>
Aortic segmentation is crucial for minimally invasive treatments of aortic diseases.  Inaccurate segmentation can lead to errors in surgical planning and endograft construction. Previous methods treated aortic segmentation as a binary image segmentation problem, neglecting the need to distinguish individual aortic branches.

CIS-UNet addresses this limitation by performing multi-class segmentation of the aorta and thirteen aortic branches. It combines the strengths of convolutional neural networks (CNNs) and Swin transformers, resulting in a hierarchical encoder-decoder architecture with skip connections. Notably, CIS-UNet introduces a novel Context-aware Shifted Window Self-Attention (CSW-SA) block that enhances feature representation by capturing long-range dependencies between pixels.
</div>
<div align=center>
<img src="assets/CIS_UNet_Architecture.png" alt="CIS UNet Architecture">
</div>

## Directory Structure <a id="directory-structure"></a>

<h3> 1. Clone the Repository:</h3>

  Open your terminal or command prompt and clone the project directory as follows:
  ```
  git clone https://github.com/mirthAI/CIS-UNet.git
  ```

<h3> 2. Navigate to the Directory: </h3>

  Once the repository is cloned, navigate to the desired directory using the `cd` command as follows:
  ```
  cd CIS-UNet
  ```
<h3> 3. Directory Structure of CIS-UNet </h3>


```
CIS-UNet/
├── data/                                        # Folder containing the data
|     ├── Volumes/
|     │     ├── Subject001_CTA.nii.gz            # Input CTA image
|     │     ├── Subject002_CTA.nii.gz            # Input CTA image
|     │     │     
|     │     └── ...                              #(similar structure for other data samples)
|     |
|     └── Labels/
|           ├── Subject001_CTA_Label.nii.gz      # Input Segmentation
|           ├── Subject002_CTA_Label.nii.gz      # Input Segmentation
|           |
|           └── ...                              # (similar structure for other data samples)  
│
├── InteractiveCodes/                            # Folder containing the Jupyternotebooks
│      ├── Training.ipynb                        # Jupyter notebook containing the script for executing the training of the model.
|      ├── Prediction_and_Evaluation.ipynb       # Jupyter notebook containing the code to generate segmentation files using the trained models and to produce the metrics (DCS and MSD)
|      ├── utils/                                # Folder containing utility functions used during training
|      |     ├── init.py                         # Empty file to mark utils as a Python package
|      |     ├── CIS_UNet.py                     # CIS_UNet model definition
|      |     ├── dataset_processor.py            # Python script containing functions for loading and processing data
|      |     └── training_validation.py          # Python script containing functions for training the model
|      ├── results                               # The directory where the segmenation files and the computed metrics will be saved.
|      └── saved_models                          # The directory where your best trained models for each fold will be saved.
│
└── ScriptedCodes/                               # Folder for scripted Jupyter notebooks
       ├── run_training.sh                       # Bash script to execute training process
       ├── segmentation_pipeline.py              # Python script containing the core training logic
       ├── predict_and_evaluate.py               # Python script to generate segmentation files using the trained models and to produce the metrics (DCS and MSD)
       ├── run_prediction_and_evaluation.sh      # Bash script to execute prediction and evaluation processes
       ├── utils/                                # Folder containing utility functions used during training
       |     ├── init.py                         # Empty file to mark utils as a Python package
       |     ├── CIS_UNet.py                     # CIS_UNet model definition
       |     ├── dataset_processor.py            # Python script containing functions for loading and processing data
       |     └── training_validation.py          # Python script containing functions for training the model
       ├── results                               # The directory where the segmenation files and the computed metrics will be saved.
       └── saved_models                          # The directory where your best trained models for each fold will be saved.                                                 

```


## Dataset Detail <a id="dataset-detail"></a>

<div align=justify>

<img align="right" width="600" height="500" src="assets/LabelAnnotation.png" alt="Aorta and Its Branches">
  
**Our dataset consists of 59 CTA images**, each with an axial size of **512×512 pixels** and an isotropic in-plane resolution ranging from **0.759 mm to 1.007 mm**, with an average of **0.875 mm**. The number of axial slices varies between **347 and 962**, with a mean of **734 slices**. The axial slice thickness ranges from **0.8 mm to 2 mm**, averaging **0.969 mm**. 

In addition to the imaging data, our dataset includes 59 3D scorresponding egmentation volumes, containing the **annotations for thirteen vascular branches**. These branches include the Aorta, Innominate Artery (IA), Right Subclavian Artery, Right Common Carotid Artery, Left Common Carotid Artery (LCC), Left Subclavian Artery (LSA), Celiac Artery (CA), Superior Mesenteric Artery (SMA), Left Renal Artery (LRA), Right Renal Artery (RRA), Left Common Iliac Artery (LCIA), Left External Iliac Artery (LEIA), Left Internal Iliac Artery (LIIA), Right Common Iliac Artery (RCIA), Right External Iliac Artery (REIA), and Right Internal Iliac Artery (RIIA).


To expedite model training, we resampled the volumes to a uniform spacing of **1.5 mm×1.5 mm×1.5 mm**. We used the **"[RandCropByPosNegLabeld](https://docs.monai.io/en/stable/transforms.html)" function from the [MONAI](https://monai.io/) library** to facilitate random cropping of a fixed-size region from a large 3D image. The cropping center can be either a foreground or background voxel, determined by a specified foreground-to-background ratio. By leveraging this function, we selected random **128×128×28 patches** from the resampled volumes for training, enhancing data diversity and mitigating overfitting.


<h3>Accessing the Dataset</h3>

To access the dataset, please participate in the **[AortaSeg24 Challenge](https://aortaseg24.grand-challenge.org/)** hosted on the Grand Challenge platform. Begin by visiting the challenge page and proceed to complete the **[Data Agreement Form](https://aortaseg24.grand-challenge.org/dataset-access-information/)**. Upon submission and approval, you will gain full access to the dataset. Please download the dataset and place it in the data directory. Ensure your final data directory matches the structure defined above. 

**Currently, the dataset is only accessible to participants. Once the challenge is over, it will be made accessible to the general audience.**

</div>

---


<div align=center> <h1> 
  <a id="scripted_code"></a>
  Scipted Code
</h1></div>

For those who prefer running scripts from the shell, follow these steps to train the model:

<h3> Training From the Shell </h3>


1. **Create an Environment:** Create a new virtual environment using `conda`.
   ```bash
   conda create --name CIS_UNet python=3.10
   ```
2. **Activate the Enviornment:** Activate the newly created environment.
   ```bash
   conda activate CIS_UNet
   ```
3. **Install Required Packages:** Install the necessary packages listed in the **[requirements.txt](https://github.com/ImranNust/CIS-UNet-Context-Infused-Swin-UNet-for-Aortic-Segmentation/blob/main/requirements.txt)** file.
   ```bash
   pip install -r requirements.txt
   ```
4. **Prepare the Dataset:** Prepare the dataset for the training of deep-learning based image registration network.
   - Navigate to the directory where the script is saved:
     ```bash
     cd ScriptedCodes
     ```
     
    - Now runt the following commands to prepare the dataset for image registration task:
    
      ```bash
      chmod +x ./run_data_preparation.sh
      ./run_data_preparation.sh   
      ```
      
   This will create two folders, `png_data` and `processed_png_data`, inside the `data` directory. The images inside `processed_png_data` will be used to train the networks; you may delete the `png_images` directory if desired.

6. **Train the Image Registration Network:**
   - Confirm that your current working directory is `ScriptedCodes`.
   - To initiate the training process for both the affine and deformable registration networks across six folds, execute the following commands:
        ```bash
        chmod +x ./run_training.sh
        ./run_training.sh
        ```
   - The script will automatically create a `saved_model` folder within the `ScriptedCodes` directory.
   - The training process will proceed, and the model achieving the lowest loss will be stored in the `saved_model` directory.
   
7. **Prediction and Evaluation:**
   - Verify that you are located within the `ScriptedCodes` directory.
   - Execute the following commands to commence the prediction and evaluation process:

     ```bash
     chmod +x ./run_prediction_and_evaluation.sh
     ./run_prediction_and_evaluation.sh
     ```
   - Upon successful execution, the results directory will be populated with the deformed registered images.
   - Concurrently, a comprehensive CSV file detailing the evaluation metrics will be generated. This file includes the **Dice coefficient**, **Hausdorff distance**, **Urethra distance**, and distances for **Landmark 1**, **Landmark 2**, and **Landmark 3**, along with the **average landmark distance**.

---
