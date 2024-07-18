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
5. [Interactive Code](#interactive)
6. [Dependencies](#dependencies)
7. [Citations](#citations)


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
├── InteractiveCodes/                            # Jupyter notebook containing the script for executing the training of the model.
│      ├── Training.ipynb                        # Jupyter notebook containing the code to generate segmentation files using the trained models and to produce the metrics (DCS and MSD)
|      ├── Prediction_and_Evaluation.ipynb       # Jupyter notebook containing the code to generate segmentation files using the trained models and to produce the metrics (DCS and MSD)
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


