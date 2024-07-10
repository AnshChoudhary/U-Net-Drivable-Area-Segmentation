# Drivable Area Semantic Segmentation with U-Net

![DBSCAN+Polygon](https://github.com/AnshChoudhary/U-Net-Drivable-Area-Segmentation/blob/main/results/polygon_mask.png)

This repository contains the implementation of a U-Net model for drivable area semantic segmentation using the BDD100K dataset. The model is trained on 70,000 images and their corresponding colormap masks for drivable area. The model was trained on NVIDIA RTX A6000 with a VRAM of 48GB. The training lasts approximately 10 hours on the mentioned specs. 

## Overview

- **`bdd100k_dataset.py`**: Dataset class (`BDD100KDataset`) for loading images and masks from the BDD100K dataset.
- **`traintorch3.py`**: Script for training the U-Net model on the dataset.
- **`eval.py`**: Script for evaluating the trained model on a validation set.
- **`requirements.txt`**: List of Python dependencies required to run the scripts.
- **`README.md`**: Detailed description of the project, instructions for usage, and results.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- Torchvision
- NumPy
- tqdm
- scikit-learn
- Albumentations

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/drivable-area-segmentation.git
   cd drivable-area-segmentation
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. Training: Modify train.py to specify your dataset directories (train_image_dir, train_mask_dir) and model parameters.
  If you are training on CUDA, you have to specify which GPU you want the model to be trained on. Run the training script:
    ```bash
    CUDA_VISIBLE_DEVICES=3 nohup python traintorch3.py
    ```
2. Evaluation: Modify eval.py to specify your validation dataset directory (val_image_dir, val_mask_dir) and load the trained model.
   Run the evaluation script:
    ```bash
    python eval.py
    ```

## Results
The predicted mask is generated from the model and compared to the ground truth mask in the following:

![Predicted Mask](https://github.com/AnshChoudhary/U-Net-Drivable-Area-Segmentation/blob/main/results/unet3_0a0a0b1a-7c39d841.png)

The result is vastly accurate in generating the mask for drivable area but looks extremely unstable and there aren't any defined boundaries. To overcome the above problems, the predicted mask was initially passed through some post processing such as the following:
1. Morphological Cleanup: This helps to remove small noise and fill small holes in each frame's segmentation.
2. Optical Flow Stabilization: This uses the movement between frames to create a more stable segmentation, reducing flickering.
3. Temporal Smoothing: This averages the segmentation over several frames, further reducing fluctuations.
4. Final Gaussian Blur: This provides a last bit of smoothing to the output. 

The result of the postprocessed mask looked like the following:

![PostProcessed](https://github.com/AnshChoudhary/U-Net-Drivable-Area-Segmentation/blob/main/results/pp.png?raw=true)

The Post Processed Mask is still not the best solution to the problem at hand. Therefore, we apply DBSCAN to the form clusters based on the density of the pixels in the predicted masks and then create polygons using coutour and convex hulls that would fit the clusters. Clusters are selected on the basis of pixels belonging to each cluster (20 pixels minimum) and then polygons are drawn to fit the cluster with a maximum of 6 edges. 

The result of DBSCAN + Polygon Fitting is depicted in the following:
![DBSCAN+Polygon](https://github.com/AnshChoudhary/U-Net-Drivable-Area-Segmentation/blob/main/results/polygon_mask.png)

