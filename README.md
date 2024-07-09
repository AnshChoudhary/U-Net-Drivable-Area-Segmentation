# Drivable Area Semantic Segmentation with U-Net

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
Include images or visualizations of your model predictions, evaluation metrics (IoU, accuracy, F1-score), and any relevant analysis of results.

