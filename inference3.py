import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from PIL import Image

# Define your UNet model here (make sure it's the same as the one used during training)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# Transformations
def get_transforms():
    return Compose([
        Resize(256, 256),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# Load the model
device = torch.device('cpu')
model = UNet(n_channels=3, n_classes=3).to(device)
model_path = './unet3_model.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Inference function
def inference(model, image_path, transform):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]  # Keep the original image size
    augmented = transform(image=image)
    input_image = augmented['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_image)
        output = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        output = cv2.resize(output, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

    return image, output

# Display function
def display_results(image, predicted_mask, ground_truth_path):
    ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
    ground_truth = cv2.resize(ground_truth, (image.shape[1], image.shape[0]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[1].imshow(predicted_mask, cmap='jet')
    axes[1].set_title("Predicted Mask")
    axes[2].imshow(ground_truth, cmap='jet')
    axes[2].set_title("Ground Truth Mask")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Paths to the image and mask
image_path = '/Users/anshchoudhary/Downloads/u-net-torch/images/0a0a0b1a-7c39d841.jpg'
mask_path = '/Users/anshchoudhary/Downloads/u-net-torch/masks/0a0a0b1a-7c39d841.png'

# Perform inference
transform = get_transforms()
image, predicted_mask = inference(model, image_path, transform)

# Display results
display_results(image, predicted_mask, mask_path)
