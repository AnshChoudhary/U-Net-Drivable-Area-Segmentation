import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import os
# Define the DoubleConv, Down, Up, OutConv, and UNet classes as before
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
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
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

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
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
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

# Dataset class
class BDD100KDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

def get_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

# Evaluation and IoU computation functions
def compute_iou(pred, target, num_classes):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item() - intersection
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / max(union, 1))
    return np.array(ious)

def compute_accuracy(pred, target):
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total

def evaluate_model(model, dataloader, device, num_classes):
    model.eval()
    total_iou = []
    total_accuracy = []
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            ious = compute_iou(preds, masks, num_classes)
            accuracy = compute_accuracy(preds, masks)

            total_iou.append(ious)
            total_accuracy.append(accuracy)
    
    avg_iou = np.nanmean(total_iou, axis=0)  # Average IoU per class
    avg_accuracy = np.mean(total_accuracy)   # Average accuracy
    
    return avg_iou, avg_accuracy

# Load model and data
model = UNet(n_channels=3, n_classes=2)
model.load_state_dict(torch.load('unet_model.pth', map_location='cpu'))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

val_image_dir = '/Users/anshchoudhary/Downloads/u-net-torch/images'
val_mask_dir = '/Users/anshchoudhary/Downloads/u-net-torch/masks'

transform = get_transforms()
val_dataset = BDD100KDataset(val_image_dir, val_mask_dir, transform)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Evaluate the model
num_classes = 3
avg_iou, avg_accuracy = evaluate_model(model, val_loader, device, num_classes)
print(f'Average IoU: {avg_iou}')
print(f'Average Accuracy: {avg_accuracy}')

# Display the first image, its generated mask, and the ground truth mask
def display_prediction(model, image_path, mask_path, device):
    model.eval()
    transform = get_transforms()
    
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        _, pred_mask = torch.max(output, 1)
    
    pred_mask = pred_mask.squeeze().cpu().numpy()
    mask = np.array(mask)

    # Define colors
    colors = np.array([
        [0, 0, 0],      # Background
        [0, 255, 0],    # Drivable area
    ])

    # Create colored masks
    pred_colored_mask = colors[pred_mask]
    gt_colored_mask = colors[mask]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(gt_colored_mask)
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    axes[2].imshow(pred_colored_mask)
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')
    
    plt.show()

# Example usage
image_path = '/path/to/single/image.png'
mask_path = '/path/to/single/mask.png'
display_prediction(model, image_path, mask_path, device)
