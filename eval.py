import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score
from PIL import Image

# U-Net Model Definition
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

# Dataset Definition
class BDD100KDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '.png'))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        mask = torch.tensor(mask, dtype=torch.long)  # Ensure mask is of type Long
        # Ensure that the mask contains only valid class indices (0, 1, 2)
        mask = mask.clamp(min=0, max=2)  # This ensures that mask values are within [0, 2]

        return image, mask

# Define the device to run the evaluation on (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths to your validation data directories
val_image_dir = '/data/BDD100K/bdd100k/bdd_data/images/100k/val'
val_mask_dir = '/data/BDD100K/bdd100k/bdd_data/drivable_maps/labels/val'

# Transformation for validation images (assuming it's similar to training)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Initialize the dataset and data loader
val_dataset = BDD100KDataset(val_image_dir, val_mask_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Initialize the model and load the trained weights
model = UNet(n_channels=3, n_classes=3).to(device)  # Adjust n_classes according to your task
model.load_state_dict(torch.load('unet3_model.pth', map_location=device))
model.eval()

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

def compute_f1_score(pred, target, average='weighted'):
    return f1_score(target.cpu().numpy().flatten(), pred.cpu().numpy().flatten(), average=average)

# Evaluation loop
model.eval()
total_iou = []
total_accuracy = []
total_f1 = []

with torch.no_grad():
    for images, masks in tqdm(val_loader):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        ious = compute_iou(preds, masks, num_classes=3)  # Adjust num_classes according to your task
        accuracy = compute_accuracy(preds, masks)
        f1 = compute_f1_score(preds, masks)

        total_iou.append(ious)
        total_accuracy.append(accuracy)
        total_f1.append(f1)

        # Optionally, print scores for each image
        print(f"Image {val_loader.dataset.images[val_loader.dataset.indices[0]]}: IoU: {ious}, Accuracy: {accuracy}, F1-score: {f1}")

# Compute average scores across all images
avg_iou = np.nanmean(total_iou, axis=0)  # Average IoU per class
avg_accuracy = np.mean(total_accuracy)   # Average accuracy
avg_f1 = np.mean(total_f1)               # Average F1-score

print(f'Average IoU: {avg_iou}')
print(f'Average Accuracy: {avg_accuracy}')
print(f'Average F1-score: {avg_f1}')
