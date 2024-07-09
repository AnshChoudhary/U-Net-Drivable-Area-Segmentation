import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2

# Define DoubleConv, Down, Up, OutConv, and UNet classes
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
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
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
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
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

# Define the model loading function
def load_model(model_path, device):
    model = UNet(n_channels=3, n_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Define the transformation function
def get_transform():
    return Compose([
        Resize(256, 256),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# Function to perform inference on a single frame
def infer_frame(model, frame, device, transform):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)
    image = augmented['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
        pred = pred.squeeze().cpu().numpy()
    
    return pred

# Function to overlay mask on the original frame
def overlay_mask(frame, mask):
    # Resize the mask to the original frame size
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    color_mask = np.zeros_like(frame)
    color_mask[mask_resized == 1] = [0, 255, 0]  # Green for drivable area

    overlayed_frame = cv2.addWeighted(frame, 1, color_mask, 0.5, 0)
    return overlayed_frame


# Main function to process the video
def process_video(input_video_path, output_video_path, model_path, device):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    model = load_model(model_path, device)
    model.eval()

    transform = Compose([
        Resize(256, 256),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess the frame
            augmented = transform(image=frame)
            image = augmented['image'].unsqueeze(0).to(device)

            # Run inference
            output = model(image)
            mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # Overlay the mask on the original frame
            overlayed_frame = overlay_mask(frame, mask)
            
            out.write(overlayed_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Paths for the input video, output video, and model

if __name__ == "__main__":
    input_video_path = '/Users/anshchoudhary/Downloads/u-net-torch/video.mp4'
    output_video_path = '/Users/anshchoudhary/Downloads/u-net-torch/output-video1.mp4'
    model_path = 'unet_model.pth'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    process_video(input_video_path, output_video_path, model_path, device)