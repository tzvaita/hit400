# stage5_unet.py
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from torchvision import transforms
import os

# Load image and preprocess
img = cv2.imread('./images/b.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Resize or crop as needed for the model (example: 512x512)
img_resized = cv2.resize(img_rgb, (512, 512))
# Convert to tensor and normalize (match encoder's training stats)
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                         std=[0.229, 0.224, 0.225])
])
input_tensor = preprocess(img_resized).unsqueeze(0)  # shape [1,3,H,W]

# Load pretrained U-Net with ResNet34 encoder (trained on ImageNet)&#8203;:contentReference[oaicite:13]{index=13}
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", classes=1, activation=None)
# If you have a trained model file, load it:
# model.load_state_dict(torch.load("unet_resnet34.pth"))
model.eval()

# Run inference
with torch.no_grad():
    output = model(input_tensor)
    output_np = output.squeeze().numpy()

# Convert output to binary mask (threshold at 0)
veg_mask = (output_np > 0).astype(np.uint8) * 255

# Resize mask back to original size and save
veg_mask = cv2.resize(veg_mask, (img.shape[1], img.shape[0]))
cv2.imwrite('output/unet_mask.png', veg_mask)
print("Stage5 complete: Saved U-Net vegetation mask (output/unet_mask.png).")
