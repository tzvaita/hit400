# stage1_exg.py
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

# Load the image (BGR format in OpenCV)
img = cv2.imread('./images/b.png')
b, g, r = cv2.split(img.astype(np.float32))

# Compute Excess Green (ExG) index: ExG = 2*G - (R + B)&#8203;:contentReference[oaicite:1]{index=1}
exg = 2 * g - (r + b)

# Normalize ExG to [0,1] for display
exg_norm = (exg - exg.min()) / (exg.max() - exg.min() + 1e-6)

# Create a red-to-green colormap (custom gradient)
colors = ["red", "yellow", "green"]
cmap = LinearSegmentedColormap.from_list('RedGreen', colors, N=256)

# Apply colormap to ExG
heatmap = (cmap(exg_norm)[:, :, :3] * 255).astype(np.uint8)

# Prepare output directory
os.makedirs('output', exist_ok=True)
# Save heatmap image
cv2.imwrite('output/exg_heatmap.png', cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
# Also save raw ExG array for later stages
np.save('output/exg_array.npy', exg)

print("Stage1 complete: Saved ExG heatmap (output/exg_heatmap.png) and array (output/exg_array.npy).")
