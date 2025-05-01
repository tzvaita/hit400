# stage2_anomaly.py
import cv2
import numpy as np
import os

# Load the ExG array from Stage 1
exg = np.load('output/exg_array.npy')

# Create an anomaly mask: mark pixels with very low ExG (non-green)
# Here threshold=0 (no green); adjust if needed.
anomaly_mask = (exg < 0).astype(np.uint8) * 255

# Morphological operations to remove noise&#8203;:contentReference[oaicite:5]{index=5}
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
anomaly_clean = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel)
anomaly_clean = cv2.morphologyEx(anomaly_clean, cv2.MORPH_CLOSE, kernel)

# Save the anomaly mask image
cv2.imwrite('output/anomaly_mask.png', anomaly_clean)
print("Stage2 complete: Saved anomaly mask (output/anomaly_mask.png).")
