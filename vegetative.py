import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image (ensure the image is in BGR format as loaded by OpenCV)
img = cv2.imread('./images/mk.jpg')
if img is None:
    raise Exception("Image not found. Please check the image path.")

# Optionally, resize or pre-process the image if needed
img = cv2.resize(img, (640, 480))  # example resize

# Convert image to float for precise computation and split into channels
img_float = img.astype(np.float32)
B, G, R = cv2.split(img_float)

# Calculate Excess Green Index: ExG = 2*G - R - B
exg = 2 * G - R - B

# Normalize ExG for visualization (optional)
exg_norm = cv2.normalize(exg, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
exg_norm = exg_norm.astype(np.uint8)

# Display using OpenCV windows (or matplotlib)
cv2.imshow("Original Image", img)
cv2.imshow("Excess Green Index", exg_norm)
cv2.waitKey(10000)
cv2.destroyAllWindows()

# Alternatively, display using Matplotlib:
plt.figure(figsize=(8, 6))
plt.imshow(exg_norm, cmap='YlGn')
plt.title("Excess Green Index (ExG)")
plt.colorbar(label='ExG Intensity')
# plt.show()
plt.savefig("exg_output_one.png")
print("Figure saved as exg_output.png")

# Compute summary metric: average ExG (only consider canopy regions if you have a mask)
mean_exg = np.mean(exg)
print(f"Mean Excess Green Index: {mean_exg:.2f}")
