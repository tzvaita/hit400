import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = './images/b.png'  # Update with the correct path
img = cv2.imread(image_path)
if img is None:
    raise Exception("Image not found. Please check the image path.")

# Convert image from BGR to RGB (for correct display with matplotlib)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert image from BGR to HSV for more effective color segmentation
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the green color range in HSV (adjust these values if needed)
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

# Create a mask that identifies "green" areas
mask = cv2.inRange(hsv, lower_green, upper_green)

# Calculate the Excess Green Index (ExG) using the RGB channels.
# Formula: ExG = 2*G - R - B
# Split the RGB channels from the original (or the converted) image
R, G, B = cv2.split(img_rgb)
exg = 2 * G.astype(np.float32) - R.astype(np.float32) - B.astype(np.float32)

# Optionally, you can mask out the non-green areas from the ExG (to focus on canopy),
# Here we set ExG to a minimal value where mask is zero.
exg_masked = np.where(mask > 0, exg, 0)

# Normalize ExG values for better visualization (optional)
exg_norm = cv2.normalize(exg_masked, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
exg_norm = exg_norm.astype(np.uint8)

# Display the heatmap using matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(exg_norm, cmap='YlGn')  # 'YlGn' is a greenish colormap suitable for vegetation
plt.title("Excess Green Index Heatmap")
plt.colorbar(label='Normalized ExG Value')
plt.show()

# Alternatively, if running in a headless environment, save the heatmap instead:
plt.savefig("exg_heatmap.png")
