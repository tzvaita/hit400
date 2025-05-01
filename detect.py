import cv2
import numpy as np

# Load image
img = cv2.imread('./images/b.png')
if img is None:
    raise Exception("Image not found. Please check the image path.")

# Convert image from BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the green color range (you might need to adjust these values)
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

# Create a mask for green colors
mask = cv2.inRange(hsv, lower_green, upper_green)

# Optional: refine mask using morphology (remove noise)
kernel = np.ones((5,5),np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Calculate canopy cover percentage
canopy_area = np.sum(mask > 0)
total_area = img.shape[0] * img.shape[1]
canopy_cover_percentage = (canopy_area / total_area) * 100
print(f"Canopy Cover Percentage: {canopy_cover_percentage:.2f}%")
cv2.imwrite("green_mask.jpg", mask)

# Display the mask and original image for comparison
cv2.imshow("Original Image", img)
cv2.imshow("Green Mask", mask)
cv2.waitKey(20000)
cv2.destroyAllWindows()

