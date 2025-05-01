import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# -------------------------------
# Step 1: Load and preprocess the image
# -------------------------------
# Specify the path to your image file (update accordingly)
image_path = './images/b.png'
img = cv2.imread(image_path)
if img is None:
    raise Exception("Image not found. Please check the image path.")

# Convert image from BGR (OpenCV default) to RGB for correct processing and display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# -------------------------------
# Step 2: Compute the Excess Green Index (ExG)
# -------------------------------
# Split the RGB channels
R, G, B = cv2.split(img_rgb)

# Compute the Excess Green Index: ExG = 2*G - R - B
exg = 2 * G.astype(np.float32) - R.astype(np.float32) - B.astype(np.float32)

# Optional: Print the range of ExG values for reference
print("ExG range: min =", np.min(exg), "max =", np.max(exg))

# -------------------------------
# Step 3: Define more bins for categorization
# -------------------------------
# Here we define 5 bins. You can adjust these thresholds based on the distribution of ExG values in your image.
# For example, we can divide the range into 5 equal intervals.
min_exg = np.min(exg)
max_exg = np.max(exg)
# Create 6 edges for 5 bins (we add 1 to include the max value)
bins = np.linspace(min_exg, max_exg, num=6)
print("Bins:", bins)

# Digitize the ExG values into bin indices (0 through 4)
exg_binned = np.digitize(exg, bins) - 1  # Subtract 1 so that bin indices start at 0

# -------------------------------
# Step 4: Create a custom colormap
# -------------------------------
# We want to have a gradient such that:
#   - The greenest (highest ExG) areas are displayed in dark green
#   - Then lighter greens, then yellow, and red for the least green areas
# Define a discrete colormap with 5 colors:
# For example:
#   Bin 0 (lowest values): red (poor vegetation)
#   Bin 1: yellow
#   Bin 2: light green
#   Bin 3: green
#   Bin 4 (highest values): dark green (very healthy vegetation)
colors = ['red', 'yellow', 'lightgreen', 'green', 'darkgreen']
cmap = mcolors.ListedColormap(colors)
# Set up boundaries: We have 5 bins, so boundaries 0,1,2,3,4,5
boundaries = np.arange(0, 6, 1)
norm = mcolors.BoundaryNorm(boundaries, cmap.N)

# -------------------------------
# Step 5: Plot the Heatmap
# -------------------------------
plt.figure(figsize=(12, 6))
plt.title("Excess Green Index (ExG) Heatmap")

# Display the binned ExG data as a heatmap
img_plot = plt.imshow(exg_binned, cmap=cmap, norm=norm)
plt.xlabel("Image Width (pixels)")
plt.ylabel("Image Height (pixels)")

# Create a color bar with appropriate tick labels (center of each bin)
cbar = plt.colorbar(img_plot, ticks=[0.5, 1.5, 2.5, 3.5, 4.5])
cbar.ax.set_yticklabels(['Very Low', 'Low', 'Moderate', 'High', 'Very High'])
plt.show()

# -------------------------------
# Optional: Save the heatmap to a file
# -------------------------------
plt.savefig("heatmap.png")
print("Heatmap saved as exg.png")
