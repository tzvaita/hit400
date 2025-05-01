import cv2

# Load an example image
image = cv2.imread('path/to/your/image.jpg')
# Resize for consistency (e.g., width=640)
resized_image = cv2.resize(image, (640, int(image.shape[0] * 640 / image.shape[1])))
cv2.imshow("Resized Image", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
