# vegetation_masking.py
# Input: RGB image file; optional HSV lower/upper bounds for vegetation.
# Output: Binary mask image of vegetation (saved or shown).
# Usage example: python vegetation_masking.py --image field.jpg --lower 35 50 50 --upper 85 255 255

import cv2
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Vegetation Masking with HSV thresholding')
    parser.add_argument('./images/b.png', required=True, help='Path to input RGB image')
    parser.add_argument('--lower', type=int, nargs=3, default=[35, 100, 50],
                        help='Lower HSV threshold (H S V)')
    parser.add_argument('--upper', type=int, nargs=3, default=[85, 255, 255],
                        help='Upper HSV threshold (H S V)')
    parser.add_argument('--save', default=None, help='Path to save mask image (optional)')
    return parser.parse_args()

def main():
    args = parse_args()
    # Load image (OpenCV loads BGR)
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Image not found: {args.image}")
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array(args.lower, dtype=np.uint8)
    upper = np.array(args.upper, dtype=np.uint8)
    # Threshold to get vegetation mask
    mask = cv2.inRange(hsv, lower, upper)
    # Morphological operations to clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Display results
    cv2.imshow('Vegetation Mask', mask_clean)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Optionally save mask
    if args.save:
        cv2.imwrite(args.save, mask_clean)
        print(f"Saved mask to {args.save}")

if __name__ == '__main__':
    main()
