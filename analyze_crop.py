import cv2
import numpy as np
import os
from PIL import Image
import analyze_ss

def analyze_crop():
    img_path = os.path.expanduser("~/Desktop/debug_vill_crop_gold.png")
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found")
        return

    print(f"Analyzing {img_path}...")
    img = Image.open(img_path)
    img_gray = img.convert('L')
    arr = np.array(img_gray)
    
    print(f"Shape: {arr.shape}")
    print(f"Min val: {arr.min()}, Max val: {arr.max()}, Mean: {arr.mean()}")
    
    # Try thresholding using the same logic as _parse_number_from_region -> threshold_image
    # analyze_ss.threshold_image uses:
    # _, thresh = cv2.threshold(src, 190, 255, cv2.THRESH_BINARY)
    
    _, thresh = cv2.threshold(arr, 190, 255, cv2.THRESH_BINARY)
    non_zero = cv2.countNonZero(thresh)
    print(f"Pixels > 190: {non_zero}")
    
    if non_zero == 0:
        print("Crop is empty after thresholding! Threshold might be too high.")
        # Try lower threshold
        _, thresh_low = cv2.threshold(arr, 150, 255, cv2.THRESH_BINARY)
        print(f"Pixels > 150: {cv2.countNonZero(thresh_low)}")
        
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Contours found (thresh 190): {len(contours)}")
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        print(f"  Contour {i}: x={x}, y={y}, w={w}, h={h}, area={area}")

if __name__ == "__main__":
    analyze_crop()
