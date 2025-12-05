
import os
from PIL import Image
import numpy as np
import cv2

def scan_vertical():
    img_path = os.path.expanduser("~/Desktop/debug_screenshot_new.png")
    try:
        img = Image.open(img_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Convert to grayscale numpy
    img_gray = np.array(img.convert('L'))
    
    # Scan column x=14 to x=44 (30px width)
    # y from 100 to 600
    x_start = 14
    x_end = 44
    
    col = img_gray[100:600, x_start:x_end]
    
    # Calculate variance or edge density per row (or sliding window of 30px height)
    # We want to find 30x30 blocks with high activity.
    
    variances = []
    for y in range(0, 500 - 30): # 500 is height of col
        window = col[y:y+30, :]
        var = np.var(window)
        variances.append(var)
        
    # Find peaks
    variances = np.array(variances)
    # Simple peak finding: value > threshold and local max
    threshold = 200 # Arbitrary
    peaks = []
    
    for y in range(1, len(variances)-1):
        if variances[y] > threshold and variances[y] > variances[y-1] and variances[y] > variances[y+1]:
            # Filter close peaks
            if not peaks or (y - peaks[-1][0] > 20):
                peaks.append((y, variances[y]))
                
    print("Peaks found (y relative to 100):")
    for y, var in peaks:
        abs_y = 100 + y
        print(f"y={abs_y}, var={var:.2f}")

if __name__ == "__main__":
    scan_vertical()
