import cv2
import numpy as np
import os

def generate_ascii():
    crop_path = 'debug_repro/crop_bottom_left.png'
    if not os.path.exists(crop_path):
        print(f"Error: {crop_path} not found.")
        return
        
    img = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    print(f"Crop shape: {h}x{w}")
    
    # Threshold to keep only reasonably bright pixels (text, icons)
    _, thresh = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
    
    # Downsample for ASCII representation
    # Target size: 80 columns, 40 rows
    target_w = 100
    target_h = 50
    downsampled = cv2.resize(thresh, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    chars = " .:-=+*#%@"
    ascii_rows = []
    for r in range(target_h):
        row_str = ""
        for c in range(target_w):
            val = downsampled[r, c]
            char_idx = int(val / 255.0 * (len(chars) - 1))
            row_str += chars[char_idx]
        ascii_rows.append(row_str)
        
    print("\n--- ASCII Art of Bottom-Left Crop (100x50) ---")
    for r, row in enumerate(ascii_rows):
        # Print y coordinate on the left
        y_coord = 800 + int(r / target_h * h)
        print(f"{y_coord:4d} | {row}")
        
if __name__ == "__main__":
    generate_ascii()
