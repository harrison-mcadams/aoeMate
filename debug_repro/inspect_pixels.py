import cv2
import numpy as np
import os

def inspect():
    img_path = 'debug_verification_resources/input_capture.png'
    if not os.path.exists(img_path):
        print("Error: Capture not found.")
        return
        
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # We want to print the grid for x = 0..100, y = 940..1170
    # To make it readable, we downsample horizontally by 2 and vertically by 3
    # This gives us a grid of width 50 and height ~77.
    x_start, x_end = 0, 100
    y_start, y_end = 940, 1170
    
    region = img[y_start:y_end, x_start:x_end]
    h, w = region.shape
    
    # Downsample
    ds_w = 60
    ds_h = 46
    downsampled = cv2.resize(region, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
    
    print("\n--- High-Res Pixel Brightness Grid (60x46) ---")
    print("     " + "".join(f"{int((x_start + c * (x_end - x_start) / ds_w)/10)%10}" for c in range(ds_w)))
    print("     " + "".join(f"{int(x_start + c * (x_end - x_start) / ds_w)%10}" for c in range(ds_w)))
    print("     " + "-" * ds_w)
    
    for r in range(ds_h):
        y_val = y_start + int(r * (y_end - y_start) / ds_h)
        row_str = ""
        for c in range(ds_w):
            val = downsampled[r, c]
            # Map 0..255 to 0..9
            digit = int(val / 255.0 * 9.9)
            if digit == 0:
                row_str += " "
            else:
                row_str += str(digit)
        print(f"{y_val:4d} | {row_str}")

if __name__ == "__main__":
    inspect()
