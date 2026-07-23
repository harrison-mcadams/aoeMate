import cv2
import numpy as np
import os
from PIL import Image

def analyze_file(path):
    if not os.path.exists(path):
        print(f"{path} not found.")
        return
    img = cv2.imread(path)
    h, w, c = img.shape
    print(f"\n--- Analysis of {path} ---")
    print(f"Dimensions: {w}x{h}")
    # Print average pixel values of the corner regions
    # Bottom-left corner (typically where resources are)
    bl_crop = img[max(0, h-300):h, 0:min(w, 300)]
    print(f"Bottom-left 300x300 average brightness: {np.mean(bl_crop):.2f}")
    
    # Save a crop of the bottom-left corner
    out_crop_path = f"debug_repro/bl_{os.path.basename(path)}"
    cv2.imwrite(out_crop_path, bl_crop)
    print(f"Saved bottom-left crop to {out_crop_path}")

def main():
    analyze_file('debug_bbox_check.png')
    analyze_file('debug_bbox_huge.png')
    analyze_file('debug_verification_resources/input_capture.png')

if __name__ == "__main__":
    main()
