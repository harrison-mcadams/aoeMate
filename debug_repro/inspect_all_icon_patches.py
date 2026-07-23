import cv2
import numpy as np
import os

def inspect_all():
    img_path = 'debug_verification_resources/input_capture.png'
    if not os.path.exists(img_path):
        print("Error: Capture not found.")
        return
        
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    crops = {
        'food': 952,
        'wood': 1003,
        'gold': 1050,
        'stone': 1098,
        'silver': 1146
    }
    
    chars = " .:-=+*#%@"
    
    for name, cy in crops.items():
        # Inspect 25 pixels above and below cy, x=15..55
        crop = img[cy-20:cy+20, 15:55]
        h, w = crop.shape
        print(f"\n--- {name.upper()} Icon Region (cy={cy}, shape={w}x{h}) ---")
        for r in range(h):
            row_str = ""
            for c in range(w):
                val = crop[r, c]
                char_idx = int(val / 255.0 * (len(chars) - 1))
                row_str += chars[char_idx]
            print(f"y={cy-20+r:4d} | {row_str}")

if __name__ == "__main__":
    inspect_all()
