import cv2
import numpy as np
import os

def inspect_gold():
    img_path = 'debug_verification_resources/input_capture.png'
    if not os.path.exists(img_path):
        print("Error: Capture not found.")
        return
        
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Gold is around y=1050, x=35. Let's crop y=1035..1065, x=15..55
    crop = img[1030:1070, 15:55]
    h, w = crop.shape
    
    chars = " .:-=+*#%@"
    print("\n--- Gold Icon Area (40x40) ---")
    for r in range(h):
        row_str = ""
        for c in range(w):
            val = crop[r, c]
            char_idx = int(val / 255.0 * (len(chars) - 1))
            row_str += chars[char_idx]
        print(f"y={1030+r:4d} | {row_str}")

if __name__ == "__main__":
    inspect_gold()
