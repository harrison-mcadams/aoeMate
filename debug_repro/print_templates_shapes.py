import cv2
import numpy as np
import os
from PIL import Image

def print_template(name):
    path = f"templates/{name}.png"
    if not os.path.exists(path):
        print(f"{name} not found.")
        return
        
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    
    chars = " .:-=+*#%@"
    print(f"\n=== Reference {name.upper()} Template ({w}x{h}) ===")
    for r in range(h):
        row = ""
        for c in range(w):
            val = img[r, c]
            char_idx = int(val / 255.0 * (len(chars) - 1))
            row += chars[char_idx]
        print(row)

def main():
    for name in ['food', 'wood', 'gold', 'stone', 'silver']:
        print_template(name)

if __name__ == "__main__":
    main()
