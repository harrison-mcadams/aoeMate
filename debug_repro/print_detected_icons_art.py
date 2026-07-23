import cv2
import numpy as np
import os

def print_icon(name):
    path = f"debug_repro/{name}_detected.png"
    if not os.path.exists(path):
        print(f"{name} not found.")
        return
        
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # Resize to 20x20 for display
    img_resized = cv2.resize(img, (30, 20), interpolation=cv2.INTER_AREA)
    
    chars = " .:-=+*#%@"
    print(f"\n=== Detected {name.upper()} Icon (30x20) ===")
    for r in range(20):
        row = ""
        for c in range(30):
            val = img_resized[r, c]
            char_idx = int(val / 255.0 * (len(chars) - 1))
            row += chars[char_idx]
        print(row)

def main():
    for name in ['food', 'wood', 'gold', 'stone', 'silver']:
        print_icon(name)

if __name__ == "__main__":
    main()
