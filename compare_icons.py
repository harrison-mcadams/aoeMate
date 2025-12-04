
import cv2
import numpy as np
import os

def compare(name, extracted_fname, original_fname):
    ext_path = os.path.expanduser(f"~/Desktop/{extracted_fname}")
    orig_path = os.path.expanduser(f"~/Desktop/{original_fname}")
    
    if not os.path.exists(ext_path) or not os.path.exists(orig_path):
        print(f"Missing files for {name}")
        return

    ext = cv2.imread(ext_path, cv2.IMREAD_GRAYSCALE)
    orig = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize ext to match orig if needed
    if ext.shape != orig.shape:
        ext = cv2.resize(ext, (orig.shape[1], orig.shape[0]))
        
    res = cv2.matchTemplate(orig, ext, cv2.TM_CCOEFF_NORMED)
    print(f"{name}: Correlation = {res[0][0]:.4f}")

compare('food', 'food_extracted.png', 'food_icon.png')
compare('wood', 'wood_extracted.png', 'wood_icon.png')
compare('stone', 'stone_extracted.png', 'stone_icon.png')
