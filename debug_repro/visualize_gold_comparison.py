import cv2
import numpy as np
import os
from PIL import Image

def compare_gold():
    template_path = 'templates/gold.png'
    crop_path = 'debug_repro/gold_large.png'
    
    if not os.path.exists(template_path) or not os.path.exists(crop_path):
        print("Error: Template or crop not found.")
        return
        
    k = Image.open(template_path)
    k_gray = np.array(k.convert('L'), dtype=np.float32)
    
    # Scale template to 1.5
    scale = 1.5
    k_scaled = cv2.resize(k_gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    kh, kw = k_scaled.shape
    
    # Load gold crop (which is BGR, convert to grayscale)
    crop = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
    
    # We want to match template in the crop to find the exact location of the icon
    res = cv2.matchTemplate(crop.astype(np.float32), k_scaled, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    print(f"Max match score: {max_val:.4f} at {max_loc}")
    
    # Extract the matched region from the crop
    x, y = max_loc
    matched_region = crop[y:y+kh, x:x+kw]
    
    # Print side-by-side ASCII art of template vs matched region
    chars = " .:-=+*#%@"
    print("\n--- Side-by-Side Comparison (Template vs Crop Region) ---")
    print(f"Template size: {kw}x{kh}")
    for r in range(kh):
        row_tmpl = ""
        row_crop = ""
        for c in range(kw):
            val_t = k_scaled[r, c]
            char_idx_t = int(val_t / 255.0 * (len(chars) - 1))
            row_tmpl += chars[char_idx_t]
            
            val_c = matched_region[r, c]
            char_idx_c = int(val_c / 255.0 * (len(chars) - 1))
            row_crop += chars[char_idx_c]
        print(f"Row {r:2d} | {row_tmpl}   ||   {row_crop}")

if __name__ == "__main__":
    compare_gold()
