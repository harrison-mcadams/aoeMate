import cv2
import numpy as np
import os
from PIL import Image

def check():
    img_path = 'debug_verification_resources/input_capture.png'
    if not os.path.exists(img_path):
        print("Error: Capture not found.")
        return
        
    img = Image.open(img_path)
    ss_gray = np.array(img.convert('L'), dtype=np.float32)
    
    crops = {
        'food': 979,
        'wood': 1037,
        'gold': 1075,
        'stone': 1115,
        'silver': 1154
    }
    
    scale = 1.5
    print(f"--- Checking Historical Log Coordinates at Scale {scale} ---")
    
    for name, cy in crops.items():
        template_path = f"templates/{name}.png"
        k = Image.open(template_path)
        k_gray = np.array(k.convert('L'), dtype=np.float32)
        k_scaled = cv2.resize(k_gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        kh, kw = k_scaled.shape
        
        # Match template in a tiny window around cy and x=43
        y1 = max(0, cy - 5)
        y2 = min(ss_gray.shape[0], cy + kh + 5)
        x1 = 15
        x2 = 85
        
        strip = ss_gray[y1:y2, x1:x2]
        res = cv2.matchTemplate(strip, k_scaled, cv2.TM_CCOEFF_NORMED)
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        abs_x = x1 + max_loc[0]
        abs_y = y1 + max_loc[1]
        
        print(f"  {name:6s} | max score = {max_val:.4f} at x={abs_x}, y={abs_y} (Expected: x=43, y={cy})")

if __name__ == "__main__":
    check()
