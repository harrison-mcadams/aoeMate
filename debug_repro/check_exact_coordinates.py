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
        'food': 952,
        'wood': 1003,
        'gold': 1050,
        'stone': 1098,
        'silver': 1146
    }
    
    for scale in [1.5, 2.0]:
        print(f"\n================ Scale {scale} ================")
        # Load and scale kernels
        for name, cy in crops.items():
            template_path = f"templates/{name}.png"
            k = Image.open(template_path)
            k_gray = np.array(k.convert('L'), dtype=np.float32)
            k_scaled = cv2.resize(k_gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            kh, kw = k_scaled.shape
            
            # Match template in a horizontal strip around cy, x=20..45
            # We crop the screenshot around cy to avoid boundary errors
            y1 = max(0, cy - 5)
            y2 = min(ss_gray.shape[0], cy + kh + 5)
            x1 = 0
            x2 = 100
            
            strip = ss_gray[y1:y2, x1:x2]
            res = cv2.matchTemplate(strip, k_scaled, cv2.TM_CCOEFF_NORMED)
            
            # Find best match in x=20..45 (relative to strip)
            best_score = -1.0
            best_x = -1
            best_y_offset = -1
            
            for y_off in range(res.shape[0]):
                for x in range(20, min(res.shape[1], 45)):
                    score = res[y_off, x]
                    if score > best_score:
                        best_score = score
                        best_x = x
                        best_y_offset = y_off
                        
            abs_y = y1 + best_y_offset
            print(f"  {name:6s} | Best score = {best_score:.4f} at x={best_x}, y={abs_y}")

if __name__ == "__main__":
    check()
