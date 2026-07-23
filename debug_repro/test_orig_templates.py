import cv2
import numpy as np
import os
from PIL import Image

def check_orig():
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
    
    scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.8, 2.0]
    
    for scale in scales:
        print(f"\n================ Scale {scale:.2f} (templates_orig) ================")
        total_score = 0.0
        for name, cy in crops.items():
            template_path = f"templates_orig/{name}.png"
            if not os.path.exists(template_path):
                print(f"  {name} template not found in templates_orig.")
                continue
            k = Image.open(template_path)
            k_gray = np.array(k.convert('L'), dtype=np.float32)
            
            # Scale template
            if scale != 1.0:
                k_scaled = cv2.resize(k_gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            else:
                k_scaled = k_gray
                
            kh, kw = k_scaled.shape
            if kh > ss_gray.shape[0] or kw > ss_gray.shape[1]:
                continue
                
            # Match template in a horizontal strip around cy, x=15..45
            y1 = max(0, cy - 8)
            y2 = min(ss_gray.shape[0], cy + kh + 8)
            x1 = 0
            x2 = 60
            
            strip = ss_gray[y1:y2, x1:x2]
            res = cv2.matchTemplate(strip, k_scaled, cv2.TM_CCOEFF_NORMED)
            
            best_score = -1.0
            best_x = -1
            best_y_offset = -1
            for y_off in range(res.shape[0]):
                for x in range(15, min(res.shape[1], 45)):
                    score = res[y_off, x]
                    if score > best_score:
                        best_score = score
                        best_x = x
                        best_y_offset = y_off
                        
            abs_y = y1 + best_y_offset
            total_score += best_score
            print(f"  {name:6s} | Best score = {best_score:.4f} at x={best_x}, y={abs_y}")
        print(f"Average Score: {total_score / 5.0:.4f}")

if __name__ == "__main__":
    check_orig()
