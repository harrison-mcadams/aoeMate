import cv2
import numpy as np
import os
from PIL import Image

def compare():
    resources = ['food', 'wood', 'gold', 'stone', 'silver']
    scales = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    print("--- Comparing Detected Icons with Templates ---")
    for name in resources:
        crop_path = f"debug_repro/{name}_detected.png"
        template_path = f"templates/{name}.png"
        
        if not os.path.exists(crop_path) or not os.path.exists(template_path):
            print(f"  {name}: Crop or template not found.")
            continue
            
        crop_img = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
        template_img = Image.open(template_path)
        k_gray = np.array(template_img.convert('L'), dtype=np.float32)
        
        best_score = -1.0
        best_scale = 1.0
        best_loc = (0, 0)
        
        for scale in scales:
            if scale != 1.0:
                k_scaled = cv2.resize(k_gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            else:
                k_scaled = k_gray
                
            kh, kw = k_scaled.shape
            if kh > crop_img.shape[0] or kw > crop_img.shape[1]:
                continue
                
            res = cv2.matchTemplate(crop_img.astype(np.float32), k_scaled, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            if max_val > best_score:
                best_score = max_val
                best_scale = scale
                best_loc = max_loc
                
        # Get absolute coordinate in input_capture.png
        # The crop was centered at cx=35, cy=CY, which means it starts at cy-20, cx-20.
        # So absolute X = cx - 20 + best_loc[0]
        # Absolute Y = cy - 20 + best_loc[1]
        cy, cx = {
            'food': (952, 35),
            'wood': (1003, 35),
            'gold': (1050, 35),
            'stone': (1098, 35),
            'silver': (1146, 35)
        }[name]
        abs_x = cx - 20 + best_loc[0]
        abs_y = cy - 20 + best_loc[1]
        
        print(f"{name}: Best match at Scale {best_scale:.1f} with score = {best_score:.4f} at local {best_loc} (Abs: {abs_x}, {abs_y})")

if __name__ == "__main__":
    compare()
