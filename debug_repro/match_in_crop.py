import cv2
import numpy as np
import os
from PIL import Image
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import main

def test_crop_matching():
    img_path = 'debug_verification_resources/input_capture.png'
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return
        
    img = Image.open(img_path)
    img_gray = np.array(img.convert('L'), dtype=np.float32)
    sh, sw = img_gray.shape
    print(f"Full Image Size: {sw}x{sh}")
    
    # Crop to where the icons actually are
    # x: 0 to 80
    # y: 900 to 1200
    crop = img_gray[900:1200, 0:80]
    ch, cw = crop.shape
    print(f"Crop Size: {cw}x{ch}")
    
    resources = [
        ('food', 'food.png'),
        ('wood', 'wood.png'),
        ('gold', 'gold.png'),
        ('stone', 'stone.png'),
        ('silver', 'silver.png'),
    ]
    main._init_kernels_and_executors(resources)
    
    scales = [1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4]
    
    for scale in scales:
        print(f"\n--- Scale {scale} ---")
        for name, fname in resources:
            k = main._RESOURCE_KERNELS_GRAY.get(fname)
            if k is None:
                print(f"  {name}: Kernel not loaded")
                continue
                
            if scale != 1.0:
                k_scaled = cv2.resize(k, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            else:
                k_scaled = k
                
            kh, kw = k_scaled.shape
            if kh > ch or kw > cw:
                continue
                
            res = cv2.matchTemplate(crop, k_scaled, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            abs_x = max_loc[0]
            abs_y = 900 + max_loc[1]
            print(f"  {name:6s} | Score: {max_val:.4f} at x={abs_x}, y={abs_y}")

if __name__ == "__main__":
    test_crop_matching()
