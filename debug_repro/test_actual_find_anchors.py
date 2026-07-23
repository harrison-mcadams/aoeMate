import cv2
import numpy as np
import os
from PIL import Image
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import main

def test():
    img_path = 'debug_verification_resources/input_capture.png'
    if not os.path.exists(img_path):
        print("Error: Capture not found.")
        return
        
    img = Image.open(img_path)
    ss_gray = np.array(img.convert('L'), dtype=np.float32)
    
    resources = [
        ('food', 'food.png'),
        ('wood', 'wood.png'),
        ('gold', 'gold.png'),
        ('stone', 'stone.png'),
        ('silver', 'silver.png'),
    ]
    main._init_kernels_and_executors(resources)
    
    for scale in [1.5, 1.8, 2.0, 2.25, 2.5]:
        print(f"\n--- Testing scale_factor = {scale} ---")
        res_kernels_gray = {}
        for k, v in main._RESOURCE_KERNELS_GRAY.items():
            if v is not None:
                res_kernels_gray[k] = cv2.resize(v, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                
        anchors = main._find_anchors(ss_gray, res_kernels_gray, scale_factor=scale)
        print("Resolved anchors:")
        for name, pos in anchors.items():
            print(f"  {name}: {pos}")

if __name__ == "__main__":
    test()
