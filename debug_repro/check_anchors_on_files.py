import cv2
import numpy as np
import os
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)

import main
import get_ss

def test_file(path, scale_factor):
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return
        
    print(f"\n--- Testing {path} with Scale Factor {scale_factor} ---")
    img = Image.open(path)
    ss_gray = np.array(img.convert('L'), dtype=np.float32)
    
    resources = [
        ('food', 'food.png'),
        ('wood', 'wood.png'),
        ('gold', 'gold.png'),
        ('stone', 'stone.png'),
        ('silver', 'silver.png'),
    ]
    
    # Load resource templates
    main._init_kernels_and_executors(resources)
    
    # Scale resource templates
    res_kernels_gray = {}
    for k, v in main._RESOURCE_KERNELS_GRAY.items():
        if v is not None:
            if scale_factor > 1.05:
                res_kernels_gray[k] = cv2.resize(v, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
            else:
                res_kernels_gray[k] = v
                
    # Find anchors
    anchors = main._find_anchors(ss_gray, res_kernels_gray, scale_factor=scale_factor)
    print("Found anchors:")
    for name, pos in anchors.items():
        print(f"  {name}: {pos}")

def run_tests():
    test_file('debug_bbox_check.png', scale_factor=1.5)
    test_file('debug_bbox_huge.png', scale_factor=2.0)
    test_file('debug_verification_resources/input_capture.png', scale_factor=1.5)

if __name__ == "__main__":
    run_tests()
