import cv2
import numpy as np
import os
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)

import main
import analyze_ss
import get_ss

def dump():
    img_path = 'debug_verification_resources/input_capture.png'
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return
        
    img = Image.open(img_path)
    print(f"Loaded image size: {img.size}")
    
    scale_factor = 1.0
    if img.width > 600:
        scale_factor = 2.0
    print(f"Scale factor: {scale_factor}")
    
    ss_gray = np.array(img.convert('L'), dtype=np.float32)
    
    resources = [
        ('food', 'food.png'),
        ('wood', 'wood.png'),
        ('gold', 'gold.png'),
        ('stone', 'stone.png'),
        ('silver', 'silver.png'),
    ]
    
    # Load and scale kernels
    main._init_kernels_and_executors(resources)
    res_kernels_gray = {}
    for k, v in main._RESOURCE_KERNELS_GRAY.items():
        if v is not None:
            if scale_factor > 1.5:
                res_kernels_gray[k] = cv2.resize(v, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
            else:
                res_kernels_gray[k] = v
                
    print("\n--- Raw Peaks for each template ---")
    for name, fname in resources:
        k_gray = res_kernels_gray.get(fname)
        if k_gray is None:
            print(f"{name}: Template not loaded.")
            continue
            
        res_conv = analyze_ss.match_template_arrays(ss_gray, k_gray)
        found, peaks = analyze_ss.is_target_in_ss(res_conv, None, return_peaks=True, threshold=0.30)
        
        print(f"\n{name} ({fname}):")
        if not found or not peaks:
            print("  No peaks found >= 0.30")
            continue
            
        # Sort by score descending
        peaks.sort(key=lambda p: p[2], reverse=True)
        for x, y, score in peaks[:10]:
            print(f"  x={x}, y={y}, score={score:.4f}")
            
if __name__ == "__main__":
    dump()
