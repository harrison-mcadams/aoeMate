import cv2
import numpy as np
import os
from PIL import Image
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import main
import analyze_ss

def test_restricted():
    img_path = 'debug_verification_resources/input_capture.png'
    if not os.path.exists(img_path):
        print("Error: Capture not found.")
        return
        
    img = Image.open(img_path)
    ss_gray = np.array(img.convert('L'), dtype=np.float32)
    sh, sw = ss_gray.shape
    
    resources = [
        ('food', 'food.png'),
        ('wood', 'wood.png'),
        ('gold', 'gold.png'),
        ('stone', 'stone.png'),
        ('silver', 'silver.png'),
    ]
    main._init_kernels_and_executors(resources)
    
    scales = [1.2, 1.5]
    
    for scale in scales:
        print(f"\n================ Scale {scale:.2f} ================")
        
        # Scale resource templates
        res_kernels_gray = {}
        for k, v in main._RESOURCE_KERNELS_GRAY.items():
            if v is not None:
                if scale != 1.0:
                    res_kernels_gray[k] = cv2.resize(v, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                else:
                    res_kernels_gray[k] = v
                    
        # Find matches ONLY in x <= 80, y >= 900
        candidates = []
        for name, fname in resources:
            k_gray = res_kernels_gray.get(fname)
            if k_gray is None:
                continue
            res_conv = analyze_ss.match_template_arrays(ss_gray, k_gray)
            found, peaks = analyze_ss.is_target_in_ss(res_conv, None, return_peaks=True, threshold=0.30)
            
            if found and peaks:
                for x, y, score in peaks:
                    if x <= 80 and y >= 900:
                        candidates.append({'name': name, 'score': score, 'x': int(x), 'y': int(y)})
                        
        # Sort candidates by Y coordinate
        candidates.sort(key=lambda c: c['y'])
        print(f"Candidates found: {len(candidates)}")
        for c in candidates:
            print(f"  {c['name']:6s}: x={c['x']}, y={c['y']}, score={c['score']:.4f}")

if __name__ == "__main__":
    test_restricted()
