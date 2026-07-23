import cv2
import numpy as np
import os
from PIL import Image
import logging

logging.basicConfig(level=logging.WARNING)

import main
import analyze_ss
import get_ss

def test_scales():
    img_path = 'debug_verification_resources/input_capture.png'
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return
        
    img = Image.open(img_path)
    ss_gray = np.array(img.convert('L'), dtype=np.float32)
    sh, sw = ss_gray.shape
    print(f"Image size: {sw}x{sh}")
    
    resources = [
        ('food', 'food.png'),
        ('wood', 'wood.png'),
        ('gold', 'gold.png'),
        ('stone', 'stone.png'),
        ('silver', 'silver.png'),
    ]
    
    # Pre-load kernels at 1.0
    main._init_kernels_and_executors(resources)
    
    scales = [1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
    
    for scale in scales:
        # Resize kernels
        res_kernels_gray = {}
        for k, v in main._RESOURCE_KERNELS_GRAY.items():
            if v is not None:
                if abs(scale - 1.0) > 0.05:
                    res_kernels_gray[k] = cv2.resize(v, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                else:
                    res_kernels_gray[k] = v
                    
        candidates = []
        for name, fname in resources:
            k_gray = res_kernels_gray.get(fname)
            if k_gray is None:
                continue
            res_conv = analyze_ss.match_template_arrays(ss_gray, k_gray)
            found, peaks = analyze_ss.is_target_in_ss(res_conv, None, return_peaks=True, threshold=0.45)
            if found and peaks:
                for x, y, score in peaks:
                    # Spatial restrictions
                    # We expect resources on the left side of the screen
                    if x < 150 * scale:
                        candidates.append({'name': name, 'score': score, 'x': int(x), 'y': int(y)})
                        
        if not candidates:
            continue
            
        # Group candidates by X coordinate
        x_tolerance = 5 * scale
        columns = {}
        for c in candidates:
            placed = False
            for col_x in list(columns.keys()):
                if abs(c['x'] - col_x) <= x_tolerance:
                    columns[col_x].append(c)
                    placed = True
                    break
            if not placed:
                columns[c['x']] = [c]
                
        # Find best column
        best_col_x = None
        best_col_score = 0
        best_col_cands = []
        
        for col_x, cands in columns.items():
            # Count unique resource names
            names = set(c['name'] for c in cands)
            # Find best candidate for each name in this column
            resolved = {}
            for c in cands:
                name = c['name']
                if name not in resolved or c['score'] > resolved[name]['score']:
                    resolved[name] = c
            
            # Score this column based on number of unique resources and sum of correlation scores
            col_score = len(resolved) * 10.0 + sum(c['score'] for c in resolved.values())
            if col_score > best_col_score:
                best_col_score = col_score
                best_col_x = col_x
                best_col_cands = list(resolved.values())
                
        if best_col_cands:
            best_col_cands.sort(key=lambda c: c['y'])
            avg_x = sum(c['x'] for c in best_col_cands) / len(best_col_cands)
            avg_score = sum(c['score'] for c in best_col_cands) / len(best_col_cands)
            print(f"Scale {scale:.1f}: Best column at avg_x={avg_x:.1f}, count={len(best_col_cands)}, avg_score={avg_score:.4f}")
            for c in best_col_cands:
                print(f"  {c['name']}: x={c['x']}, y={c['y']}, score={c['score']:.4f}")
            # print vertical diffs
            if len(best_col_cands) > 1:
                diffs = [best_col_cands[i]['y'] - best_col_cands[i-1]['y'] for i in range(1, len(best_col_cands))]
                print(f"  Vertical gaps: {diffs}")

if __name__ == "__main__":
    test_scales()
