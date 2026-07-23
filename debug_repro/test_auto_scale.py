import cv2
import numpy as np
import os
from PIL import Image
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import analyze_ss
import main

def test_auto_scale():
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
    
    # We test multiple scales: 1.5 and 2.0 (and 1.0 just in case)
    scales = [1.0, 1.5, 2.0]
    best_scale = None
    best_anchors = {}
    best_score = -1.0
    
    for scale in scales:
        print(f"\nEvaluating Scale {scale}:")
        
        # Scale resource templates
        res_kernels_gray = {}
        for k, v in main._RESOURCE_KERNELS_GRAY.items():
            if v is not None:
                if scale != 1.0:
                    res_kernels_gray[k] = cv2.resize(v, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                else:
                    res_kernels_gray[k] = v
                    
        # Find candidates with restricted bounds
        candidates = []
        for name, fname in resources:
            k_gray = res_kernels_gray.get(fname)
            if k_gray is None:
                continue
            res_conv = analyze_ss.match_template_arrays(ss_gray, k_gray)
            found, peaks = analyze_ss.is_target_in_ss(res_conv, None, return_peaks=True, threshold=0.35, min_distance=15)
            
            if found and peaks:
                for x, y, score in peaks:
                    # Enforce strict spatial bounds for the resource panel
                    max_x = int(100 * scale)
                    min_y = sh - int(350 * scale)
                    
                    if x <= max_x and y >= min_y:
                        # Add a small boost for x close to expected 14 * scale
                        expected_x = int(14 * scale)
                        if abs(x - expected_x) < int(15 * scale):
                            score += 0.2
                        candidates.append({'name': name, 'score': score, 'x': int(x), 'y': int(y)})
                        
        if not candidates:
            print("  No candidates found in restricted region.")
            continue
            
        # Group into columns
        x_tolerance = int(5 * scale)
        columns = []
        for c in candidates:
            added = False
            for col in columns:
                avg_x = col['x_sum'] / col['count']
                if abs(c['x'] - avg_x) <= x_tolerance:
                    col['x_sum'] += c['x']
                    col['count'] += 1
                    col['candidates'].append(c)
                    added = True
                    break
            if not added:
                columns.append({'x_sum': c['x'], 'count': 1, 'candidates': [c]})
                
        # Evaluate columns
        valid_columns = []
        for col in columns:
            col['total_score'] = sum(cand['score'] for cand in col['candidates'])
            col['has_food'] = any(cand['name'] == 'food' for cand in col['candidates'])
            avg_x = col['x_sum'] / col['count']
            expected_x = 14 * scale
            col['is_aligned'] = abs(avg_x - expected_x) < (30 * scale)
            if col['is_aligned']:
                valid_columns.append(col)
                
        if not valid_columns:
            print("  No aligned columns found.")
            continue
            
        # Sort valid columns
        valid_columns.sort(key=lambda col: (col['has_food'], col['count'], col['total_score']), reverse=True)
        best_col = valid_columns[0]
        final_cands = best_col['candidates']
        
        # Sort by Y
        final_cands.sort(key=lambda c: c['y'])
        
        # Enforce spacing (equidistant ~ 26 * scale to 55 * scale)
        # Filter outliers
        if len(final_cands) > 1:
            groups = []
            current_group = [final_cands[0]]
            for i in range(1, len(final_cands)):
                prev = final_cands[i-1]
                curr = final_cands[i]
                dist = curr['y'] - prev['y']
                max_gap = 65.0 * scale
                if dist < max_gap:
                    current_group.append(curr)
                else:
                    groups.append(current_group)
                    current_group = [curr]
            groups.append(current_group)
            
            groups.sort(key=lambda g: (any(c['name'] == 'food' for c in g), len(g), sum(c['score'] for c in g)), reverse=True)
            final_cands = groups[0]
            
        final_cands.sort(key=lambda c: c['y'])
        
        # Determine Pop icon (effective_start)
        # Expected Food Y from bottom is ~ 248 pixels at Scale 2.25, i.e. ~ 110 * scale
        # Expected Pop Y from bottom is ~ 135 * scale
        # If candidate 0 Y is higher up (distance from bottom > 125 * scale), it's Pop
        effective_start = 0
        if final_cands:
            dist_from_bottom = sh - final_cands[0]['y']
            if dist_from_bottom > int(125 * scale):
                effective_start = 1
                
        anchors = {}
        resource_order = ['food', 'wood', 'gold', 'stone', 'silver']
        for i in range(effective_start, len(final_cands)):
            idx = i - effective_start
            if idx < len(resource_order):
                c = final_cands[i]
                anchors[resource_order[idx]] = (c['x'], c['y'])
                
        print(f"  Resolved Anchors: {anchors}")
        
        # Calculate metric: count of matched resources + average correlation score
        if len(anchors) >= 2:
            avg_corr = sum(c['score'] for c in final_cands) / len(final_cands)
            metric = len(anchors) * 10.0 + avg_corr
            print(f"  Scale {scale} metric: {metric:.4f} (count={len(anchors)}, avg_score={avg_corr:.4f})")
            
            if metric > best_score:
                best_score = metric
                best_scale = scale
                best_anchors = anchors
                
    print(f"\n=== Auto-detected Scale Factor: {best_scale} ===")
    print(f"Anchors: {best_anchors}")

if __name__ == "__main__":
    test_auto_scale()
