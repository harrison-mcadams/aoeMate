import cv2
import numpy as np
import os
from PIL import Image
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import analyze_ss
import main

def test_restricted(scale):
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
    
    res_kernels_gray = {}
    for k, v in main._RESOURCE_KERNELS_GRAY.items():
        if v is not None:
            res_kernels_gray[k] = cv2.resize(v, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            
    # Copy main._find_anchors logic but with restricted bounds
    candidates = []
    for name, fname in resources:
        k_gray = res_kernels_gray.get(fname)
        if k_gray is None:
            continue
        res_conv = analyze_ss.match_template_arrays(ss_gray, k_gray)
        found, peaks = analyze_ss.is_target_in_ss(res_conv, None, return_peaks=True, threshold=0.35, min_distance=15)
        
        if found and peaks:
            for x, y, score in peaks:
                # Spatial Restriction
                max_x = int(80 * scale)
                min_y = sh - int(250 * scale)
                if x > max_x or y < min_y:
                    continue
                    
                # Spatial Prior Boost
                expected_x = int(14 * scale)
                dist_from_expected = abs(x - expected_x)
                if dist_from_expected < int(10 * scale):
                    score += 0.2
                elif dist_from_expected < int(20 * scale):
                    score += 0.1
                    
                candidates.append({'name': name, 'score': score, 'x': int(x), 'y': int(y)})
                
    # Sort candidates
    candidates.sort(key=lambda c: c['score'], reverse=True)
    
    # Spatial NMS
    unique_candidates = []
    min_dist = int(10 * scale)
    for c in candidates:
        is_occupied = False
        for oc in unique_candidates:
            dist = np.hypot(c['x'] - oc['x'], c['y'] - oc['y'])
            if dist < min_dist:
                is_occupied = True
                break
        if not is_occupied:
            unique_candidates.append(c)
            
    # Group into columns
    x_tolerance = int(5 * scale)
    columns = []
    for c in unique_candidates:
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
    for col in columns:
        col['total_score'] = sum(cand['score'] for cand in col['candidates'])
        col['has_food'] = any(cand['name'] == 'food' for cand in col['candidates'])
        avg_x = col['x_sum'] / col['count']
        expected_x = 14 * scale
        col['is_aligned'] = abs(avg_x - expected_x) < (30 * scale)
        
    columns = [col for col in columns if col['is_aligned']]
    columns.sort(key=lambda col: (col['has_food'], col['count'], col['total_score']), reverse=True)
    
    if not columns:
        print(f"Scale {scale}: No aligned columns found.")
        return
        
    best_col = columns[0]
    final_candidates = best_col['candidates']
    final_candidates.sort(key=lambda c: c['y'])
    
    # Spacing filtering
    if len(final_candidates) > 1:
        groups = []
        current_group = [final_candidates[0]]
        for i in range(1, len(final_candidates)):
            prev = final_candidates[i-1]
            curr = final_candidates[i]
            dist = curr['y'] - prev['y']
            max_gap = 65.0 * scale
            if dist < max_gap:
                current_group.append(curr)
            else:
                groups.append(current_group)
                current_group = [curr]
        groups.append(current_group)
        
        groups.sort(key=lambda g: (any(c['name'] == 'food' for c in g), len(g), sum(c['score'] for c in g)), reverse=True)
        final_candidates = groups[0]
        
    final_candidates.sort(key=lambda c: c['y'])
    
    # Determine Pop icon
    effective_start = 0
    if final_candidates:
        dist_from_bottom = sh - final_candidates[0]['y']
        if dist_from_bottom > int(125 * scale):
            effective_start = 1
            
    anchors = {}
    resource_order = ['food', 'wood', 'gold', 'stone', 'silver']
    for i in range(effective_start, len(final_candidates)):
        idx = i - effective_start
        if idx < len(resource_order):
            c = final_candidates[i]
            anchors[resource_order[idx]] = (c['x'], c['y'])
            
    print(f"Scale {scale} Resolved Anchors:")
    for name, pos in anchors.items():
        print(f"  {name}: {pos}")

def main_run():
    for scale in [1.5, 1.8, 2.0, 2.25]:
        print(f"\n--- Evaluating Scale {scale} ---")
        test_restricted(scale)

if __name__ == "__main__":
    main_run()
