
import main
import cv2
import numpy as np
import os
from PIL import Image

def debug_anchors():
    img_path = os.path.expanduser("~/Desktop/debug_screenshot_new.png")
    try:
        img = Image.open(img_path)
        print(f"Loaded image: {img.size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Convert to grayscale numpy array
    img_gray = np.array(img.convert('L'), dtype=np.float32)
    
    # Load resource kernels manually
    resource_kernels = {}
    # List from main.py
    resources = [
        ('food', 'food_icon.png'),
        ('wood', 'wood_icon.png'),
        ('gold', 'gold_icon.png'),
        ('stone', 'stone_icon.png'),
        ('silver', 'silver_icon_macedonia.png'),
        ('food', 'food_icon_macedonia.png'),
        ('wood', 'wood_icon_macedonia.png'),
        ('stone', 'stone_icon_macedonia.png'),
    ]
    
    kernel_path = os.environ.get('AOE_KERNEL_PATH', '/Users/harrisonmcadams/Desktop/')
    
    print(f"Loading kernels from {kernel_path}...")
    for _, fname in resources:
        try:
            k_path = os.path.join(kernel_path, fname)
            if os.path.exists(k_path):
                k = Image.open(k_path)
                resource_kernels[fname] = np.array(k.convert('L'), dtype=np.float32)
            else:
                print(f"Kernel not found: {k_path}")
        except Exception as e:
            print(f"Error loading {fname}: {e}")

    print("Finding anchors...")
    # main._find_anchors returns a dict of name -> (x, y)
    # It doesn't return scores directly.
    # I need to modify main.py to return scores or copy the logic.
    # Actually, I can just copy the logic here to see the scores.
    
    import analyze_ss
    import math
    
    candidates = []
    img_gray_arr = img_gray
    
    for name, k_gray in resource_kernels.items():
        res_conv = analyze_ss.match_template_arrays(img_gray_arr, k_gray)
        found, peaks = analyze_ss.is_target_in_ss(res_conv, None, return_peaks=True, threshold=0.6)
        if found and peaks:
            for x, y, score in peaks:
                candidates.append({'name': name, 'score': score, 'x': int(x), 'y': int(y)})
                
    candidates.sort(key=lambda c: c['score'], reverse=True)
    
    anchors = {}
    occupied = []
    min_dist = 10
    
    print("Top 20 Candidates:")
    for c in candidates[:20]:
        print(f"{c['name']}: ({c['x']}, {c['y']}) score={c['score']:.4f}")
        
    # Replicate main.py logic
    for c in candidates:
        is_occ = False
        for ox, oy in occupied:
            if math.hypot(c['x']-ox, c['y']-oy) < min_dist:
                is_occ = True
                break
        if not is_occ:
            if c['name'] not in anchors:
                anchors[c['name']] = (c['x'], c['y'])
                occupied.append((c['x'], c['y']))
                
    print(f"Anchors found: {anchors}")
    
    # Draw anchors on image
    debug_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    for name, (x, y) in anchors.items():
        print(f"Drawing {name} at ({x}, {y})")
        cv2.rectangle(debug_img, (x, y), (x + 30, y + 30), (0, 0, 255), 2)
        cv2.putText(debug_img, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
    out_path = os.path.expanduser("~/Desktop/debug_anchors.png")
    cv2.imwrite(out_path, debug_img)
    print(f"Saved debug image to {out_path}")

if __name__ == "__main__":
    debug_anchors()
