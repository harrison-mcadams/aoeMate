
import main
import get_ss
import cv2
import numpy as np
import os
from PIL import Image
import analyze_ss
import logging

# Configure logging to console
logging.basicConfig(level=logging.INFO, format='%(message)s')

def capture_and_debug():
    print("Capturing screenshot...")
    eco_summary = get_ss.get_bbox('eco_summary')
    screenshot = get_ss.capture_gfn_screen_region(eco_summary)
    
    # Save for reference
    debug_ss_path = os.path.expanduser("~/Desktop/debug_capture.png")
    screenshot.save(debug_ss_path)
    print(f"Saved capture to {debug_ss_path}")

    # Convert to grayscale
    ss_gray = np.array(screenshot.convert('L'), dtype=np.float32)
    
    # Load kernels
    resource_kernels = {}
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
    
    for name, fname in resources:
        try:
            k_path = os.path.join(kernel_path, fname)
            if os.path.exists(k_path):
                k = Image.open(k_path)
                resource_kernels[fname] = np.array(k.convert('L'), dtype=np.float32)
            else:
                print(f"Kernel not found: {k_path}")
        except Exception as e:
            print(f"Error loading {fname}: {e}")

    print("\n--- Detailed Anchor Search ---")
    candidates = []
    
    for name, fname in resources:
        k_gray = resource_kernels.get(fname)
        if k_gray is None:
            continue
            
        res_conv = analyze_ss.match_template_arrays(ss_gray, k_gray)
        found, peaks = analyze_ss.is_target_in_ss(res_conv, None, return_peaks=True, threshold=0.4) # Lower threshold to see weak matches
        
        if found and peaks:
            for x, y, score in peaks:
                print(f"Match: {name} ({fname}) at ({x}, {y}) score={score:.4f}")
                candidates.append({'name': name, 'fname': fname, 'score': score, 'x': int(x), 'y': int(y)})
        else:
            print(f"No match for {name} ({fname}) (threshold=0.4)")

    # Run the actual anchor finding logic from main (simplified)
    print("\n--- Running main._find_anchors logic ---")
    anchors = main._find_anchors(ss_gray, resource_kernels)
    print(f"Final Anchors: {anchors}")
    
    # Visualize
    debug_img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    for name, (ax, ay) in anchors.items():
        cv2.rectangle(debug_img, (ax, ay), (ax + 30, ay + 30), (0, 255, 0), 2)
        cv2.putText(debug_img, name, (ax, ay - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
    out_path = os.path.expanduser("~/Desktop/debug_anchors_visual.png")
    cv2.imwrite(out_path, debug_img)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    capture_and_debug()
