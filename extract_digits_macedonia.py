
import cv2
import numpy as np
from PIL import Image
import main
import analyze_ss
import os

img_path = os.path.expanduser("~/Desktop/debug_panel.png")
img = Image.open(img_path)
ss_gray = np.array(img.convert('L'), dtype=np.float32)

# Known anchors (from previous step)
anchors = {
    'food': (5, 161),
    'wood': (5, 191),
    'silver': (5, 221),
    'stone': (5, 251)
}

# Ground truth
truth = {
    'food': '605',
    'wood': '611',
    'silver': '223',
    'stone': '28'
}

# Icon width approx 30
icon_w = 30
fudge = 10

for name, (ax, ay) in anchors.items():
    # Define count region
    count_left = ax + icon_w + 2
    count_top = ay - fudge
    count_w = 100
    count_h = 30 + fudge * 2
    
    expected_digits = truth[name]
    
    crop = img.crop((count_left, count_top, count_left + count_w, count_top + count_h))
    
    best_thresh = 0
    best_bboxes = []
    
    # Try multiple thresholds
    for thresh_val in [80, 100, 120, 140, 160]:
        thresh = analyze_ss.threshold_image(crop, threshold=thresh_val)
        
        # Find contours using helper
        # Relax constraints slightly for low quality
        bboxes = analyze_ss.find_digit_contours(thresh, min_h=6, max_h=30)
        
        # Sort by x
        bboxes.sort(key=lambda x: x[0])
        
        # Filter
        filtered = []
        if bboxes:
            last_x = bboxes[0][0] + bboxes[0][2]
            filtered.append(bboxes[0])
            for i in range(1, len(bboxes)):
                x, y, w, h = bboxes[i]
                if x - last_x > 12: 
                    break
                filtered.append(bboxes[i])
                last_x = x + w
        
        print(f"{name} (thresh={thresh_val}): Found {len(filtered)} digits")
        
        if len(filtered) == len(expected_digits):
            best_thresh = thresh_val
            best_bboxes = filtered
            break
            
    if best_bboxes:
        print(f"{name}: Success with threshold {best_thresh}")
        for i, digit_char in enumerate(expected_digits):
            x, y, w, h = best_bboxes[i]
            # Re-threshold with best value
            thresh = analyze_ss.threshold_image(crop, threshold=best_thresh)
            digit_img = thresh[y:y+h, x:x+w]
            
            # Save
            fname = f"digit_{digit_char}_macedonia_{name}_{i}.png"
            Image.fromarray(digit_img).save(os.path.expanduser(f"~/Desktop/{fname}"))
            print(f"Saved {fname}")
    else:
        print(f"{name}: Failed to find expected digits")
