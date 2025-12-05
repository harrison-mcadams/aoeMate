import cv2
import numpy as np
import os
from PIL import Image
import main
import analyze_ss

def analyze_layout():
    # Load debug screenshot
    img_path = os.path.expanduser("~/Desktop/debug_screenshot_new.png")
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found")
        return

    img = Image.open(img_path)
    ss_gray = np.array(img.convert('L'), dtype=np.float32)
    
    # Initialize kernels
    resources = [
        ('food', 'food_icon.png'),
        ('wood', 'wood_icon.png'),
        ('gold', 'gold_icon.png'),
        ('stone', 'stone_icon.png'),
        ('silver', 'silver_icon_macedonia.png'),
    ]
    main._init_kernels_and_executors(resources)
    
    # Find anchors
    print("Finding anchors...")
    anchors = main._find_anchors(ss_gray, main._RESOURCE_KERNELS_GRAY)
    print(f"Anchors: {anchors}")
    
    if not anchors:
        print("No anchors found!")
        return

    print("\n--- Row Analysis ---")
    for name, fname in resources:
        anchor = anchors.get(name)
        if not anchor:
            continue
            
        ax, ay = anchor
        k_gray = main._RESOURCE_KERNELS_GRAY.get(fname)
        h, w = k_gray.shape
        
        # Define a wide strip to the right of the icon
        strip_left = ax + w
        strip_top = ay
        strip_w = 150 # Look far enough to catch villager count
        strip_h = h
        
        # Crop strip
        strip = ss_gray[strip_top:strip_top+strip_h, strip_left:strip_left+strip_w]
        
        # Threshold to find content
        _, thresh = cv2.threshold(strip, 180, 255, cv2.THRESH_BINARY)
        
        # Horizontal projection (sum columns)
        proj = np.sum(thresh, axis=0)
        
        # Find non-zero segments
        segments = []
        in_segment = False
        start_x = 0
        
        for x, val in enumerate(proj):
            if val > 0:
                if not in_segment:
                    in_segment = True
                    start_x = x
            else:
                if in_segment:
                    in_segment = False
                    segments.append((start_x, x))
                    
        if in_segment:
             segments.append((start_x, len(proj)))
             
        # Filter small noise
        valid_segments = [s for s in segments if (s[1] - s[0]) > 2]
        
        # Group segments that are close together (digits of the same number)
        groups = []
        if valid_segments:
            curr_group_start = valid_segments[0][0]
            curr_group_end = valid_segments[0][1]
            
            for i in range(1, len(valid_segments)):
                s_start, s_end = valid_segments[i]
                if s_start - curr_group_end < 15: # If gap is small, it's the same number
                    curr_group_end = s_end
                else:
                    groups.append((curr_group_start, curr_group_end))
                    curr_group_start = s_start
                    curr_group_end = s_end
            groups.append((curr_group_start, curr_group_end))
            
        print(f"Resource: {name}")
        for i, (start, end) in enumerate(groups):
            print(f"  Group {i}: Start={start} End={end} Width={end-start}")
            
if __name__ == "__main__":
    analyze_layout()
