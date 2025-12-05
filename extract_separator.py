import cv2
import numpy as np
import os
from PIL import Image
import main

def extract_separator():
    # Load debug screenshot
    img_path = os.path.expanduser("~/Desktop/debug_screenshot_new.png")
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found")
        return

    img = Image.open(img_path)
    ss_gray = np.array(img.convert('L'), dtype=np.float32)
    
    # Initialize kernels to find gold anchor
    resources = [('gold', 'gold_icon.png')]
    main._init_kernels_and_executors(resources)
    
    # Find anchors
    print("Finding anchors...")
    anchors = main._find_anchors(ss_gray, main._RESOURCE_KERNELS_GRAY)
    print(f"Anchors: {anchors}")
    
    anchor = anchors.get('gold')
    if not anchor:
        print("Gold anchor not found!")
        return
        
    ax, ay = anchor
    k_gray = main._RESOURCE_KERNELS_GRAY.get('gold_icon.png')
    h, w = k_gray.shape
    
    # Define region where separator is expected
    # Based on previous analysis, resource count is ~37px wide for gold (1460)
    # Villager count starts around 90px?
    # Let's look in the gap.
    
    # Crop a strip to visualize and select
    strip_left = ax + w + 40 # Skip most of the number
    strip_top = ay
    strip_w = 60 
    strip_h = h
    
    strip = ss_gray[strip_top:strip_top+strip_h, strip_left:strip_left+strip_w]
    
    # Save strip for inspection (optional, but we want to automate extraction)
    # Let's assume the separator is the first significant blob in this strip.
    
    _, thresh = cv2.threshold(strip, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour in this strip (should be the person icon)
    best_cnt = None
    best_area = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > best_area:
            best_area = area
            best_cnt = cnt
            
    if best_cnt is not None:
        x, y, cw, ch = cv2.boundingRect(best_cnt)
        print(f"Found separator candidate at relative (x={x}, y={y}, w={cw}, h={ch})")
        
        # Crop the icon from the original gray image
        # Add a small padding
        pad = 1
        sep_x = strip_left + x - pad
        sep_y = strip_top + y - pad
        sep_w = cw + 2*pad
        sep_h = ch + 2*pad
        
        separator_icon = ss_gray[sep_y:sep_y+sep_h, sep_x:sep_x+sep_w]
        
        out_path = os.path.join(os.path.dirname(main._KERNEL_PATH), 'villager_separator.png')
        # Ensure it's saved as uint8
        cv2.imwrite(out_path, separator_icon.astype(np.uint8))
        print(f"Saved separator icon to {out_path}")
        
        # Also print the offset from the resource icon
        offset_x = sep_x - (ax + w)
        print(f"Offset from resource icon right edge: {offset_x} pixels")
        
    else:
        print("No separator found in strip")

if __name__ == "__main__":
    extract_separator()
