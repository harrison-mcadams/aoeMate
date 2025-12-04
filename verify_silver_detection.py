
import cv2
import numpy as np
from PIL import Image
import main
import analyze_ss
import os

# Load debug screenshot
img_path = os.path.expanduser("~/Desktop/debug_screenshot_new.png")
try:
    img = Image.open(img_path)
except Exception as e:
    print(f"Error loading image: {e}")
    exit(1)

ss_gray = np.array(img.convert('L'), dtype=np.float32)

# Load kernels including silver
resources = [
    ('food', 'food_icon.png'),
    ('wood', 'wood_icon.png'),
    ('gold', 'gold_icon.png'),
    ('stone', 'stone_icon.png'),
    ('silver', 'silver_icon_macedonia.png'),
]
main._init_kernels_and_executors(resources)

# Find anchors
anchors = main._find_anchors(ss_gray, main._RESOURCE_KERNELS_GRAY)
print("Anchors found:", anchors)

# Draw anchors
debug_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
for name, (x, y) in anchors.items():
    # Find which file corresponds to this name for size
    # This is a bit hacky as multiple files map to same name in main.py, 
    # but here we just want to draw a box.
    fname = None
    for n, f in resources:
        if n == name:
            fname = f
            break
    
    if not fname or fname not in main._RESOURCE_KERNELS_GRAY: 
        continue
        
    k = main._RESOURCE_KERNELS_GRAY[fname]
    if k is None: continue
    
    h, w = k.shape
    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(debug_img, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

out_path = os.path.expanduser("~/Desktop/debug_silver_verification.png")
cv2.imwrite(out_path, debug_img)
print(f"Saved debug image to {out_path}")
