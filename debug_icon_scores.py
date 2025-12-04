
import cv2
import numpy as np
from PIL import Image
import main
import analyze_ss
import os

img_path = os.path.expanduser("~/Desktop/debug_panel.png")
if not os.path.exists(img_path):
    print("debug_panel.png not found")
    exit()

img = Image.open(img_path)
ss_gray = np.array(img.convert('L'), dtype=np.float32)

resources = [
    ('food', 'food_icon.png'),
    ('wood', 'wood_icon.png'),
    ('gold', 'gold_icon.png'),
    ('stone', 'stone_icon.png'),
    ('silver', 'silver_icon.png'),
]
main._init_kernels_and_executors(resources)

print(f"Image size: {img.size}")

for name, fname in resources:
    k_gray = main._RESOURCE_KERNELS_GRAY.get(fname)
    if k_gray is None:
        print(f"Kernel missing: {fname}")
        continue
        
    res = cv2.matchTemplate(ss_gray, k_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    print(f"{name}: Max Score = {max_val:.4f} at {max_loc}")
