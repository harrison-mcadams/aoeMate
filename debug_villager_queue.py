import cv2
import numpy as np
from PIL import Image
import analyze_ss
import os

# Load debug screenshot
img_path = "/Users/harrisonmcadams/Desktop/debug_screenshot_new.png"
if not os.path.exists(img_path):
    print(f"Error: {img_path} not found.")
    exit(1)

img = Image.open(img_path)
print(f"Loaded image: {img.size}")

# Load villager kernel
kernel_path = "/Users/harrisonmcadams/Desktop/villager_icon.png"
if not os.path.exists(kernel_path):
    print(f"Error: {kernel_path} not found.")
    exit(1)

kernel = Image.open(kernel_path)
print(f"Loaded kernel: {kernel.size}")

# Define ROI (Search Whole Image)
# roi_box = (0, 0, 60, 60)
# roi = img.crop(roi_box)
roi = img # Use whole image
roi.save("/Users/harrisonmcadams/Desktop/debug_vill_roi.png")
print(f"Saved ROI to debug_vill_roi.png")

# Run template matching
try:
    # Convert to numpy for OpenCV
    roi_np = np.array(roi.convert('RGB'))
    roi_gray = cv2.cvtColor(roi_np, cv2.COLOR_RGB2GRAY)
    
    k_np = np.array(kernel.convert('RGB'))
    k_gray = cv2.cvtColor(k_np, cv2.COLOR_RGB2GRAY)
    
    res = cv2.matchTemplate(roi_gray, k_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    print(f"Max correlation score: {max_val:.4f}")
    print(f"Location: {max_loc}")
    
    # Save result visualization
    res_norm = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite("/Users/harrisonmcadams/Desktop/debug_vill_match.png", res_norm)
    
    threshold = 0.65
    if max_val >= threshold:
        print("PASS: Villager detected.")
    else:
        print(f"FAIL: Score below threshold {threshold}")

except Exception as e:
    print(f"Error running template matching: {e}")
