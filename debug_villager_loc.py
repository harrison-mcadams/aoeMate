
import main
import analyze_ss
import get_ss
from PIL import Image
import os
import cv2
import numpy as np

# Load debug screenshot
img_path = os.path.expanduser("~/Desktop/debug_screenshot_new.png")
try:
    img = Image.open(img_path)
except Exception as e:
    print(f"Error loading image: {e}")
    exit(1)

# Load villager kernel
try:
    vill_kernel = Image.open("/Users/harrisonmcadams/Desktop/villager_icon.png")
except Exception:
    print("Could not load villager kernel")
    exit(1)

# Convolve
print("Convolving...")
conv = analyze_ss.convolve_ssXkernel(img, vill_kernel)

# Find peaks with low threshold
threshold = 0.4 # Lower threshold to find it
print(f"Finding peaks with threshold {threshold}...")
found, peaks = analyze_ss.is_target_in_ss(conv, vill_kernel, return_peaks=True, threshold=threshold)

print(f"Found: {found}")
print(f"Peaks: {peaks}")

# Draw results
debug_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
if peaks:
    h, w = np.array(vill_kernel.convert('L')).shape
    for x, y, score in peaks:
        print(f"Match at ({x}, {y}) with score {score}")
        cv2.rectangle(debug_img, (int(x), int(y)), (int(x)+w, int(y)+h), (0, 255, 0), 2)
        cv2.putText(debug_img, f"{score:.2f}", (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

out_path = os.path.expanduser("~/Desktop/debug_villager_loc.png")
cv2.imwrite(out_path, debug_img)
print(f"Saved debug image to {out_path}")
