
import main
import analyze_ss
import get_ss
from PIL import Image
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

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

print("Testing generalized villager detection...")

# Replicate the logic in main.py
roi = img.crop((0, 0, 60, 60))
conv = analyze_ss.convolve_ssXkernel(roi, vill_kernel)
binary = analyze_ss.is_target_in_ss(conv, vill_kernel, threshold=0.65)

if binary:
    print("[PASS] Villagers detected (Producing)")
else:
    print("[FAIL] Villagers NOT detected (Not Producing)")
