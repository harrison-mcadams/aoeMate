
import main
import get_ss
from PIL import Image
import os
import logging

# Configure logging to see output
logging.basicConfig(level=logging.INFO)

# Load debug screenshot
img_path = os.path.expanduser("~/Desktop/debug_screenshot_new.png")
try:
    img = Image.open(img_path)
    print(f"Loaded {img_path}")
except Exception as e:
    print(f"Error loading image: {e}")
    exit(1)

# Run summarize_eco on the loaded image
# Note: summarize_eco expects a PIL image
results = main.summarize_eco(screenshot=img)

print("\n--- Extraction Results ---")
for key, value in results.items():
    print(f"{key}: {value}")

# Verification logic
if 'silver' in results and results['silver'] is not None:
    print("\n[PASS] Silver resource detected.")
else:
    print("\n[FAIL] Silver resource NOT detected.")

if 'silver_vills' in results and results['silver_vills'] is None:
    print("[PASS] Silver villagers correctly reported as None.")
else:
    print(f"[FAIL] Silver villagers reported as {results.get('silver_vills')} (Expected None).")
