
import main
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

# Create debug output directory
debug_dir = os.path.expanduser("~/Desktop/debug_silver_extraction")
os.makedirs(debug_dir, exist_ok=True)

# Run summarize_eco with out_path to trigger debug image saving
print(f"Running extraction and saving debug images to {debug_dir}...")
results = main.summarize_eco(screenshot=img, out_path=debug_dir)

print("\n--- Extraction Results ---")
print(f"Silver: {results.get('silver')}")
