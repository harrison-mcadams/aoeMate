import main
import cv2
import os
from PIL import Image
import logging
import numpy as np

# Setup logging to console
logging.basicConfig(level=logging.INFO)

def run_reproduction():
    img_path = 'debug_bbox_check.png'
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return

    print(f"Loading {img_path}...")
    # Load as CV2 BGR for consistency with main loop capture
    img_cv = cv2.imread(img_path)
    print(f"Image shape: {img_cv.shape}")

    # Load templates
    print("Loading villager kernels...")
    vill_kernels = main._load_villager_kernels()
    
    print("\nRunning check_villager_production...")
    is_producing, score = main.check_villager_production(img_cv, vill_kernels)
    
    print(f"\nResult: Producing={is_producing}, Score={score}")
    
    # Also run summarize_eco just to be sure we didn't break it
    # summarize_eco expects PIL or it converts internally. Let's pass the PIL image
    # img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    # results = main.summarize_eco(screenshot=img_pil)
    # print("\nEco Results:", results)

if __name__ == "__main__":
    run_reproduction()
