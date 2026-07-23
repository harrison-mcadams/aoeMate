import cv2
import numpy as np
import os
from PIL import Image
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import main

def test_final():
    img_path = 'debug_verification_resources/input_capture.png'
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return
        
    img = Image.open(img_path)
    print("Testing updated summarize_eco on desktop capture...")
    
    # Run summarize_eco
    results = main.summarize_eco(screenshot=img)
    
    print("\nDetection Results:")
    import json
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    test_final()
