import cv2
import numpy as np
import os
from PIL import Image

def analyze():
    img_path = 'debug_verification_resources/input_capture.png'
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return
        
    img = cv2.imread(img_path)
    h, w, c = img.shape
    print(f"Captured Image Shape: {h}x{w}x{c}")
    
    # Calculate average brightness in different quadrants/strips
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    print(f"Overall average brightness (0-255): {mean_val:.2f}")
    
    # Let's save a crop of the bottom-left region of the capture.
    # The bottom-left of the screen should be at the bottom of the capture (since it's positioned at the bottom left).
    # Specifically, left = 0 to 400, top = h-400 to h.
    crop_bottom_left = img[max(0, h-400):h, 0:min(w, 400)]
    cv2.imwrite('debug_repro/crop_bottom_left.png', crop_bottom_left)
    print(f"Saved bottom-left crop (400x400) to debug_repro/crop_bottom_left.png")
    
    # Let's search for food template specifically in the bottom-left crop.
    food_template = cv2.imread('templates/food.png', cv2.IMREAD_GRAYSCALE)
    if food_template is not None:
        print(f"Food template size: {food_template.shape}")
        # Let's test template matching on the bottom-left crop at scale 1.0, 1.5, 2.0, 2.5, 3.0
        crop_gray = cv2.cvtColor(crop_bottom_left, cv2.COLOR_BGR2GRAY)
        for scale in [1.0, 1.5, 2.0, 2.5, 3.0]:
            temp_scaled = cv2.resize(food_template, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            if temp_scaled.shape[0] > crop_gray.shape[0] or temp_scaled.shape[1] > crop_gray.shape[1]:
                continue
            res = cv2.matchTemplate(crop_gray.astype(np.float32), temp_scaled.astype(np.float32), cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            print(f"Scale {scale:.1f}: Max Correlation Score = {max_val:.4f} at {max_loc}")
            
if __name__ == "__main__":
    analyze()
