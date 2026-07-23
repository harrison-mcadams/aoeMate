import cv2
import numpy as np
import os

def inspect():
    img_path = 'debug_verification_resources/input_capture.png'
    if not os.path.exists(img_path):
        print("Error: Capture not found.")
        return
        
    img = cv2.imread(img_path)
    h, w, c = img.shape
    print(f"Image shape: {h}x{w}x{c}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate average intensity for each row (1200 rows)
    row_means = np.mean(gray, axis=1)
    
    # Group rows into bins of 20 pixels to print a compact summary
    bin_size = 30
    print("\n--- Vertical Brightness Profile (30px bins) ---")
    for b in range(h // bin_size):
        start_y = b * bin_size
        end_y = (b + 1) * bin_size
        val = np.mean(row_means[start_y:end_y])
        # Print a small bar graph
        bar = "#" * int(val / 255.0 * 60)
        print(f"y={start_y:4d}..{end_y:4d} | avg={val:5.1f} | {bar}")

if __name__ == "__main__":
    inspect()
