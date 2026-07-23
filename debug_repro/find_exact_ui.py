import cv2
import numpy as np
import os
from PIL import Image
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import analyze_ss

def find_ui():
    img_path = 'debug_verification_resources/input_capture.png'
    if not os.path.exists(img_path):
        print("Error: Capture not found.")
        return
        
    img = Image.open(img_path)
    ss_gray = np.array(img.convert('L'), dtype=np.float32)
    sh, sw = ss_gray.shape
    
    resources = ['food', 'wood', 'gold', 'stone', 'silver']
    scale = 1.5
    
    print(f"Searching at Scale {scale} in region x < 150, y > 800:")
    
    for name in resources:
        fname = f"templates/{name}.png"
        k = Image.open(fname)
        k_gray = np.array(k.convert('L'), dtype=np.float32)
        k_scaled = cv2.resize(k_gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        
        res = cv2.matchTemplate(ss_gray, k_scaled, cv2.TM_CCOEFF_NORMED)
        
        # Find all peaks with score >= 0.30 in x < 150, y > 800
        peaks = []
        for y in range(800, res.shape[0]):
            for x in range(0, min(res.shape[1], 150)):
                score = res[y, x]
                if score >= 0.30:
                    peaks.append((x, y, score))
                    
        # NMS
        peaks.sort(key=lambda p: p[2], reverse=True)
        unique_peaks = []
        for p in peaks:
            if all(np.hypot(p[0] - up[0], p[1] - up[1]) > 10 for up in unique_peaks):
                unique_peaks.append(p)
                
        print(f"\n{name} peaks:")
        unique_peaks.sort(key=lambda p: p[1])
        for x, y, score in unique_peaks:
            print(f"  x={x}, y={y}, score={score:.4f}")
            
if __name__ == "__main__":
    find_ui()
