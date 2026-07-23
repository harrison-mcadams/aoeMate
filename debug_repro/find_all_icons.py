import cv2
import numpy as np
import os
from PIL import Image

def find():
    img_path = 'debug_verification_resources/input_capture.png'
    if not os.path.exists(img_path):
        print("Error: Capture not found.")
        return
        
    img = Image.open(img_path)
    ss_gray = np.array(img.convert('L'), dtype=np.float32)
    sh, sw = ss_gray.shape
    
    resources = ['food', 'wood', 'gold', 'stone', 'silver']
    scale = 2.0
    
    print(f"Searching at Scale {scale} in region x < 200, y > 800:")
    
    for name in resources:
        fname = f"templates/{name}.png"
        k = Image.open(fname)
        k_gray = np.array(k.convert('L'), dtype=np.float32)
        # Scale kernel
        k_scaled = cv2.resize(k_gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        kh, kw = k_scaled.shape
        
        # Match template
        res = cv2.matchTemplate(ss_gray, k_scaled, cv2.TM_CCOEFF_NORMED)
        
        # We only want peaks in the bottom-left region: x < 200, y > 800
        # The matchTemplate response size is (sh - kh + 1, sw - kw + 1)
        peaks = []
        for y in range(max(0, 800), min(res.shape[0], sh)):
            for x in range(0, min(res.shape[1], 200)):
                score = res[y, x]
                if score >= 0.35:
                    peaks.append((x, y, score))
                    
        # Apply basic NMS to list of peaks
        peaks.sort(key=lambda p: p[2], reverse=True)
        unique_peaks = []
        for p in peaks:
            # check distance
            if all(np.hypot(p[0] - up[0], p[1] - up[1]) > 10 for up in unique_peaks):
                unique_peaks.append(p)
                
        print(f"\n{name} matches:")
        for x, y, score in unique_peaks[:5]:
            print(f"  x={x}, y={y}, score={score:.4f}")
            
if __name__ == "__main__":
    find()
