import cv2
import numpy as np
import os
from PIL import Image

def profile():
    img_path = 'debug_verification_resources/input_capture.png'
    if not os.path.exists(img_path):
        print("Error: Capture not found.")
        return
        
    img = Image.open(img_path)
    ss_gray = np.array(img.convert('L'), dtype=np.float32)
    sh, sw = ss_gray.shape
    
    resources = ['food', 'wood', 'gold', 'stone', 'silver']
    scale = 2.0
    
    print(f"Scanning column x = 20 to 45, y = 300 to 1180 at Scale {scale}:")
    
    for name in resources:
        fname = f"templates/{name}.png"
        k = Image.open(fname)
        k_gray = np.array(k.convert('L'), dtype=np.float32)
        k_scaled = cv2.resize(k_gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        kh, kw = k_scaled.shape
        
        res = cv2.matchTemplate(ss_gray, k_scaled, cv2.TM_CCOEFF_NORMED)
        
        # Find local peaks in the column
        peaks = []
        for y in range(300, min(res.shape[0], 1180)):
            for x in range(20, min(res.shape[1], 45)):
                score = res[y, x]
                if score >= 0.35:
                    peaks.append((x, y, score))
                    
        # NMS for this template in the column
        peaks.sort(key=lambda p: p[2], reverse=True)
        unique_peaks = []
        for p in peaks:
            if all(abs(p[1] - up[1]) > 15 for up in unique_peaks):
                unique_peaks.append(p)
                
        print(f"\n{name} peaks:")
        unique_peaks.sort(key=lambda p: p[1])
        for x, y, score in unique_peaks:
            print(f"  y={y} (x={x}), score={score:.4f}")
            
if __name__ == "__main__":
    profile()
