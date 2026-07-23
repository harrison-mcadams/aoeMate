import cv2
import numpy as np
import os
from PIL import Image

def search_all():
    img_path = 'debug_verification_resources/input_capture.png'
    if not os.path.exists(img_path):
        print("Error: Capture not found.")
        return
        
    img = Image.open(img_path)
    img_gray = np.array(img.convert('L'), dtype=np.float32)
    sh, sw = img_gray.shape
    print(f"Searching entire screenshot {sw}x{sh}...")
    
    resources = ['food', 'wood', 'gold', 'stone']
    scales = [1.0, 1.5, 2.0, 2.5]
    
    for name in resources:
        template_path = f"templates/{name}.png"
        k = Image.open(template_path)
        k_gray = np.array(k.convert('L'), dtype=np.float32)
        
        print(f"\n--- Top matches for {name} ---")
        best_overall = []
        
        for scale in scales:
            k_scaled = cv2.resize(k_gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            kh, kw = k_scaled.shape
            
            res = cv2.matchTemplate(img_gray, k_scaled, cv2.TM_CCOEFF_NORMED)
            
            # Find peaks with score >= 0.65
            locs = np.where(res >= 0.65)
            for pt in zip(*locs[::-1]):
                score = res[pt[1], pt[0]]
                best_overall.append((pt[0], pt[1], score, scale))
                
        # NMS on best_overall
        best_overall.sort(key=lambda x: x[2], reverse=True)
        unique_matches = []
        for m in best_overall:
            # check distance
            if all(np.hypot(m[0] - um[0], m[1] - um[1]) > 30 for um in unique_matches):
                unique_matches.append(m)
                
        for x, y, score, scale in unique_matches[:10]:
            print(f"  x={x}, y={y}, score={score:.4f} at Scale {scale:.1f}")

if __name__ == "__main__":
    search_all()
