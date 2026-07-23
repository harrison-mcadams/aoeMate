import cv2
import numpy as np
import os
from PIL import Image
import sys

# Add root directory to path to import local modules
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import analyze_ss

def visualize():
    img_path = 'debug_verification_resources/input_capture.png'
    if not os.path.exists(img_path):
        print("Error: Capture not found.")
        return
        
    img_cv = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    resources = {
        'food': (0, 255, 0),      # Green
        'wood': (255, 255, 0),    # Cyan/Yellow
        'gold': (0, 255, 255),    # Yellow
        'stone': (255, 0, 255),   # Magenta
        'silver': (255, 255, 255) # White
    }
    
    scales = [1.0, 1.5, 2.0, 2.5]
    
    report_lines = []
    
    for scale in scales:
        viz_img = img_cv.copy()
        report_lines.append(f"\nScale {scale:.1f}:")
        
        for name, color in resources.items():
            fname = f"templates/{name}.png"
            k = Image.open(fname)
            k_gray = np.array(k.convert('L'), dtype=np.float32)
            
            if scale != 1.0:
                k_scaled = cv2.resize(k_gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            else:
                k_scaled = k_gray
                
            kh, kw = k_scaled.shape
            
            # Match template
            res = cv2.matchTemplate(img_gray, k_scaled, cv2.TM_CCOEFF_NORMED)
            
            # Use analyze_ss.is_target_in_ss which is extremely fast and uses OpenCV morph dilation for NMS
            found, peaks = analyze_ss.is_target_in_ss(res, None, return_peaks=True, threshold=0.45, min_distance=15)
            
            # Filter peaks to only keep those on the left side (x < 300) to keep visualization clean
            peaks = [p for p in peaks if p[0] < 300]
            
            # Sort by score descending
            peaks.sort(key=lambda p: p[2], reverse=True)
            
            report_lines.append(f"  {name}:")
            for x, y, score in peaks[:5]:
                report_lines.append(f"    x={x}, y={y}, score={score:.4f}")
                # Draw on visualization image
                cv2.rectangle(viz_img, (x, y), (x + kw, y + kh), color, 2)
                cv2.putText(viz_img, f"{name}:{score:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
        out_path = f"debug_repro/matches_scale_{scale:.1f}.png"
        cv2.imwrite(out_path, viz_img)
        print(f"Saved visualization to {out_path}")
        
    with open("debug_repro/matches_report.txt", "w") as f:
        f.write("\n".join(report_lines))
    print("Saved text report to debug_repro/matches_report.txt")

if __name__ == "__main__":
    visualize()
