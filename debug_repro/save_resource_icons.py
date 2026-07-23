import cv2
import os

def crop_icons():
    img_path = 'debug_verification_resources/input_capture.png'
    if not os.path.exists(img_path):
        print("Error: Capture not found.")
        return
        
    img = cv2.imread(img_path)
    
    # We crop 40x40 regions around the suspected center of each icon (x ~ 35)
    # Suspected Y centers:
    # Food: 952, Wood: 1003, Gold: 1050, Stone: 1098, Silver: 1146
    
    crops = {
        'food': (952, 35),
        'wood': (1003, 35),
        'gold': (1050, 35),
        'stone': (1098, 35),
        'silver': (1146, 35)
    }
    
    for name, (cy, cx) in crops.items():
        # Crop 40x40 area centered at (cx, cy)
        y1 = max(0, cy - 20)
        y2 = min(img.shape[0], cy + 20)
        x1 = max(0, cx - 20)
        x2 = min(img.shape[1], cx + 20)
        
        crop = img[y1:y2, x1:x2]
        out_path = f"debug_repro/{name}_detected.png"
        cv2.imwrite(out_path, crop)
        print(f"Saved {name} icon crop to {out_path} (size {crop.shape})")

if __name__ == "__main__":
    crop_icons()
