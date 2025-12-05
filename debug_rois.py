
import main
import cv2
import numpy as np
import os
from PIL import Image
import analyze_ss

def debug_rois():
    img_path = os.path.expanduser("~/Desktop/debug_screenshot_new.png")
    try:
        img = Image.open(img_path)
        print(f"Loaded image: {img.size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Convert to grayscale numpy array
    img_gray = np.array(img.convert('L'), dtype=np.float32)
    
    # Load resource kernels manually
    resource_kernels = {}
    resources = [
        ('food', 'food_icon.png'),
        ('wood', 'wood_icon.png'),
        ('gold', 'gold_icon.png'),
        ('stone', 'stone_icon.png'),
        ('silver', 'silver_icon_macedonia.png'),
        ('food', 'food_icon_macedonia.png'),
        ('wood', 'wood_icon_macedonia.png'),
        ('stone', 'stone_icon_macedonia.png'),
        ('villager_separator', 'villager_separator.png')
    ]
    
    kernel_path = os.environ.get('AOE_KERNEL_PATH', '/Users/harrisonmcadams/Desktop/')
    
    print(f"Loading kernels from {kernel_path}...")
    for _, fname in resources:
        try:
            k_path = os.path.join(kernel_path, fname)
            if os.path.exists(k_path):
                k = Image.open(k_path)
                resource_kernels[fname] = np.array(k.convert('L'), dtype=np.float32)
            else:
                print(f"Kernel not found: {k_path}")
        except Exception as e:
            print(f"Error loading {fname}: {e}")

    print("Finding anchors...")
    anchors = main._find_anchors(img_gray, resource_kernels)
    print(f"Anchors found: {anchors}")
    
    debug_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Replicate summarize_eco logic for ROIs
    fudge_factor = 4
    sw, sh = img.size
    
    for name, (ax, ay) in anchors.items():
        # Get width/height of the icon kernel
        # We need to find which kernel matched. For simplicity, assume standard size or look up.
        # main.py uses 'food_icon.png' etc.
        # Let's just use a default w=30, h=30 for visualization if we can't easily get it.
        # Or better, check which key in resource_kernels matches 'name'.
        
        # In main.py, it iterates resources.
        # Let's just use w=30, h=30.
        w, h = 30, 30
        
        # Separator detection logic
        separator_found = False
        sep_x_rel = 0
        sep_w_found = 0
        
        # Search for separator to the right of the icon
        sep_search_left = ax + w
        sep_search_right = min(sw, ax + 200) # Look up to 200px right
        sep_search_top = ay - 5
        sep_search_bottom = ay + h + 5
        
        if sep_search_right > sep_search_left:
            try:
                sep_strip = img_gray[sep_search_top:sep_search_bottom, sep_search_left:sep_search_right]
                k_sep = resource_kernels.get('villager_separator.png')
                if k_sep is not None:
                    res_conv = analyze_ss.match_template_arrays(sep_strip, k_sep)
                    found, peaks = analyze_ss.is_target_in_ss(res_conv, None, return_peaks=True, threshold=0.6)
                    if found and peaks:
                        peaks.sort(key=lambda p: p[2], reverse=True)
                        px, py, _ = peaks[0]
                        separator_found = True
                        print(f"Separator found for {name} at relative x={px}")
                        sep_x_rel = int(px)
                        sep_w_found = k_sep.shape[1]
                        
                        # Draw separator
                        sep_abs_x = sep_search_left + sep_x_rel
                        sep_abs_y = sep_search_top + int(py)
                        cv2.rectangle(debug_img, (sep_abs_x, sep_abs_y), (sep_abs_x + sep_w_found, sep_abs_y + k_sep.shape[0]), (255, 0, 255), 1)
            except Exception as e:
                print(f"Error searching separator for {name}: {e}")

        # Define ROIs
        if separator_found:
            sep_abs_x = sep_search_left + sep_x_rel
            
            res_roi_left = ax + w - 5
            res_roi_right = sep_abs_x - 2
            res_roi_top = ay - fudge_factor
            res_roi_bottom = ay + h + fudge_factor
            
            vill_roi_left = sep_abs_x + sep_w_found + 2
            vill_roi_right = vill_roi_left + 50
            vill_roi_top = ay - fudge_factor
            vill_roi_bottom = ay + h + fudge_factor
        else:
            res_roi_left = ax + w - 5 
            res_roi_top = ay - fudge_factor
            res_roi_w = 75 
            res_roi_h = h + fudge_factor * 2
            res_roi_right = res_roi_left + res_roi_w
            res_roi_bottom = res_roi_top + res_roi_h
            
            vill_roi_left = ax + w + 80 
            vill_roi_top = ay - fudge_factor
            vill_roi_w = 50 
            vill_roi_h = h + fudge_factor * 2
            vill_roi_right = vill_roi_left + vill_roi_w
            vill_roi_bottom = vill_roi_top + vill_roi_h
            
        # Draw ROIs
        # Resource: Blue
        cv2.rectangle(debug_img, (int(res_roi_left), int(res_roi_top)), (int(res_roi_right), int(res_roi_bottom)), (255, 0, 0), 2)
        # Villager: Green
        cv2.rectangle(debug_img, (int(vill_roi_left), int(vill_roi_top)), (int(vill_roi_right), int(vill_roi_bottom)), (0, 255, 0), 2)
        
        # Draw Anchor: Red
        cv2.rectangle(debug_img, (ax, ay), (ax + w, ay + h), (0, 0, 255), 2)
        cv2.putText(debug_img, name, (ax, ay - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    out_path = os.path.expanduser("~/Desktop/debug_rois.png")
    cv2.imwrite(out_path, debug_img)
    print(f"Saved debug image to {out_path}")

if __name__ == "__main__":
    debug_rois()
