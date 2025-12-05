
import main
import cv2
import numpy as np
import os
from PIL import Image

def debug_digits():
    img_path = os.path.expanduser("~/Desktop/debug_screenshot_new.png")
    try:
        img = Image.open(img_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Food Anchor: 14, 189
    # Separator relative x: 83 (from debug_rois.py)
    # Icon width: 30
    
    ax, ay = 14, 189
    w = 30
    sep_x_rel = 83
    sep_w = 6 # from fix_templates.py
    
    sep_abs_x = ax + w + sep_x_rel # 14 + 30 + 83 = 127
    
    # Resource ROI
    res_roi_left = ax + w - 5 # 39
    res_roi_right = sep_abs_x - 2 # 125
    res_roi_top = ay - 4
    res_roi_bottom = ay + 30 + 4
    
    # Villager ROI
    vill_roi_left = sep_abs_x + sep_w + 2 # 127 + 6 + 2 = 135
    vill_roi_right = vill_roi_left + 50 # 185
    vill_roi_top = ay - 4
    vill_roi_bottom = ay + 30 + 4
    
    print(f"Food Res ROI: {res_roi_left}, {res_roi_top}, {res_roi_right}, {res_roi_bottom}")
    print(f"Food Vill ROI: {vill_roi_left}, {vill_roi_top}, {vill_roi_right}, {vill_roi_bottom}")
    
    res_img = img.crop((res_roi_left, res_roi_top, res_roi_right, res_roi_bottom))
    vill_img = img.crop((vill_roi_left, vill_roi_top, vill_roi_right, vill_roi_bottom))
    
    res_img.save(os.path.expanduser("~/Desktop/debug_food_res.png"))
    vill_img.save(os.path.expanduser("~/Desktop/debug_food_vill.png"))
    
    # Load digit kernels
    # We need to initialize them. main._init_kernels_and_executors needs a list.
    # But we can just manually load them like in debug_anchors.py if we want, 
    # OR use main._parse_number_from_region which expects a dict of kernels.
    
    digit_kernels_gray = {}
    kernel_path = os.environ.get('AOE_KERNEL_PATH', '/Users/harrisonmcadams/Desktop/')
    for d in range(10):
        try:
            k = Image.open(os.path.join(kernel_path, f'{d}.png'))
            digit_kernels_gray[d] = np.array(k.convert('L'), dtype=np.float32)
        except Exception:
            pass
            
    print("Parsing Resource Count...")
    res_val = main._parse_number_from_region(res_img, digit_kernels_gray, out_path=os.path.expanduser("~/Desktop/"), name="debug_food_res")
    print(f"Resource Count: {res_val}")
    
    print("Parsing Villager Count...")
    vill_val = main._parse_number_from_region(vill_img, digit_kernels_gray, out_path=os.path.expanduser("~/Desktop/"), name="debug_food_vill")
    print(f"Villager Count: {vill_val}")

if __name__ == "__main__":
    debug_digits()
