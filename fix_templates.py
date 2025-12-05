
import os
from PIL import Image
import numpy as np
import cv2

def fix_templates():
    img_path = os.path.expanduser("~/Desktop/debug_screenshot_new.png")
    try:
        img = Image.open(img_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # 1. Extract Silver Icon
    # Expected position based on others: x=14, y=344
    # Size: 30x30
    silver_crop = img.crop((14, 344, 14+30, 344+30))
    silver_path = os.path.expanduser("~/Desktop/silver_icon_macedonia.png")
    silver_crop.save(silver_path)
    print(f"Saved new silver template to {silver_path}")
    
    # 2. Extract Separator
    # Look at Food row: y=189.
    # Separator is usually a vertical bar or icon between Resource Count and Villager Count.
    # It's likely around x=100-130?
    # Let's crop a strip and try to find it using connected components or just save a region?
    # Since I can't see it, I'll try to find the *villager icon* in the row?
    # No, the separator is the "villager head" icon usually?
    # Wait, `villager_separator.png` is the name.
    # If the separator IS the villager icon, then I should look for the villager icon!
    
    # Let's look for the villager icon in the Food row.
    # We have `villager_icon.png` (the queue icon). Is it the same?
    # Usually the eco panel uses a different, smaller icon or a separator line.
    # The code calls it `villager_separator.png`.
    
    # Let's try to find a high-contrast object in the expected separator region.
    # Region: x=80 to x=150, y=189 to y=219.
    strip = img.crop((80, 189, 150, 219))
    # Convert to numpy
    strip_arr = np.array(strip.convert('L'))
    
    # Threshold to find dark/light object
    # The UI is likely dark, text/icons light.
    _, thresh = cv2.threshold(strip_arr, 150, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour that looks like a separator (vertical-ish or icon-ish)
    best_cnt = None
    max_area = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 10 and w > 5: # Reasonable size
            if w * h > max_area:
                max_area = w * h
                best_cnt = (x, y, w, h)
    
    if best_cnt:
        x, y, w, h = best_cnt
        # Crop from strip
        # Add padding?
        sep_crop = strip.crop((x, y, x+w, y+h))
        sep_path = os.path.expanduser("~/Desktop/villager_separator.png")
        sep_crop.save(sep_path)
        print(f"Saved new separator template to {sep_path} (from crop {x},{y} {w}x{h})")
    else:
        print("Could not find separator candidate in strip.")
        # Fallback: Save the whole strip so user can see?
        # Or just try to crop a fixed region?
        # Let's crop a fixed region where it *should* be.
        # x=115, y=0 (relative to strip start x=80 -> x=195 absolute)
        # Maybe x=110?
        pass

if __name__ == "__main__":
    fix_templates()
