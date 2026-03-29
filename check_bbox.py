import get_ss
import os
import datetime

# Clean up any old debug files just in case
try:
    if os.path.exists('debug_bbox_check.png'):
        os.remove('debug_bbox_check.png')
except:
    pass

print("Capturing with WIDENED BBox (1200px)...")
bbox = get_ss.get_bbox('eco_summary')
img = get_ss.capture_gfn_screen_region(bbox, out_path='debug_bbox_check.png')
print(f"Captured size: {img.size}")
print("Saved to debug_bbox_check.png")
