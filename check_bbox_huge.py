import get_ss
import os
import datetime

# Clean up
try:
    if os.path.exists('debug_bbox_huge.png'):
        os.remove('debug_bbox_huge.png')
except:
    pass

print("Capturing with HUGE BBox (1600x1200)...")
bbox = get_ss.get_bbox('eco_summary')
img = get_ss.capture_gfn_screen_region(bbox, out_path='debug_bbox_huge.png')
print(f"Captured size: {img.size}")
print("Saved to debug_bbox_huge.png")
