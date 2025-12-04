
import get_ss
import os

out_path = os.path.expanduser("~/Desktop/debug_screenshot_new.png")
bbox = get_ss.get_bbox('eco_summary')
print(f"Capturing region {bbox} to {out_path}...")
get_ss.capture_gfn_screen_region(bbox, out_path=out_path)
print("Done.")
