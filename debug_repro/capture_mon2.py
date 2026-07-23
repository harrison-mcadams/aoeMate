import get_ss
import os

def capture():
    get_ss.set_monitor_index(2)
    bbox = get_ss.get_bbox('eco_summary')
    out_path = 'debug_verification_resources/input_capture_mon2.png'
    img = get_ss.capture_gfn_screen_region(bbox, out_path=out_path)
    print(f"Captured Monitor 2 screenshot to {out_path}, size = {img.size}")

if __name__ == "__main__":
    capture()
