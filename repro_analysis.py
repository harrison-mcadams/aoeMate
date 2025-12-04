
import os
import cv2
import numpy as np
from PIL import Image
import analyze_ss
import main
import time

# Mock get_ss to return the static image
import get_ss

def test_current_implementation():
    images = [
        ("Macedonia (Silver)", os.path.expanduser("~/Desktop/debug_panel.png")),
        # ("Original (Gold)", os.path.expanduser("~/Desktop/debug_screenshot_q.png")),
    ]
    
    for label, img_path in images:
        print(f"\n--- Testing {label} ---")
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
    
        full_img = Image.open(img_path)
        print(f"Image size: {full_img.size}")

        # The current code in main.summarize_eco calls get_ss.get_bbox('eco_summary')
        # which returns {'top': 850, 'left': 0, 'width': 300, 'height': 350}
        # We need to crop this region from the full image to simulate capture_gfn_screen_region
        
        # bbox = get_ss.get_bbox('eco_summary')
        # Note: bbox is relative to monitor. Assuming the screenshot is the full monitor or at least contains this region at the correct offset.
        # If the screenshot is just the game window, we might need to adjust.
        # Let's assume the screenshot is full screen for now.
        

        # Now we can just pass the full image to summarize_eco
        # But summarize_eco expects the image to be the "eco_summary" crop if it captures it itself?
        # No, my new summarize_eco implementation:
        # 1. Takes `screenshot` arg.
        # 2. Converts it to grayscale: `ss_gray = np.array(screenshot.convert('L'), ...)`
        # 3. Finds anchors in `ss_gray`.
        
        # So if I pass the FULL image, `_find_anchors` will search the FULL image.
        # And the anchors will be relative to the FULL image.
        # And then the crops will be correct.
        
        # HOWEVER, `summarize_eco` has this line:
        # eco_summary = get_ss.get_bbox('eco_summary')
        # if screenshot is None: screenshot = get_ss.capture_gfn_screen_region(eco_summary)
        
        # If I pass `screenshot`, it uses it.
        # The `_find_anchors` logic searches `ss_gray`.
        # The `_CACHED_ANCHORS` will store coordinates relative to `screenshot`.
        
        # So yes, passing the full image should work, and it will find anchors in the full image.
        # This is actually BETTER because it makes the logic independent of the exact crop,
        # AS LONG AS the anchors are found.
        
        try:
            # We need to reset the cache because it's a global in main
            main._CACHED_ANCHORS = None
            
            start_time = time.time()
            # out_debug = os.path.expanduser(f"~/Desktop/aoe_debug_{label.split()[0]}")
            results = main.summarize_eco(screenshot=full_img)
            end_time = time.time()
            
            print(f"Results: {results}")
            print(f"Time taken: {end_time - start_time:.4f}s")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_current_implementation()
