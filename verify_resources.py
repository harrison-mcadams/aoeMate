import get_ss
import main
import logging
import os
import json

# Configure logging to console
logging.basicConfig(level=logging.INFO)

print("Testing Resource Detection...")

try:
    # Create debug directory
    debug_dir = "debug_verification_resources"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    # 1. Test Capture
    print("Capturing eco_summary region...")
    bbox = get_ss.get_bbox('eco_summary')
    print(f"BBox: {bbox}")
    
    img = get_ss.capture_gfn_screen_region(bbox, out_path=os.path.join(debug_dir, 'input_capture.png'))
    print(f"Capture successful. Image size: {img.size}")
    
    # 2. Run summarize_eco
    print("Analyzing resources...")
    results = main.summarize_eco(screenshot=img, out_path=debug_dir)
    
    print("\nDetection Results:")
    print(json.dumps(results, indent=2))
    
    print(f"\nDebug images saved to {os.path.abspath(debug_dir)}")
    print("Please check 'debug_separator_viz.png' in that folder to visualize the detection regions.")

except Exception as e:
    print(f"Verification Failed: {e}")
    import traceback
    traceback.print_exc()
