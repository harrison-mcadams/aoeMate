import get_ss
import main
import logging

# Configure logging to console
logging.basicConfig(level=logging.INFO)

print("Testing 4K Monitor Capture and Detection...")

try:
    # 1. Test Capture
    print("Capturing eco_summary region...")
    bbox = get_ss.get_bbox('eco_summary')
    print(f"BBox: {bbox}")
    
    img = get_ss.capture_gfn_screen_region(bbox, out_path='debug_verification.png')
    print(f"Capture successful. Image size: {img.size}")
    
    # 2. Test Detection Logic (Villager Production)
    print("Running villager detection...")
    is_producing, score = main.check_villager_production(img)
    print(f"Detection Result: Producing={is_producing}, Score={score:.4f}")
    
    print("Verification Passed!")

except Exception as e:
    print(f"Verification Failed: {e}")
    import traceback
    traceback.print_exc()
