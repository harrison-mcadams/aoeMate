from mss import mss
from PIL import Image
from pathlib import Path
from typing import Optional
import os

def get_bbox(behavior):
    """
    Returns a bounding box dictionary for the given behavior.
    Args:
        behavior (str): The behavior name. Supported: 'gfn_menu', 'gfn_in_game'.
    Returns:
        dict: A dictionary with keys 'top', 'left', 'width', 'height'.
    Raises:
        ValueError: If the behavior is not recognized.
    """
    if behavior == 'eco_summary':
        # Adjusted based on user feedback: 
        # - Top moved up (was 850, now 600) to capture more above.
        # - Width reduced (was 600, now 400) to avoid excess.
        # - Height adjusted (was 350) to fit 1080p screen (600+480=1080).
        # Dynamic check is tricky without passing monitor info, but let's assume standard layout.
        # Ideally, we should detect resolution. For now, rely on default bbox being for 1080p height
        # unless overridden or we can detect context. 
        # Actually, get_bbox is called before we pick the monitor in main(), but we can check resolution here if we want.
        # But capture_gfn_screen_region picks the monitor.
        # Let's RETURN a "relative" box or handle scaling in capture.
        # Better yet, let capture helper pass the resolution to this function?
        # For simplicity in this quick fix script:
        return {'top': 600, 'left': 0, 'width': 400, 'height': 480}
    elif behavior == 'gfn_in_game':
        return {'top': 850, 'left': 0, 'width': 300, 'height': 350}
    # elif behavior == 'global_queue':
    #     # REMOVED: User indicates queue is in the top of eco_summary
    #     pass
    else:
        raise ValueError(f"Unrecognized behavior: {behavior}")

def capture_gfn_screen_region(bbox, *, out_path: Optional[str] = None):
    """
    Captures a screenshot of a specific region of the screen.
    Args:
        bbox (dict): A dictionary with keys 'top', 'left', 'width', 'height'.
        out_path (str, optional): If provided, the captured image will be saved
            to this path. If None, no file will be written.
    Returns:
        PIL.Image: the captured image
    """
    with mss() as sct:
        # User requested left monitor, which seems to be index 2 based on debug output.
        # Allow override via AOE_MONITOR_INDEX
        try:
             mon_idx = int(os.environ.get('AOE_MONITOR_INDEX', 2))
        except ValueError:
             mon_idx = 2
        
        if mon_idx >= len(sct.monitors):
            mon_idx = 0 # Fallback
            
        mon = sct.monitors[mon_idx]
        # logging.info(f"Capturing from monitor {mon_idx}: {mon}")
        
        # Adjust bbox for 4K if detected (Height > 1440)
        # Note: bbox passed in is hardcoded for 1080p (Height 1080)
        # 1080p: top=600 (bottom-480). 
        # 4K (2160p): Equivalent bottom region would be top = 2160 - X.
        # UI Scaling usually makes it bigger. Let's start with a heuristic:
        # If 4K, aim for bottom left with proportionate or fixed size.
        
        adjusted_bbox = bbox.copy()
        if mon['height'] > 1440:
             # Assuming 4K
             # Position at bottom left.
             # Standard capture height was 480. 
             # On 4K with scaling, it might be roughly same pixel count if 'pixel perfect', 
             # but usually UI is scaled 2x so it takes more pixels? 
             # Or UI is same visual size, so 2x pixels.
             # Let's try grabbing a larger chunk at the bottom left.
             # Let's try grabbing a larger chunk at the bottom left.
             adjusted_bbox['height'] = 1200  # Increased height
             adjusted_bbox['width'] = 1600   # Increased width to catch everything
             # adjusted_bbox['top'] = mon['height'] - adjusted_bbox['height'] - 100 # buffer from bottom? 
             # Actually, AoE4 UI usually flush bottom.
             adjusted_bbox['top'] = mon['height'] - adjusted_bbox['height']
             
        monitor_bbox = {
            'top': mon['top'] + adjusted_bbox['top'],
            'left': mon['left'] + adjusted_bbox['left'],
            'width': adjusted_bbox['width'],
            'height': adjusted_bbox['height']
        }

        # The bounding box to capture
        sct_img = sct.grab(monitor_bbox)
        # Convert the raw pixels to a PIL Image
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

        # If an output path was provided, save the image there; otherwise do not write
        if out_path:
            try:
                out_p = Path(out_path)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                img.save(str(out_p))
            except Exception as e:
                # Surface the error to the caller but keep the image in memory
                raise

        return img

def main(target='eco_summary', out_path=None):
    """
    High-level wrapper to capture a target region and save it to disk.
    Args:
        target (str): The behavior/target region name.
        out_path (str, optional): The path to save the screenshot. 
                                  If None, saves to a temporary file or fixed location.
    Returns:
        str: The absolute path to the saved screenshot, or None if failed.
    """
    try:
        bbox = get_bbox(target)
        if not out_path:
            # Default to a fixed path in temp or current directory if not specified
            # For this app, let's just use the target name in the current dir or temp
            # But main.py loop calls this repeatedly, so overwriting is fine/desired.
            import tempfile
            fname = f"aoe_capture_{target}.png"
            out_path = os.path.join(tempfile.gettempdir(), fname)
        
        capture_gfn_screen_region(bbox, out_path=out_path)
        return out_path
    except Exception as e:
        print(f"Error in get_ss.main({target}): {e}")
        return None


if __name__ == "__main__":
    # Basic self-check when executed as a script: capture a demo region and save it.
    demo_bbox = {'top': 100, 'left': 100, 'width': 400, 'height': 300}
    try:
        home = Path.home()
        desktop = home / "Desktop"
        out_dir = desktop if desktop.exists() else home
        out_path = str(out_dir / "debug_screenshot_test.png")

        img = capture_gfn_screen_region(demo_bbox, out_path=out_path)
        print(f"Capture succeeded — saved test image to: {out_path}")
    except Exception as e:
        print("Capture failed:", e)
