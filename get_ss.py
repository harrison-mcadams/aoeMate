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

        monitor_bbox = {
            'top': mon['top'] + bbox['top'],
            'left': mon['left'] + bbox['left'],
            'width': bbox['width'],
            'height': bbox['height']
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
