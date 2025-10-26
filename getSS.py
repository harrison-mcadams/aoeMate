from mss import mss
from PIL import Image
from pathlib import Path

def capture_gfn_screen_region(bbox):
    """
    Captures a screenshot of a specific region of the screen.
    Args:
        bbox (dict): A dictionary with keys 'top', 'left', 'width', 'height'.
    """
    with mss() as sct:
        mon = sct.monitors[0]  # Use the primary monitor

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

        # Determine an output path based on the current user's home directory.
        # Prefer the Desktop folder if it exists, otherwise fall back to the home directory.
        home = Path.home()
        desktop = home / "Desktop"
        output_dir = desktop if desktop.exists() else home

        output_filename = str(output_dir / "debug_screenshot.png")
        img.save(output_filename)
        return img
