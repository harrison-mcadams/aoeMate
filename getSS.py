from mss import mss
from PIL import Image
from pathlib import Path
from typing import Optional


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
