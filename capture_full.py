from mss import mss
from PIL import Image
import os

def capture_full():
    with mss() as sct:
        # Capture the primary monitor (or union of all if 0)
        # Using 0 to match the behavior of get_ss.py's monitor selection base
        mon = sct.monitors[0]
        
        output = os.path.expanduser("~/Desktop/debug_full_screen.png")
        
        print(f"Capturing full screen: {mon}")
        sct_img = sct.grab(mon)
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        img.save(output)
        print(f"Saved full screen capture to {output}")

if __name__ == "__main__":
    capture_full()
