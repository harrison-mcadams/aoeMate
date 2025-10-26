"""
main.py

Interactive monitor that repeatedly captures a small region of the screen,
runs a template-matching based detector, and displays a large centered
status window (green when the target is detected, red otherwise).

This file intentionally contains few configuration knobs; most are provided
via environment variables documented in README.md (for example,
`AOEMATE_POLL_MS`, `AOEMATE_SCREEN_W`, `AOEMATE_SCREEN_H`).

Behavior is: load the kernel image once, then loop: capture -> analyze -> show status.
Press 'q' or Esc in the status window to exit the loop cleanly.
"""

import getSS
import analyzeSS
from PIL import Image
import cv2
import numpy as np
import os

# AOEMate -- Python code that analyzes live-streaming AOE4 game. First goal is to monitor the production queue to
# warn if no villages are being made.


def areVillsProducing():
    """Run the live monitoring loop until the user quits.

    This function performs the following steps (repeat until quit):
      1. Capture a small screen region using `getSS.capture_gfn_screen_region`.
      2. Run `analyzeSS.convolveSSbyKernel` and `analyzeSS.isTargetInSS` to
         decide whether the target (villager icon) is present.
      3. Update a large centered OpenCV window with a green (detected) or red
         (not detected) full-screen panel and a text label.

    Configuration is read via environment variables (see README.md):
      - AOEMATE_POLL_MS : polling interval in milliseconds (default 100)
      - AOEMATE_SCREEN_W / AOEMATE_SCREEN_H : screen size fallback
      - AOEMATE_WIN_W / AOEMATE_WIN_H / AOEMATE_WIN_X / AOEMATE_WIN_Y : override window geometry

    Note: this function intentionally does not write debug images by default
    (out_path is set to None). To enable debug outputs, set `out_path` to a
    directory path when calling the analysis functions.
    """

    # Load kernel once (template used for matchTemplate). Keep this outside the loop
    # so we avoid repeatedly reading from disk.
    vill_kernel = Image.open("/Users/harrisonmcadams/Desktop/villager_icon.png")

    # Create a resizable OpenCV window used solely for key polling and display.
    cv2.namedWindow('AOEMate', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('AOEMate', 200, 100)

    # Poll interval (ms) used by cv2.waitKey; increasing this reduces CPU usage.
    poll_ms = int(os.environ.get('AOEMATE_POLL_MS', '100'))

    # Determine a centered window size. We attempt several safe methods in order
    # and fall back to environment variables or defaults if detection fails.
    screen_w = screen_h = None
    try:
        import pyautogui
        sw, sh = pyautogui.size()
        screen_w, screen_h = int(sw), int(sh)
    except Exception:
        pass

    # On macOS, a safe fallback is to query Finder for desktop bounds using
    # osascript; this avoids importing GUI toolkits that may initialize AppKit.
    if (screen_w is None or screen_h is None) and (os.sys.platform == 'darwin'):
        try:
            import subprocess
            out = subprocess.check_output([
                'osascript', '-e', 'tell application "Finder" to get bounds of window of desktop'
            ], text=True)
            parts = [int(p.strip()) for p in out.strip().split(',')]
            if len(parts) >= 4:
                screen_w = parts[2]
                screen_h = parts[3]
        except Exception:
            pass

    # Final fallback: environment variables or defaults
    if (screen_w is None or screen_h is None):
        try:
            screen_w = int(os.environ.get('AOEMATE_SCREEN_W', '1280'))
            screen_h = int(os.environ.get('AOEMATE_SCREEN_H', '800'))
        except Exception:
            screen_w, screen_h = 1280, 800

    # Window geometry (defaults to 50% of screen size, centered)
    win_w = int(int(os.environ.get('AOEMATE_WIN_W', str(int(screen_w * 0.5)))))
    win_h = int(int(os.environ.get('AOEMATE_WIN_H', str(int(screen_h * 0.5)))))
    win_x = int(os.environ.get('AOEMATE_WIN_X', str((screen_w - win_w) // 2)))
    win_y = int(os.environ.get('AOEMATE_WIN_Y', str((screen_h - win_h) // 2)))

    try:
        ii = 0
        # Main loop: keep running until the user quits with 'q' or Esc
        while True:
            ii += 1
            # Region to capture. Tweak this bbox for your screen layout.
            gfn_region = {'top': 850, 'left': 0, 'width': 300, 'height': 350}

            # Capture the region. We do not save to disk here (out_path=None) by default.
            screenshot = getSS.capture_gfn_screen_region(gfn_region)

            # Disable debug saving during normal operation; set to a directory string
            # to enable debug outputs (heatmaps, histograms, etc.).
            out_path = None

            # Analysis step: compute matchTemplate response and detect peaks.
            try:
                convolved_image = analyzeSS.convolveSSbyKernel(screenshot, vill_kernel, out_path=out_path)
                binary = analyzeSS.isTargetInSS(convolved_image, vill_kernel, out_path=out_path)
            except Exception as e:
                # Analysis errors should not crash the loop; report and continue.
                print('Analysis error:', e)
                binary = False

            # Log a short textual status in the console too (useful when running in background)
            if binary:
                print('Villagers are producing!')
            else:
                print('Villagers are NOT producing! :-(')

            # Build a large single-color status image and overlay a centered label.
            # Green indicates detected, red indicates not detected.
            try:
                color = (0, 255, 0) if binary else (0, 0, 255)  # BGR order for OpenCV
                status = np.full((win_h, win_w, 3), color, dtype=np.uint8)

                # Put a readable label in the middle of the panel. The text scale is
                # chosen relative to the window size so it remains legible at different sizes.
                label = 'Producing' if binary else 'Not producing'
                font = cv2.FONT_HERSHEY_SIMPLEX
                base_scale = max(1.0, min(win_w, win_h) / 400.0)
                thickness = max(2, int(base_scale))
                (tw, th), _ = cv2.getTextSize(label, font, base_scale * 2.0, thickness)
                tx = max(10, (win_w - tw) // 2)
                ty = max(30, (win_h + th) // 2)
                cv2.putText(status, label, (tx, ty), font, base_scale * 2.0, (255, 255, 255), thickness, cv2.LINE_AA)

                # Show and position the status window in the computed center position.
                cv2.imshow('AOEMate', status)
                cv2.resizeWindow('AOEMate', win_w, win_h)
                cv2.moveWindow('AOEMate', win_x, win_y)
            except Exception:
                # If display fails for any reason (headless environment), show a small blank
                # window so we can still poll for key presses and quit cleanly.
                blank = 255 * np.ones((200, 300, 3), dtype='uint8')
                cv2.imshow('AOEMate', blank)

            # Poll for key press: 'q' or Esc to quit
            key = cv2.waitKey(poll_ms) & 0xFF
            if key == ord('q') or key == 27:
                print('Quit key pressed - exiting loop')
                break

    except KeyboardInterrupt:
        print('Interrupted by user')
    finally:
        # Ensure the OpenCV window is destroyed on exit so the OS cleans up resources.
        cv2.destroyAllWindows()


if __name__ == "__main__":

    areVillsProducing()
