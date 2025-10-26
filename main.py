import getSS
import analyzeSS
from PIL import Image
import cv2
import numpy as np
import os

# AOEMate -- Python code that analyzes live-streaming AOE4 game. First goal is to monitor the production queue to
# warn if no villages are being made.

if __name__ == "__main__":

    # Load kernel once
    vill_kernel = Image.open("/Users/harrisonmcadams/Desktop/villager_icon.png")

    # Prepare a small OpenCV window so we can poll for key presses
    cv2.namedWindow('AOEMate', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('AOEMate', 200, 100)

    # Polling interval in milliseconds for cv2.waitKey. Increase to reduce CPU usage.
    # Can be overridden by environment variable AOEMATE_POLL_MS.
    poll_ms = int(os.environ.get('AOEMATE_POLL_MS', '100'))

    # Determine centered window size safely. Avoid importing GUI toolkits that
    # may initialize Cocoa/AppKit on macOS (they can raise Objective-C exceptions
    # in some environments). Allow overriding via environment variables.
    try:
        screen_w = int(os.environ.get('AOEMATE_SCREEN_W', '1280'))
        screen_h = int(os.environ.get('AOEMATE_SCREEN_H', '800'))
    except Exception:
        screen_w, screen_h = 1280, 800

    win_w = int(int(os.environ.get('AOEMATE_WIN_W', str(int(screen_w * 0.5)))))
    win_h = int(int(os.environ.get('AOEMATE_WIN_H', str(int(screen_h * 0.5)))))
    win_x = int(os.environ.get('AOEMATE_WIN_X', str((screen_w - win_w) // 2)))
    win_y = int(os.environ.get('AOEMATE_WIN_Y', str((screen_h - win_h) // 2)))

    try:
        ii = 0
        while True:
            ii += 1
            # Set where to look on the screen
            gfn_region = {'top': 850, 'left': 0, 'width': 300, 'height': 350}

            # get screenshot
            screenshot = getSS.capture_gfn_screen_region(gfn_region)

            out_path = '/Users/harrisonmcadams/Desktop/'

            # Run the convolution / detection pipeline
            try:
                convolved_image = analyzeSS.convolveSSbyKernel(screenshot, vill_kernel)
                binary = analyzeSS.isTargetInSS(convolved_image, vill_kernel)
            except Exception as e:
                print('Analysis error:', e)
                binary = False

            if binary:
                print('Villagers are producing!')
            else:
                print('Villagers are NOT producing! :-(')

            # Display a large centered status window (green=producing, red=not)
            try:
                color = (0, 255, 0) if binary else (0, 0, 255)  # BGR
                status = np.full((win_h, win_w, 3), color, dtype=np.uint8)

                # Put centered label
                label = 'Producing' if binary else 'Not producing'
                font = cv2.FONT_HERSHEY_SIMPLEX
                # scale text relative to window size
                base_scale = max(1.0, min(win_w, win_h) / 400.0)
                thickness = max(2, int(base_scale))
                (tw, th), _ = cv2.getTextSize(label, font, base_scale * 2.0, thickness)
                tx = max(10, (win_w - tw) // 2)
                ty = max(30, (win_h + th) // 2)
                cv2.putText(status, label, (tx, ty), font, base_scale * 2.0, (255, 255, 255), thickness, cv2.LINE_AA)

                # Show and position the window centered on the screen
                cv2.imshow('AOEMate', status)
                cv2.resizeWindow('AOEMate', win_w, win_h)
                cv2.moveWindow('AOEMate', win_x, win_y)
            except Exception:
                # If anything fails, fallback to a small blank image to keep the window responsive
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
        cv2.destroyAllWindows()
