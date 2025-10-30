"""
main.py

Interactive monitor for detecting a small on-screen icon (villager) using
template matching. The live loop captures a small screen region, runs a
matchTemplate pipeline in `analyze_ss`, and displays a large centered
status panel (green when the target is present, red otherwise).

This file focuses on application glue: region selection, invoking the
analyzer, and presenting a simple UI. All image-analysis specifics live in
`analyze_ss.py` and capture logic lives in `get_ss.py`.

Notes (important developer conventions):
- Coordinates throughout this code are in pixels with origin (0,0) at the
  top-left of the primary monitor. OpenCV and numpy arrays use (row, col)
  ordering internally but when presenting coordinates to humans we use
  (x, y) == (col, row) for clarity.
- All debug file-writing is optional and controlled by `out_path` values.
  Keep `out_path=None` during normal runs to avoid disk I/O.
"""

import get_ss  # screen-capture helper (captures named bboxes from config)
import analyze_ss  # analysis helpers (template matching and peak detection)
from PIL import Image  # image I/O (Pillow)
import cv2  # OpenCV for visualization and low-level image ops
import numpy as np  # numerical arrays (used by OpenCV pipelines)
import os  # environment variables and path helpers

# --------------------------------------------------------------------------------
# Live monitor loop
# --------------------------------------------------------------------------------


def are_vills_producing():
    """Run the live monitoring loop until the user quits.

    Steps per iteration:
      1. Capture a named screen region (via `get_ss.get_bbox`) -> PIL.Image
      2. Run template matching using `analyze_ss` -> match response map
      3. Detect peaks in the response map (boolean decision for live loop)
      4. Update a large centered color panel: GREEN when detected, RED when not
      5. Poll for 'q' or ESC to exit

    This function is explicitly fault tolerant: any exception raised during
    analysis is printed and the loop continues. This keeps the monitor up
    if occasional frames fail to capture or analyze.
    """

    # Load the villager icon template once to avoid repeatedly reading disk.
    # Keep this small and immutable so performance stays good.
    vill_kernel = Image.open("/Users/harrisonmcadams/Desktop/villager_icon.png")

    # Create an OpenCV window; we'll resize and move it to center it on screen.
    # The window is only used for display and to capture key events (cv2.waitKey).
    cv2.namedWindow('AOEMate', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('AOEMate', 200, 100)

    # Poll interval (ms) — controls how long cv2.waitKey blocks each loop.
    # Larger values reduce CPU usage but make reaction to changes slower.
    poll_ms = int(os.environ.get('AOEMATE_POLL_MS', '100'))

    # Determine screen geometry. We try several safe strategies and fall back
    # to environment-provided sizes or defaults. The detected `screen_w` and
    # `screen_h` are used only to center the status window.
    screen_w = screen_h = None
    try:
        import pyautogui
        sw, sh = pyautogui.size()
        screen_w, screen_h = int(sw), int(sh)
    except Exception:
        # pyautogui may not be installed — we'll try other fallbacks below.
        pass

    # macOS-specific fallback: use osascript to query Finder for desktop bounds
    # (safe subprocess call; avoids importing GUI frameworks that might crash).
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

    # final fallback: environment variables or sensible defaults
    if (screen_w is None or screen_h is None):
        try:
            screen_w = int(os.environ.get('AOEMATE_SCREEN_W', '1280'))
            screen_h = int(os.environ.get('AOEMATE_SCREEN_H', '800'))
        except Exception:
            screen_w, screen_h = 1280, 800

    # Default window size: 50% of detected/fallback screen size, centered.
    win_w = int(int(os.environ.get('AOEMATE_WIN_W', str(int(screen_w * 0.5)))))
    win_h = int(int(os.environ.get('AOEMATE_WIN_H', str(int(screen_h * 0.5)))))
    win_x = int(os.environ.get('AOEMATE_WIN_X', str((screen_w - win_w) // 2)))
    win_y = int(os.environ.get('AOEMATE_WIN_Y', str((screen_h - win_h) // 2)))

    try:
        ii = 0
        while True:
            ii += 1

            # -------------------------------
            # Capture
            # -------------------------------
            # `eco_summary` is a logical region name mapped in get_ss.get_bbox
            # (region definitions live in the get_ss module or its config).
            eco_summary = get_ss.get_bbox('eco_summary')

            # Capture the region (returns a PIL.Image). We do not save the
            # image by default; saving is controlled with `out_path` below.
            screenshot = get_ss.capture_gfn_screen_region(eco_summary)

            # -------------------------------
            # Analyze
            # -------------------------------
            # Keep debug output off during normal operation (no files written).
            out_path = None

            try:
                # Compute the template-matching response map. The analyzer returns
                # a 2D numpy array of match scores (float32). High values are better
                # matches (TM_CCOEFF_NORMED is typically in [-1,1]).
                convolved_image = analyze_ss.convolve_ssXkernel(screenshot, vill_kernel, out_path=out_path)

                # Decide whether the target exists; is_target_in_ss supports
                # returning peaks if requested. The simplest usage returns a
                # boolean. Here we call the boolean form for the live loop.
                binary = analyze_ss.is_target_in_ss(convolved_image, vill_kernel, out_path=out_path)

            except Exception as e:
                # Non-fatal: log and continue the loop so the monitor stays up.
                print('Analysis error:', e)
                binary = False

            # Console status (helps when running detached from the screen)
            if binary:
                print('Villagers are producing!')
            else:
                print('Villagers are NOT producing! :-(')

            # -------------------------------
            # Display status panel (large centered window)
            # -------------------------------
            try:
                # Single solid color background (BGR ordering for OpenCV)
                color = (0, 255, 0) if binary else (0, 0, 255)
                status = np.full((win_h, win_w, 3), color, dtype=np.uint8)

                # Put a readable label centered in the panel. Scale is relative
                # to the window size so the label is legible on different displays.
                label = 'Producing' if binary else 'Not producing'
                font = cv2.FONT_HERSHEY_SIMPLEX
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
                # If display fails (for example in headless CI), still poll keys
                # by showing a small blank window so the process can shut down
                # cleanly when requested.
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
        # Clean up the OpenCV window on exit to free OS resources
        cv2.destroyAllWindows()


# --------------------------------------------------------------------------------
# Region-specific summaries (one-shot helpers used in debugging / demos)
# --------------------------------------------------------------------------------


def summarize_eco():
    """Extract economic (eco) UI components from the `eco_summary` region.

    This function demonstrates a small pipeline to:
      1) capture the `eco_summary` region
      2) locate the gold icon via template matching (returns peaks)
      3) build small sub-bounding boxes around the icon and the number field
      4) capture those sub-regions for downstream OCR or parsing

    Notes about coordinates and peaks:
      - Peaks returned by `is_target_in_ss(..., return_peaks=True)` are tuples
        in the form (x, y, score). Coordinates are relative to the top-left of
        the image that was given to the matcher (here: the captured eco_summary
        screenshot). To convert a peak into absolute screen coordinates you
        add the top/left of the original bbox.
    """

    # Capture the named UI region. `get_ss.get_bbox` returns a dictionary of
    # the form {'top': ..., 'left': ..., 'width': ..., 'height': ...} which is
    # relative to the primary monitor. We pass that bbox to the capture helper
    # which returns a PIL.Image containing just that region.
    eco_summary = get_ss.get_bbox('eco_summary')
    screenshot = get_ss.capture_gfn_screen_region(eco_summary)

    # Debug output directory. When set to a string the analyzer will write
    # debug artifacts (convolved heatmap, annotated peaks, histograms).
    # Leave as None during normal operation to avoid disk writes.
    out_path = None
    out_path = '/Users/harrisonmcadams/Desktop/eco_summary_debug/'

    # Template path / kernel location
    kernelPath = '/Users/harrisonmcadams/Desktop/'

    # --- locate gold icon inside eco_summary ---
    gold_kernel = Image.open(kernelPath + 'gold_icon.png')

    # Run the matcher and request peaks (coordinates + scores). The
    # convolved_image is a 2D numpy array where each element corresponds to the
    # match score at that top-left alignment. Peaks are returned as (x,y,score)
    # tuples where x is horizontal offset (columns) and y is vertical offset
    # (rows) within the `screenshot` image.
    convolved_image = analyze_ss.convolve_ssXkernel(screenshot, gold_kernel, out_path=out_path)
    binary, peaks = analyze_ss.is_target_in_ss(convolved_image, gold_kernel, out_path=out_path, return_peaks=True)

    # Report peaks for quick inspection. Each `p` is (x, y, score) where x is
    # horizontal offset (columns) and y is vertical offset (rows) inside the
    # eco_summary screenshot.
    if peaks:
        print('Detected peaks (x,y,score):')
        for p in peaks:
            print(' -', p)

    # If we have at least one peak, construct small bboxes around the icon and
    # around the numeric count that typically sits to the right of the icon.
    # We use the first detected peak (peaks[0]) as the representative location.
    if not peaks:
        # Nothing to do further if we couldn't find an icon
        return

    # Base region (the eco_summary bbox on the full screen)
    eco_summary_bbox = get_ss.get_bbox('eco_summary')

    # Tuning values used to expand from icon center to final capture boxes
    fudge_factor = 2  # small padding in pixels around boxes
    height = gold_kernel.size[1]
    width = gold_kernel.size[0]

    # peaks[0] is (x, y, score) relative to the eco_summary image. To convert
    # to absolute screen coordinates, add the eco_summary's top/left offsets.
    # Note: x is columns (left-to-right), y is rows (top-to-bottom).
    peak_x, peak_y, peak_score = peaks[0]
    top = eco_summary_bbox['top'] + peak_y
    left = eco_summary_bbox['left'] + peak_x

    # Build the gold icon bbox. We add a small `fudge_factor` padding so the
    # crop includes a little margin around the icon (helps OCR and prevents
    # tight cropping when anti-aliasing shifts pixels).
    gold_bbox = {
        'top': int(top - fudge_factor),
        'left': int(left - fudge_factor),
        'width': int(width + fudge_factor * 2),
        'height': int(height + fudge_factor * 2)
    }

    # Build a bbox for the numeric count which typically appears to the right
    # of the icon. The left coordinate is offset by the icon width plus a small
    # gap. `count_width` must be tuned to accommodate the font & number of digits.
    count_width = 100  # adjust wider/narrower depending on font/spacing
    gold_count_bbox = {
        'top': int(top - fudge_factor),
        'left': int(left + width + 2),  # a small gap between icon and number
        'width': int(count_width),
        'height': int(height + fudge_factor * 2)
    }

    # Capture the numeric-count region (optionally saving to disk for debugging)
    gold_ss = get_ss.capture_gfn_screen_region(gold_count_bbox, out_path=out_path)


if __name__ == "__main__":
    # Example entry point: run the eco-summary demonstration
    summarize_eco()
