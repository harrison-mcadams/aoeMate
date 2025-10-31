"""
main.py

Interactive monitor for detecting a small on-screen icon (villager) using
template matching. The live loop captures a small screen region, runs a
matchTemplate pipeline in `analyze_ss`, and displays a large centered
status panel (green when the target is present, red otherwise).

This file focuses on application glue: region selection, invoking the
analyzer, and presenting a simple UI. All image-analysis specifics live in
`analyze_ss.py` and capture logic lives in `get_ss.py`.

Developer notes (quick):
- To tune detection, edit parameters inside `analyze_ss.py` (thresholds, min_distance)
  or change the template images placed on the Desktop (villager_icon.png, gold_icon.png).
- Keep `out_path=None` during normal runs to avoid writing debug images. Enable a path
  when you want visual debugging artifacts written to disk.

Coordinate conventions:
- BBoxes in `get_ss` and this file use a dict: {'top', 'left', 'width', 'height'}
  where top/left are pixel offsets from the PRIMARY monitor's top-left.
- Template matcher `is_target_in_ss` returns peaks as (x, y, score) where x is
  horizontal offset (columns) and y is vertical offset (rows) relative to the
  input image to the matcher (not absolute screen coords).
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

    # ------------------------------------------------------------------
    # Configuration / assets
    # ------------------------------------------------------------------
    # The kernel (template) file path is currently hard-coded to the Desktop.
    # Replace this with your own path or wire it to a config if you move files.
    vill_kernel = Image.open("/Users/harrisonmcadams/Desktop/villager_icon.png")

    # Create an OpenCV window; we'll resize and move it to center it on screen.
    # The window is also used to capture keyboard events via cv2.waitKey.
    cv2.namedWindow('AOEMate', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('AOEMate', 200, 100)

    # Poll interval (ms) controls how often the loop yields to the GUI event
    # system. Larger values reduce CPU usage at the cost of responsiveness.
    poll_ms = int(os.environ.get('AOEMATE_POLL_MS', '100'))

    # ------------------------------------------------------------------
    # Compute window placement (try a few safe methods, avoid heavy GUI libs)
    # ------------------------------------------------------------------
    screen_w = screen_h = None
    try:
        # pyautogui is convenient but optional; we try it first when available.
        import pyautogui
        sw, sh = pyautogui.size()
        screen_w, screen_h = int(sw), int(sh)
    except Exception:
        # Not installed or failed — we'll fall back below.
        pass

    # macOS: try osascript via subprocess to avoid Tk/Cocoa initialization.
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

    # Final fallback: environment variables (useful in multi-monitor or CI setups)
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

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    try:
        ii = 0
        while True:
            ii += 1

            # -------------------------------
            # Capture stage
            # -------------------------------
            # get_ss.get_bbox returns a dict with keys: top,left,width,height
            eco_summary = get_ss.get_bbox('eco_summary')

            # capture_gfn_screen_region returns a PIL Image (RGB). By default the
            # capture helper will NOT write a file unless `out_path` is provided.
            screenshot = get_ss.capture_gfn_screen_region(eco_summary)

            # -------------------------------
            # Analyze stage
            # -------------------------------
            # Keep debug output off during normal operation (set `out_path` to a
            # directory path to enable diagnostic files: heatmap, peaks, histogram)
            out_path = None

            try:
                # convolve_ssXkernel returns a 2D numpy array (float32) of match
                # scores; higher is better for TM_CCOEFF_NORMED.
                convolved_image = analyze_ss.convolve_ssXkernel(screenshot, vill_kernel, out_path=out_path)

                # The simplest API usage: return boolean true/false for detection.
                # If you need coordinates use return_peaks=True in is_target_in_ss.
                binary = analyze_ss.is_target_in_ss(convolved_image, vill_kernel, out_path=out_path)

            except Exception as e:
                # Analysis must be fault tolerant — log errors and continue.
                print('Analysis error:', e)
                binary = False

            # -------------------------------
            # Reporting / Display
            # -------------------------------
            # Console status (useful when running headless or via SSH)
            if binary:
                print('Villagers are producing!')
            else:
                print('Villagers are NOT producing! :-(')

            # Visual status panel: full color background + centered label
            try:
                color = (0, 255, 0) if binary else (0, 0, 255)  # BGR
                status = np.full((win_h, win_w, 3), color, dtype=np.uint8)

                # Draw text roughly centered. We compute size once per frame;
                # for higher performance you could precompute font scale thresholds.
                label = 'Producing' if binary else 'Not producing'
                font = cv2.FONT_HERSHEY_SIMPLEX
                base_scale = max(1.0, min(win_w, win_h) / 400.0)
                thickness = max(2, int(base_scale))
                (tw, th), _ = cv2.getTextSize(label, font, base_scale * 2.0, thickness)
                tx = max(10, (win_w - tw) // 2)
                ty = max(30, (win_h + th) // 2)
                cv2.putText(status, label, (tx, ty), font, base_scale * 2.0, (255, 255, 255), thickness, cv2.LINE_AA)

                # Present the window and position it centered on the screen
                cv2.imshow('AOEMate', status)
                cv2.resizeWindow('AOEMate', win_w, win_h)
                cv2.moveWindow('AOEMate', win_x, win_y)
            except Exception:
                # If we can't display (headless), keep a tiny window so waitKey works
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
        # Ensure OpenCV window is closed so the OS can reclaim resources
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

    # Template path / kernel location (Desktop by convention for this project)
    kernelPath = '/Users/harrisonmcadams/Desktop/'

    # --- locate gold icon inside eco_summary ---
    # Note: the kernel images are expected to be small (e.g. 32x32 or 48x48)
    # and should match the on-screen icon scale. If detection fails, try
    # providing a higher-resolution kernel or rescaling the screenshot before
    # matching.
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
    fudge_factor = 10  # small padding in pixels around boxes
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
    # NOTE: count_width is a heuristic. If your UI uses larger fonts or more
    # digits, increase this value. Consider measuring a few examples and
    # computing an adaptive width based on font size instead of a constant.
    count_width = 100  # adjust wider/narrower depending on font/spacing
    gold_count_bbox = {
        'top': int(top - fudge_factor),
        'left': int(left + width + 2),  # a small gap between icon and number
        'width': int(count_width),
        'height': int(height + fudge_factor * 2)
    }

    # gold_ss is the cropped image containing the numeric count. The image
    # returned is a PIL.Image (RGB). Downstream OCR/parsing routines should
    # convert to grayscale and optionally apply thresholding before OCR.
    # Capture the numeric-count region (optionally saving to disk for debugging)
    gold_box_ss = get_ss.capture_gfn_screen_region(gold_count_bbox, out_path=out_path)

    # Collect digit detections across 0-9, then assemble them left-to-right.
    all_digit_peaks = []  # list of (x, digit, score)
    digits_to_check = list(range(10))
    per_digit_min_distance = 5

    for digit in digits_to_check:
        try:
            number_kernel = Image.open(kernelPath + f'{digit}.png')
        except Exception:
            # Skip missing digit templates rather than crash
            continue

        # Match the single-digit kernel inside the cropped number area
        convolved_image = analyze_ss.convolve_ssXkernel(gold_box_ss, number_kernel, out_path=out_path)

        # Request peaks for this digit
        found, peaks = analyze_ss.is_target_in_ss(convolved_image, number_kernel, out_path=out_path, return_peaks=True, min_distance=per_digit_min_distance)
        if not found:
            continue

        # Each peak is (x, y, score) relative to gold_box_ss; keep x, digit, score
        for (px, py, pscore) in peaks:
            all_digit_peaks.append((int(px), int(digit), float(pscore)))

    # If no digit detections, we're done
    if not all_digit_peaks:
        print('No digit peaks detected in numeric region')
    else:
        # Simple left-to-right assembly: sort detections by x and concatenate
        # the digit values in that order. We rely on `min_distance` used when
        # detecting peaks to ensure individual digits are reported separately.
        all_digit_peaks.sort(key=lambda t: t[0])
        assembled_digits = [str(int(digit)) for (x, digit, score) in all_digit_peaks]
        assembled_number = ''.join(assembled_digits)
        print('Assembled number (left-to-right):', assembled_number)


if __name__ == "__main__":
    # Example entry point: run the eco-summary demonstration
    summarize_eco()
