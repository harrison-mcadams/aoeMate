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
import concurrent.futures as futures

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

    # The earlier single-resource (gold) detector was removed in favor of the
    # generalized resource loop below that handles gold/food/stone/wood and
    # runs detection in parallel. This avoids duplicate work and centralizes
    # the logic.

    # ...now run the loop over all resources (gold, food, stone, wood)
    resources = [
        ('gold', 'gold_icon.png'),
        ('food', 'food_icon.png'),
        ('stone', 'stone_icon.png'),
        ('wood', 'wood_icon.png'),
    ]

    # Prepare a results dict to collect assembled values per resource
    results = {}

    # Base region bbox (screen coordinates) for later absolute conversions
    eco_summary_bbox = get_ss.get_bbox('eco_summary')

    # Tuning values used to expand from icon center to final capture boxes
    fudge_factor = 10  # small padding in pixels around boxes

    # ===== Performance improvements =====
    # 1) Load and cache all resource and digit kernels once to avoid repeated
    #    file I/O inside the loop.
    # 2) Run per-digit matching in parallel using a ThreadPoolExecutor. OpenCV's
    #    matchTemplate runs in C and benefits from parallelism across cores.
    resource_kernels = {}
    for _, icon_fname in resources:
        try:
            resource_kernels[icon_fname] = Image.open(kernelPath + icon_fname)
        except Exception:
            resource_kernels[icon_fname] = None

    digit_kernels = {}
    for digit in range(10):
        try:
            digit_kernels[digit] = Image.open(kernelPath + f'{digit}.png')
        except Exception:
            digit_kernels[digit] = None

    # Number of worker threads: cap to avoid oversubscription
    max_workers = max(2, min(8, (os.cpu_count() or 4)))

    # Precompute digit kernels as grayscale arrays (cache) to avoid repeated conversions
    digit_kernels_gray = {d: (np.array(k.convert('L'), dtype=np.float32) if k is not None else None) for d, k in digit_kernels.items()}

    # Shared executor for digit-matching tasks
    ex_digits = futures.ThreadPoolExecutor(max_workers=max_workers)

    # Executor for resource-level parallelism (run each resource concurrently)
    ex_resources = futures.ThreadPoolExecutor(max_workers=min(len(resources), max_workers))

    def process_resource(res_name: str, icon_fname: str):
        """Process a single resource: find icon, crop count area, detect digits.

        Returns the assembled number (string) or None on failure.
        """
        # Load kernel for this resource
        kernel = resource_kernels.get(icon_fname)
        if kernel is None:
            return None

        # Match the resource icon within the eco_summary screenshot
        res_conv = analyze_ss.convolve_ssXkernel(screenshot, kernel, out_path=out_path)
        found, peaks = analyze_ss.is_target_in_ss(res_conv, kernel, out_path=out_path, return_peaks=True)
        if not found or not peaks:
            return None

        # Use the first detected peak as the canonical icon location (relative to screenshot)
        peak_x, peak_y, peak_score = peaks[0]
        top = eco_summary_bbox['top'] + int(peak_y)
        left = eco_summary_bbox['left'] + int(peak_x)
        icon_w, icon_h = kernel.size

        # Build a bbox to the right of the icon where the numeric count usually appears
        count_width = 100  # heuristic; adjust if necessary
        count_bbox = {
            'top': int(top - fudge_factor),
            'left': int(left + icon_w + 2),
            'width': int(count_width),
            'height': int(icon_h + fudge_factor * 2)
        }

        # Capture the region containing the numeric count
        try:
            count_img = get_ss.capture_gfn_screen_region(count_bbox, out_path=out_path)
        except Exception:
            return None

        # Precompute grayscale array for the count crop
        count_img_gray = np.array(count_img.convert('L'), dtype=np.float32)

        # Submit per-digit detection tasks to the shared digit executor
        per_digit_min_distance = 5
        futures_digits = []
        def detect_digit_worker(dd: int):
            dk = digit_kernels_gray.get(dd)
            if dk is None:
                return []
            conv_local = analyze_ss.match_template_arrays(count_img_gray, dk, out_path=out_path)
            found_d, peaks_d = analyze_ss.is_target_in_ss(conv_local, None, out_path=out_path, return_peaks=True, min_distance=per_digit_min_distance)
            if not found_d:
                return []
            return [(int(px), int(dd), float(pscore)) for (px, py, pscore) in peaks_d]

        # Submit per-digit jobs to the shared digit executor and collect results
        for d in range(10):
            if digit_kernels_gray.get(d) is None:
                continue
            futures_digits.append(ex_digits.submit(detect_digit_worker, d))

        all_digit_peaks = []
        for f in futures.as_completed(futures_digits):
            try:
                res_list = f.result()
                if res_list:
                    all_digit_peaks.extend(res_list)
            except Exception:
                # ignore worker failures for robustness
                continue

        if not all_digit_peaks:
            return None
        # sort left-to-right and assemble
        all_digit_peaks.sort(key=lambda t: t[0])
        assembled = ''.join(str(int(d)) for (_, d, _) in all_digit_peaks)
        return assembled

    # Submit resource tasks and collect results as they complete. Use a try/finally
    # so that executors are always shut down even if submission or processing fails.
    # Attempt to submit resource tasks in parallel; if submission fails, fall
    # back to a sequential loop so callers still get a results dict.
    future_to_res = {}
    try:
        print(f'Submitting {len(resources)} resource tasks to executor...')
        future_to_res = {ex_resources.submit(process_resource, name, fname): (name, fname) for name, fname in resources}
        print('Submission complete, waiting for results...')
        for fut in futures.as_completed(future_to_res):
            name, _ = future_to_res[fut]
            try:
                val = fut.result()
                if val is None:
                    print(f"{name.title()} icon or digits not found")
                    results[name] = None
                else:
                    print(f"{name.title()}:", val)
                    results[name] = val
            except Exception as e:
                print(f"Error processing resource {name}:", e)
                results[name] = None
    except Exception as e:
        # If the executor itself failed for some reason, run resources sequentially
        print('Parallel submission failed, falling back to sequential processing:', e)
        for name, fname in resources:
            try:
                val = process_resource(name, fname)
                if val is None:
                    print(f"{name.title()} icon or digits not found")
                    results[name] = None
                else:
                    print(f"{name.title()}:", val)
                    results[name] = val
            except Exception as e2:
                print(f'Error processing resource {name} sequentially:', e2)
                results[name] = None
    finally:
        # Ensure executors are always shut down
        try:
            ex_digits.shutdown(wait=True)
        except Exception:
            pass
        try:
            ex_resources.shutdown(wait=True)
        except Exception:
            pass

    return results

if __name__ == "__main__":
    results = summarize_eco()
    # Print a concise summary for quick visibility when run as a script
    print('\nResource summary:')
    for k, v in (results or {}).items():
        print(f'  {k}: {v}')
