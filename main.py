import os
import math
import logging
from datetime import datetime
from pathlib import Path

import get_ss
import analyze_ss
from PIL import Image
import cv2
import numpy as np
import concurrent.futures as futures
import threading
import atexit

# Configure logging
log_path = Path(__file__).resolve().parent / 'aoemate.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(str(log_path), mode='a'),
        logging.StreamHandler()
    ]
)

_LIVE_ANIMATION = None

# Module-level cache for kernels and executors to avoid repeated I/O and thread creation
_KERNEL_PATH = os.environ.get('AOE_KERNEL_PATH', '/Users/harrisonmcadams/Desktop/')
_RESOURCE_KERNELS = {}
_RESOURCE_KERNELS_GRAY = {}
_DIGIT_KERNELS_GRAY = {}
_KERNELS_LOADED = False
_KERNELS_LOCK = threading.Lock()
_MAX_WORKERS = max(2, min(8, (os.cpu_count() or 4)))
_EX_DIGITS = None
_EX_RESOURCES = None
_CACHED_ANCHORS = None  # {name: (x, y)} relative to eco_summary


def _parse_number_from_region(image, digit_kernels, out_path=None, name="debug"):
    """Parse a number from an image using thresholding and contour analysis."""
    # Threshold to isolate white text
    thresh = analyze_ss.threshold_image(image, threshold=140)
    
    if out_path:
        try:
            os.makedirs(out_path, exist_ok=True)
            Image.fromarray(thresh).save(os.path.join(out_path, f'{name}_thresh.png'))
        except Exception:
            pass
    
    # Find potential digit contours
    bboxes = analyze_ss.find_digit_contours(thresh, min_h=8, max_h=30)
    
    if out_path:
        # Draw contours on debug image
        try:
            debug_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            for x, y, w, h in bboxes:
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            Image.fromarray(debug_img).save(os.path.join(out_path, f'{name}_contours.png'))
        except Exception:
            pass
    
    if not bboxes:
        return None
        
    # Match each bbox to a digit
    digits = []
    for x, y, w, h in bboxes:
        # Extract digit ROI
        roi = thresh[y:y+h, x:x+w]
        
        best_digit = -1
        best_score = -1.0
        
        # Simple template matching against resized templates
        # We need to resize the ROI to match the template size or vice versa.
        # Since templates are fixed size, let's resize ROI to a standard height (e.g. 14px)
        # and aspect ratio, then match?
        # OR: Just run matchTemplate of the fixed templates against the ROI?
        # The ROI might be tight, so matchTemplate might fail if template is larger.
        # Better approach: Resize ROI to match template height, then correlation.
        
        # Actually, let's stick to the plan: "Match the candidate against 0-9 digit templates"
        # Since we have the templates loaded in `digit_kernels`, let's use them.
        # But `digit_kernels` are full images. `_DIGIT_KERNELS_GRAY` are numpy arrays.
        
        # Let's try a simpler approach for now:
        # Resize ROI to a fixed size (e.g. 10x14) and match against resized templates?
        # Or just use the existing `match_template_arrays` but on the small ROI?
        
        # Given the previous code used `match_template_arrays` on the whole image,
        # let's try to match the template *inside* the ROI (padded).
        
        roi_padded = np.pad(roi, ((2, 2), (2, 2)), mode='constant')
        
        for d, k_gray in digit_kernels.items():
            if k_gray is None:
                continue
            
            # If template is larger than ROI, we can't match inside.
            # But the digits should be roughly the same size.
            
            # Let's try resizing the template to the ROI height
            # (assuming font size varies slightly but aspect ratio is const)
            
            # For simplicity in this first pass, let's just use the standard template matching
            # on the *original* count image, but restricted to the bbox?
            # No, that defeats the purpose of finding contours.
            
        # Pad the ROI to allow template matching without resizing distortion
        # Templates likely have some padding or are of a specific size.
        # We want to find the template *inside* the padded ROI (or centered on it).
        
        # Pad with 10 pixels of black (0)
        roi_padded = np.pad(roi, ((10, 10), (10, 10)), mode='constant', constant_values=0)
        
        # Dilate the padded ROI slightly to thicken strokes (helps if template is thicker)
        # kernel = np.ones((2, 2), np.uint8)
        # roi_padded = cv2.dilate(roi_padded, kernel, iterations=1)
        
        # Try erosion?
        kernel = np.ones((2, 2), np.uint8)
        roi_padded = cv2.erode(roi_padded, kernel, iterations=1)
        
        for d, k_gray in digit_kernels.items():
            if k_gray is None:
                continue
            
            try:
                # Check if template fits in padded ROI
                ph, pw = roi_padded.shape
                th, tw = k_gray.shape
                
                if th > ph or tw > pw:
                    # Template is huge? Should not happen for digits.
                    # If it does, we can't match it inside.
                    continue
                    
                # Normalized correlation
                res = cv2.matchTemplate(roi_padded.astype(np.float32), k_gray, cv2.TM_CCOEFF_NORMED)
                score = res.max()
                
                if score > best_score:
                    best_score = score
                    best_digit = d
            except Exception:
                continue
        
        
        if best_digit != -1 and best_score > 0.4: # Lower threshold slightly
            # Store (x, digit) to sort and group later
            digits.append((x, str(best_digit), w))
            
    if not digits:
        return []
        
    # Sort by X coordinate
    digits.sort(key=lambda t: t[0])
    
    # Group digits based on gap
    groups = []
    current_group = []
    last_x_end = -1
    
    # Gap threshold: if gap > 12px, it's a new number
    # (Typical digit width is ~8-10px, space is usually wider than a digit width)
    GAP_THRESHOLD = 12 
    
    for x, d, w in digits:
        if last_x_end != -1 and (x - last_x_end) > GAP_THRESHOLD:
            # Start new group
            if current_group:
                groups.append("".join(current_group))
            current_group = [d]
        else:
            current_group.append(d)
        last_x_end = x + w
        
    if current_group:
        groups.append("".join(current_group))
        
    return groups


def _find_anchors(ss_gray, resource_kernels):
    """Find resource icons in the screenshot with conflict resolution."""
    candidates = []
    
    # Include silver and alternate templates in the search
    resources = [
        ('food', 'food_icon.png'),
        ('wood', 'wood_icon.png'),
        ('gold', 'gold_icon.png'),
        ('stone', 'stone_icon.png'),
        ('silver', 'silver_icon_macedonia.png'), # Silver only exists in this variant for now
        
        # Alternate templates for "Macedonia" / low-quality screenshots
        ('food', 'food_icon_macedonia.png'),
        ('wood', 'wood_icon_macedonia.png'),
        ('stone', 'stone_icon_macedonia.png'),
    ]
    
    for name, fname in resources:
        k_gray = resource_kernels.get(fname)
        if k_gray is None:
            continue
            
        res_conv = analyze_ss.match_template_arrays(ss_gray, k_gray)
        found, peaks = analyze_ss.is_target_in_ss(res_conv, None, return_peaks=True, threshold=0.6)
        
        if found and peaks:
            for x, y, score in peaks:
                candidates.append({'name': name, 'score': score, 'x': int(x), 'y': int(y)})
                
    # Sort by score descending
    candidates.sort(key=lambda c: c['score'], reverse=True)
    
    anchors = {}
    occupied_positions = []
    
    min_dist = 10 # Minimum distance between distinct icons
    
    for c in candidates:
        # Check if this position is already occupied by a better match
        is_occupied = False
        for ox, oy in occupied_positions:
            dist = math.hypot(c['x'] - ox, c['y'] - oy)
            if dist < min_dist:
                is_occupied = True
                break
        
        if not is_occupied:
            # This is a new valid anchor
            # Check if we already have this resource (e.g. found twice?)
            # Usually we only expect one of each, but maybe not?
            # For now, let's assume one of each.
            if c['name'] not in anchors:
                anchors[c['name']] = (c['x'], c['y'])
                occupied_positions.append((c['x'], c['y']))
            
    return anchors


def _init_kernels_and_executors(resources):
    """Load resource and digit kernels once, and create shared executors.

    Thread-safe and idempotent. `resources` is a list of (name, filename) pairs
    used to size the resource executor properly.
    """
    global _KERNELS_LOADED, _RESOURCE_KERNELS, _RESOURCE_KERNELS_GRAY, _DIGIT_KERNELS_GRAY, _EX_DIGITS, _EX_RESOURCES
    if _KERNELS_LOADED:
        return
    with _KERNELS_LOCK:
        if _KERNELS_LOADED:
            return
        # load resource kernels and precompute grayscale arrays
        for _, icon_fname in resources:
            try:
                kimg = Image.open(os.path.join(_KERNEL_PATH, icon_fname))
                _RESOURCE_KERNELS[icon_fname] = kimg
                _RESOURCE_KERNELS_GRAY[icon_fname] = np.array(kimg.convert('L'), dtype=np.float32)
            except Exception:
                _RESOURCE_KERNELS[icon_fname] = None
                _RESOURCE_KERNELS_GRAY[icon_fname] = None
        # load digit kernels and precompute gray arrays
        for d in range(10):
            try:
                k = Image.open(os.path.join(_KERNEL_PATH, f'{d}.png'))
                _DIGIT_KERNELS_GRAY[d] = np.array(k.convert('L'), dtype=np.float32)
            except Exception:
                _DIGIT_KERNELS_GRAY[d] = None
        # create executors
        try:
            _EX_DIGITS = futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS)
        except Exception:
            _EX_DIGITS = None
        try:
            _EX_RESOURCES = futures.ThreadPoolExecutor(max_workers=min(len(resources), _MAX_WORKERS))
        except Exception:
            _EX_RESOURCES = None
        _KERNELS_LOADED = True


def _shutdown_executors():
    global _EX_DIGITS, _EX_RESOURCES
    try:
        if _EX_DIGITS is not None:
            _EX_DIGITS.shutdown(wait=True)
    except Exception:
        pass
    try:
        if _EX_RESOURCES is not None:
            _EX_RESOURCES.shutdown(wait=True)
    except Exception:
        pass

# Ensure executors shut down on process exit
atexit.register(_shutdown_executors)


def are_vills_producing():
    """Small OpenCV status panel: captures eco_summary and runs villager kernel.

    This keeps the previous small visual status behavior.
    """
    try:
        vill_kernel = Image.open("/Users/harrisonmcadams/Desktop/villager_icon.png")
    except Exception:
        logging.exception('Could not load villager kernel')
        vill_kernel = None

    cv2.namedWindow('AOEMate', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('AOEMate', 200, 100)

    poll_ms = int(os.environ.get('AOEMATE_POLL_MS', '100'))

    try:
        import pyautogui
        sw, sh = pyautogui.size()
        screen_w, screen_h = int(sw), int(sh)
    except Exception:
        screen_w = int(os.environ.get('AOEMATE_SCREEN_W', '1280'))
        screen_h = int(os.environ.get('AOEMATE_SCREEN_H', '800'))

    win_w = int(int(os.environ.get('AOEMATE_WIN_W', str(int(screen_w * 0.5)))))
    win_h = int(int(os.environ.get('AOEMATE_WIN_H', str(int(screen_h * 0.5)))))
    win_x = int(os.environ.get('AOEMATE_WIN_X', str((screen_w - win_w) // 2)))
    win_y = int(os.environ.get('AOEMATE_WIN_Y', str((screen_h - win_h) // 2)))

    try:
        while True:
            eco_summary = get_ss.get_bbox('eco_summary')
            screenshot = get_ss.capture_gfn_screen_region(eco_summary)
            out_path = None
            try:
                if vill_kernel is None:
                    binary = False
                else:
                    conv = analyze_ss.convolve_ssXkernel(screenshot, vill_kernel, out_path=out_path)
                    binary = analyze_ss.is_target_in_ss(conv, vill_kernel, out_path=out_path)
            except Exception:
                logging.exception('Analysis error')
                binary = False

            color = (0, 255, 0) if binary else (0, 0, 255)
            status = np.full((win_h, win_w, 3), color, dtype=np.uint8)

            label = 'Producing' if binary else 'Not producing'
            font = cv2.FONT_HERSHEY_SIMPLEX
            base_scale = max(1.0, min(win_w, win_h) / 400.0)
            thickness = max(2, int(base_scale))
            (tw, th), _ = cv2.getTextSize(label, font, base_scale * 2.0, thickness)
            tx = max(10, (win_w - tw) // 2)
            ty = max(30, (win_h + th) // 2)
            cv2.putText(status, label, (tx, ty), font, base_scale * 2.0, (255, 255, 255), thickness, cv2.LINE_AA)

            try:
                cv2.imshow('AOEMate', status)
                cv2.resizeWindow('AOEMate', win_w, win_h)
                cv2.moveWindow('AOEMate', win_x, win_y)
            except Exception:
                blank = 255 * np.ones((200, 300, 3), dtype='uint8')
                cv2.imshow('AOEMate', blank)

            key = cv2.waitKey(poll_ms) & 0xFF
            if key == ord('q') or key == 27:
                logging.info('Quit key pressed - exiting')
                break

    except KeyboardInterrupt:
        logging.info('Interrupted by user')
    finally:
        cv2.destroyAllWindows()


def summarize_eco(screenshot=None, out_path=None):
    """Capture the eco_summary region and extract resource counts.

    Returns a dict: {'gold': '100', 'food': '20', 'stone': '0', 'wood': '200'} or None values.
    """
    global _CACHED_ANCHORS
    
    # logging.info('summarize_eco: entering')
    eco_summary = get_ss.get_bbox('eco_summary')
    # logging.info('summarize_eco: eco_summary bbox=%s', eco_summary)
    
    if screenshot is None:
        screenshot = get_ss.capture_gfn_screen_region(eco_summary)

    # out_path = None  <-- Removed this line as it's now an argument
    # Allow kernel path to be overridden with env var AOE_KERNEL_PATH
    kernelPath = _KERNEL_PATH

    resources = [
        ('food', 'food_icon.png'),
        ('wood', 'wood_icon.png'),
        ('gold', 'gold_icon.png'),
        ('stone', 'stone_icon.png'),
        ('silver', 'silver_icon_macedonia.png'),
        ('food', 'food_icon_macedonia.png'),
        ('wood', 'wood_icon_macedonia.png'),
        ('stone', 'stone_icon_macedonia.png'),
    ]

    results = {}
    fudge_factor = 10
    
    # Initialize module-level caches
    _init_kernels_and_executors(resources)
    resource_kernels_gray = _RESOURCE_KERNELS_GRAY
    digit_kernels_gray = _DIGIT_KERNELS_GRAY
    
    ss_gray = np.array(screenshot.convert('L'), dtype=np.float32)
    
    # 1. Dynamic Anchoring
    if _CACHED_ANCHORS is None:
        logging.info("Finding anchors...")
        _CACHED_ANCHORS = _find_anchors(ss_gray, resource_kernels_gray)
        logging.info("Found anchors: %s", _CACHED_ANCHORS)
        
    # If we still don't have anchors (e.g. black screen), we can't do anything
    if not _CACHED_ANCHORS:
        return {r[0]: None for r in resources}
        
    # 2. Extract and Parse Counts
    for name, fname in resources:
        anchor = _CACHED_ANCHORS.get(name)
        if not anchor:
            results[name] = None
            continue
            
        ax, ay = anchor
        # Get icon size
        k_gray = resource_kernels_gray.get(fname)
        if k_gray is None:
            results[name] = None
            continue
            
        h, w = k_gray.shape
        
        # Define count region (right of icon)
        # Adjust these offsets based on the screenshot analysis if needed
        # Previous code used: top = ay, left = ax + w + 2
        count_left = ax + w - 3  # Shift left slightly to catch leading digits
        count_top = ay - fudge_factor
        count_w = 100
        count_h = h + fudge_factor * 2
        
        # Ensure bounds
        sh, sw = ss_gray.shape
        count_left = max(0, min(sw - 1, count_left))
        count_top = max(0, min(sh - 1, count_top))
        count_right = min(sw, count_left + count_w)
        count_bottom = min(sh, count_top + count_h)
        
        if count_right <= count_left or count_bottom <= count_top:
            results[name] = None
            continue
            
        # Extract region from the ALREADY captured screenshot (converted to PIL Image for helper)
        # We have ss_gray (numpy), but helper takes PIL Image for thresholding (or we can adapt helper)
        # Helper `_parse_number_from_region` calls `analyze_ss.threshold_image` which takes PIL Image.
        # Let's crop from the original PIL screenshot.
        
        try:
            count_img = screenshot.crop((count_left, count_top, count_right, count_bottom))
            val_groups = _parse_number_from_region(count_img, digit_kernels_gray, out_path=out_path, name=name)
            
            if val_groups:
                results[name] = val_groups[0]
                if name == 'silver':
                    results[f'{name}_vills'] = None
                elif len(val_groups) > 1:
                    results[f'{name}_vills'] = val_groups[1]
                else:
                    results[f'{name}_vills'] = None
            else:
                results[name] = None
                results[f'{name}_vills'] = None
                
        except Exception:
            logging.exception("Error parsing %s", name)
            results[name] = None
            results[f'{name}_vills'] = None

    return results


def live_monitor_resources(poll_sec: float = 1.0, max_points: int = 300):
    """OpenCV-only live monitor: draws stacked resource plots and updates them.

    Press 'q' or ESC in the OpenCV window to quit.
    """
    logging.info('live_monitor_resources (OpenCV) start poll_sec=%s max_points=%s', poll_sec, max_points)
    global _LIVE_ANIMATION
    if _LIVE_ANIMATION is not None:
        logging.info('live_monitor_resources: animation already running')
        return

    resource_names = ['food', 'wood', 'gold', 'stone']
    times = []
    data = {r: [] for r in resource_names}

    sw = int(os.environ.get('AOEMATE_SCREEN_W', '1280'))
    sh = int(os.environ.get('AOEMATE_SCREEN_H', '800'))
    win_w = int(os.environ.get('AOEMATE_WIN_W', str(min(1000, int(sw * 0.6)))))
    win_h = int(os.environ.get('AOEMATE_WIN_H', str(min(900, int(sh * 0.6)))))

    cv_win_name = 'AOEMatePlot'
    try:
        cv2.namedWindow(cv_win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cv_win_name, win_w, win_h)
    except Exception:
        pass

    # Allow widening of the kept history window by a multiplicative factor.
    # Default is 1.2 (20% wider than `max_points`). Can be overridden via
    # the environment variable AOE_TIME_WINDOW_SCALE.
    try:
        time_window_scale = float(os.environ.get('AOE_TIME_WINDOW_SCALE', '1.0'))
        if time_window_scale <= 0:
            time_window_scale = 1.0
    except Exception:
        time_window_scale = 1.0

    # Rate smoothing time constant (seconds). Larger values => smoother, longer timescale.
    try:
        # Default to 90 seconds for a longer averaging window; user can override
        # with the AOE_RATE_TAU_SEC environment variable (in seconds).
        RATE_TAU = float(os.environ.get('AOE_RATE_TAU_SEC', '20.0'))
        if RATE_TAU <= 0:
            RATE_TAU = 20.0
    except Exception:
        RATE_TAU = 20.0

    start_time = datetime.now()

    def _append_values(results):
        now = datetime.now()
        times.append(now)
        for r in resource_names:
            v = None
            if results and r in results:
                v = results[r]
            if v is None:
                data[r].append(math.nan)
            else:
                sval = str(v).replace(',', '').strip()
                try:
                    data[r].append(int(sval))
                except Exception:
                    try:
                        data[r].append(float(sval))
                    except Exception:
                        data[r].append(math.nan)
        # Trim history to the scaled window (increase by ~20% by default)
        limit = max(2, int(max_points * time_window_scale))
        if len(times) > limit:
            del times[:-limit]
            for r in resource_names:
                del data[r][:-limit]

    def render_frame():
        # Increase whitespace & margins for a more spacious look
        canvas = 255 * np.ones((win_h, win_w, 3), dtype=np.uint8)
        pad = 12  # more vertical padding between plots
        left_margin = 80
        right_margin = 40
        plot_h = (win_h - (len(resource_names) + 1) * pad) // len(resource_names)

        # First pass: compute plotting primitives and smoothed rates for each resource
        entries = []  # list of dicts with drawing info per resource
        global_vals = []
        global_rates = []
        for i, r in enumerate(resource_names):
            vals = data.get(r, [])
            n = len(vals)
            xs = np.linspace(left_margin, win_w - right_margin, n) if n > 0 else np.array([])

            # compute simple per-sample smoothed rate (fallback behavior when possible)
            rates = []
            smoothed_rates = []
            t_secs = None
            if n >= 2 and len(times) >= 2:
                try:
                    t_secs = [(t - times[0]).total_seconds() for t in times[-n:]] if len(times) >= n else [(times[i] - times[0]).total_seconds() for i in range(n)]
                except Exception:
                    t_secs = [float(i) for i in range(n)]
                if len(t_secs) != n:
                    t_secs = [float(i) for i in range(n)]
                # forward-fill NaNs for rate computation
                raw_vals = []
                for ii, v in enumerate(vals):
                    if ii == 0:
                        raw_vals.append(0.0 if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v))
                    else:
                        if v is None or (isinstance(v, float) and math.isnan(v)):
                            raw_vals.append(raw_vals[-1])
                        else:
                            raw_vals.append(float(v))

                # compute forward diffs as simple rate estimate
                for ii in range(n):
                    if ii == 0:
                        dt = t_secs[1] - t_secs[0] if n > 1 else 1.0
                        dv = raw_vals[1] - raw_vals[0] if n > 1 else 0.0
                    else:
                        dt = t_secs[ii] - t_secs[ii - 1]
                        if dt == 0:
                            dt = 1.0
                        dv = raw_vals[ii] - raw_vals[ii - 1]
                    if dv < 0:
                        dv = 0.0
                    rates.append((dv / max(1e-6, dt)) * 60.0)
                # quick EMA smoothing
                for ii, rrr in enumerate(rates):
                    if ii == 0:
                        smoothed_rates.append(rrr)
                    else:
                        try:
                            dt = t_secs[ii] - t_secs[ii - 1]
                            if dt <= 0:
                                dt = 1.0
                        except Exception:
                            dt = 1.0
                        alpha = 1.0 - math.exp(-dt / RATE_TAU)
                        smoothed_rates.append(alpha * rrr + (1.0 - alpha) * smoothed_rates[-1])

            # compute vmin/vmax for left axis
            valid = [v for v in vals if not math.isnan(v)]
            if valid:
                vmin = min(valid)
                vmax = max(valid)
            else:
                vmin, vmax = 0.0, 1.0

            global_vals.extend([v for v in valid])
            global_rates.extend([r for r in smoothed_rates if not math.isnan(r)])

            entries.append({
                'name': r,
                'xs': xs,
                'vals': vals,
                'n': n,
                'vmin': vmin,
                'vmax': vmax,
                'smoothed_rates': smoothed_rates,
            })

        # Lock left y-axis across subplots
        if global_vals:
            G_vmin = min(global_vals)
            G_vmax = max(global_vals)
            if G_vmin == G_vmax:
                G_vmax = G_vmin + 1.0
        else:
            G_vmin, G_vmax = 0.0, 1.0

        # Lock right y-axis (rates) across subplots
        if global_rates:
            G_rmin = min(global_rates)
            G_rmax = max(global_rates)
            if G_rmin == G_rmax:
                G_rmax = G_rmin + 1.0
        else:
            G_rmin, G_rmax = 0.0, 1.0

        # Second pass: draw each subplot using locked axes and a bit more spacing
        for i, ent in enumerate(entries):
            top = pad + i * (plot_h + pad)
            bottom = top + plot_h
            left = left_margin
            right = win_w - right_margin
            # background box
            cv2.rectangle(canvas, (left - 10, top), (right, bottom), (240, 240, 240), -1)

            xs = ent['xs']
            vals = ent['vals']
            n = ent['n']

            # Plot resource totals scaled to global vmin/vmax
            pts = []
            vrange = G_vmax - G_vmin if G_vmax != G_vmin else 1.0
            for k in range(n):
                val = vals[k]
                if math.isnan(val):
                    y = bottom - 4
                else:
                    y = int(top + (plot_h - 20) * (1.0 - (float(val) - G_vmin) / vrange)) + 10
                x = int(xs[k])
                pts.append((x, y))
            if pts:
                pts_arr = np.array(pts, dtype=np.int32)
                cv2.polylines(canvas, [pts_arr], False, (0, 120, 255), 2, lineType=cv2.LINE_AA)
                for (x, y) in pts:
                    cv2.circle(canvas, (x, y), 3, (0, 80, 200), -1)
            # left axis labels
            cv2.putText(canvas, f'{int(G_vmax)}', (6, top + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(canvas, f'{int(G_vmin)}', (6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            # Plot smoothed rate on right axis using global rate limits
            srates = ent['smoothed_rates']
            if srates:
                rrange = G_rmax - G_rmin if G_rmax != G_rmin else 1.0
                pts_rate = []
                for k, srate in enumerate(srates):
                    if math.isnan(srate):
                        y_r = bottom - 4
                    else:
                        y_r = int(top + (plot_h - 20) * (1.0 - (srate - G_rmin) / rrange)) + 10
                    x = int(xs[k])
                    pts_rate.append((x, y_r))
                if pts_rate:
                    pts_rate_arr = np.array(pts_rate, dtype=np.int32)
                    try:
                        cv2.polylines(canvas, [pts_rate_arr], False, (34, 139, 34), 1, lineType=cv2.LINE_AA)
                    except Exception:
                        pass
                    for (xr, yr) in pts_rate:
                        cv2.circle(canvas, (xr, yr), 2, (34, 139, 34), -1)
                # right axis labels
                cv2.putText(canvas, f'{int(G_rmax)}', (win_w - right_margin - 2, top + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (34, 139, 34), 1, cv2.LINE_AA)
                cv2.putText(canvas, f'{int(G_rmin)}', (win_w - right_margin - 2, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (34, 139, 34), 1, cv2.LINE_AA)

            # Title (lowered a bit)
            title_y = top + 14
            cv2.putText(canvas, ent['name'].title(), (left_margin - 10, title_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2, cv2.LINE_AA)

            # Current value + inline rate text
            cur = vals[-1] if vals else math.nan
            cur_text = str(int(cur)) if (not math.isnan(cur)) else 'NaN'
            latest_rate_text = ''
            try:
                if ent['smoothed_rates']:
                    latest_s = ent['smoothed_rates'][-1]
                    if latest_s is not None and (not math.isnan(latest_s)):
                        latest_rate_text = f' ({latest_s:.1f}/min)'
            except Exception:
                latest_rate_text = ''
            display_text = f'{cur_text}{latest_rate_text}'
            # Right-aligned current value + rate on single line; shrink to fit if needed
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.7
            thickness = 2
            max_text_width = win_w - right_margin - left_margin - 8
            (tw, th), _ = cv2.getTextSize(display_text, font, scale, thickness)
            min_scale = 0.35
            while tw > max_text_width and scale > min_scale:
                scale -= 0.05
                (tw, th), _ = cv2.getTextSize(display_text, font, scale, thickness)
            tx = max(left_margin, win_w - right_margin - tw - 6)
            ty = top + 18
            cv2.putText(canvas, display_text, (tx, ty), font, scale, (50, 150, 50), thickness, cv2.LINE_AA)

        # elapsed time footer
        elapsed = (datetime.now() - start_time).total_seconds()
        cv2.putText(canvas, f'Elapsed: {int(elapsed)}s', (10, win_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA)
        return canvas

    try:
        _LIVE_ANIMATION = True
        frame = 0
        while _LIVE_ANIMATION:
            try:
                results = {}
                try:
                    results = summarize_eco()
                except Exception:
                    logging.exception('summarize_eco() failed')
                _append_values(results)
                canvas = render_frame()
                try:
                    cv2.imshow(cv_win_name, canvas)
                except Exception:
                    pass
                k = cv2.waitKey(int(poll_sec * 1000)) & 0xFF
                if k == ord('q') or k == 27:
                    logging.info('Quit key pressed - exiting live monitor')
                    break
                frame += 1
            except Exception:
                logging.exception('Error in OpenCV live loop; exiting')
                break
        logging.info('OpenCV live loop ended')
    finally:
        try:
            cv2.destroyWindow(cv_win_name)
        except Exception:
            pass
        _LIVE_ANIMATION = None


if __name__ == '__main__':
    try:
        poll = float(os.environ.get('AOE_POLL_SEC', '1.0'))
    except Exception:
        poll = 1.0
    try:
        max_pts = int(os.environ.get('AOE_MAX_POINTS', '300'))
    except Exception:
        max_pts = 300
    # Run the OpenCV live monitor. Use env vars to adjust behavior if desired.
    live_monitor_resources(poll_sec=poll, max_points=max_pts)
