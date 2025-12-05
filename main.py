import os
import math
import logging
from datetime import datetime
from pathlib import Path

import get_ss
import analyze_ss
from PIL import Image, ImageFont, ImageDraw
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
        # kernel = np.ones((2, 2), np.uint8)
        # roi_padded = cv2.erode(roi_padded, kernel, iterations=1)
        
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


def check_villager_production(screenshot, vill_kernel=None):
    """Check if villagers are being produced in the given screenshot."""
    if vill_kernel is None:
        try:
            vill_kernel = Image.open("/Users/harrisonmcadams/Desktop/villager_icon.png")
        except Exception:
            logging.exception('Could not load villager kernel')
            return False
            
    out_path = None
    try:
        # Generalized detection: crop to top-left ROI (60x60) and lower threshold
        roi = screenshot.crop((0, 0, 60, 60))
        conv = analyze_ss.convolve_ssXkernel(roi, vill_kernel, out_path=out_path)
        binary = analyze_ss.is_target_in_ss(conv, vill_kernel, out_path=out_path, threshold=0.65)
        return binary
    except Exception:
        logging.exception('Analysis error')
        return False


def are_vills_producing():
    """Small OpenCV status panel: captures eco_summary and runs villager kernel."""
    try:
        vill_kernel = Image.open("/Users/harrisonmcadams/Desktop/villager_icon.png")
    except Exception:
        logging.exception('Could not load villager kernel')
        vill_kernel = None

    cv2.namedWindow('AOEMate', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('AOEMate', 200, 100)

    poll_ms = int(os.environ.get('AOEMATE_POLL_MS', '100'))
    
    # ... (window positioning logic omitted for brevity, it was in original but not critical to replicate fully if just refactoring logic)
    # Actually, let's keep the original function body mostly but use the helper
    
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
            
            binary = check_villager_production(screenshot, vill_kernel)

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
        
        # Define separate regions for Resource Count and Villager Count
        # We use the "villager separator" icon (person standing) as the delimiter.
        # It was found at offset ~90px from the resource icon right edge.
        
        # 1. Search for Separator
        # Look in a strip where we expect it: [ax+w+70, ax+w+110]
        # Widen search to ensure we catch it
        sep_search_left = ax + w + 60
        sep_search_top = ay - 4
        sep_search_w = 60
        sep_search_h = h + 8
        
        # Ensure bounds
        sh, sw = ss_gray.shape
        sep_search_left = max(0, min(sw - 1, sep_search_left))
        sep_search_top = max(0, min(sh - 1, sep_search_top))
        sep_search_right = min(sw, sep_search_left + sep_search_w)
        sep_search_bottom = min(sh, sep_search_top + sep_search_h)
        
        separator_found = False
        sep_x_rel = 0 # Relative to search strip
        sep_w_found = 0
        
        if sep_search_right > sep_search_left and sep_search_bottom > sep_search_top:
            try:
                sep_strip = ss_gray[sep_search_top:sep_search_bottom, sep_search_left:sep_search_right]
                # Match template
                # We need to load it first. It's not in the main dict yet.
                # Ideally we load it in _init_kernels. For now, let's lazy load or assume it's loaded.
                # Let's add it to _RESOURCE_KERNELS_GRAY if not present.
                if 'villager_separator.png' not in resource_kernels_gray:
                     try:
                        kimg = Image.open(os.path.join(_KERNEL_PATH, 'villager_separator.png'))
                        resource_kernels_gray['villager_separator.png'] = np.array(kimg.convert('L'), dtype=np.float32)
                     except:
                        pass
                
                k_sep = resource_kernels_gray.get('villager_separator.png')
                if k_sep is not None:
                    res_conv = analyze_ss.match_template_arrays(sep_strip, k_sep)
                    found, peaks = analyze_ss.is_target_in_ss(res_conv, None, return_peaks=True, threshold=0.6)
                    if found and peaks:
                        # Take the best match
                        peaks.sort(key=lambda p: p[2], reverse=True)
                        px, py, _ = peaks[0]
                        separator_found = True
                        sep_x_rel = int(px)
                        sep_w_found = k_sep.shape[1]
            except Exception:
                pass

        # 2. Define ROIs based on Separator or Fallback
        if separator_found:
            # Separator absolute X
            sep_abs_x = sep_search_left + sep_x_rel
            
            # Resource ROI: Ends before separator
            res_roi_left = ax + w - 5
            res_roi_right = sep_abs_x - 2 # Padding
            res_roi_top = ay - fudge_factor
            res_roi_bottom = ay + h + fudge_factor
            
            # Villager ROI: Starts after separator
            vill_roi_left = sep_abs_x + sep_w_found + 2 # Padding
            vill_roi_right = vill_roi_left + 50 # Assume max width
            vill_roi_top = ay - fudge_factor
            vill_roi_bottom = ay + h + fudge_factor
            
        else:
            # Fallback to hardcoded offsets (from previous step)
            res_roi_left = ax + w - 5 
            res_roi_top = ay - fudge_factor
            res_roi_w = 75 
            res_roi_h = h + fudge_factor * 2
            res_roi_right = res_roi_left + res_roi_w
            res_roi_bottom = res_roi_top + res_roi_h
            
            vill_roi_left = ax + w + 80 
            vill_roi_top = ay - fudge_factor
            vill_roi_w = 50 
            vill_roi_h = h + fudge_factor * 2
            vill_roi_right = vill_roi_left + vill_roi_w
            vill_roi_bottom = vill_roi_top + vill_roi_h
        
        # Ensure bounds for Resource ROI
        res_roi_left = max(0, min(sw - 1, res_roi_left))
        res_roi_top = max(0, min(sh - 1, res_roi_top))
        res_roi_right = min(sw, res_roi_right)
        res_roi_bottom = min(sh, res_roi_bottom)
        
        # Extract Resource Count
        if res_roi_right > res_roi_left and res_roi_bottom > res_roi_top:
            try:
                res_img = screenshot.crop((res_roi_left, res_roi_top, res_roi_right, res_roi_bottom))
                val_groups = _parse_number_from_region(res_img, digit_kernels_gray, out_path=out_path, name=f"{name}_res")
                results[name] = val_groups[0] if val_groups else None
            except Exception:
                results[name] = None
        else:
            results[name] = None

        # Extract Villager Count (Skip for Silver)
        if name == 'silver':
            results[f'{name}_vills'] = None
        else:
            # Ensure bounds for Villager ROI
            vill_roi_left = max(0, min(sw - 1, vill_roi_left))
            vill_roi_top = max(0, min(sh - 1, vill_roi_top))
            vill_roi_right = min(sw, vill_roi_right)
            vill_roi_bottom = min(sh, vill_roi_bottom)
            
            if vill_roi_right > vill_roi_left and vill_roi_bottom > vill_roi_top:
                try:
                    vill_img = screenshot.crop((vill_roi_left, vill_roi_top, vill_roi_right, vill_roi_bottom))
                    val_groups_v = _parse_number_from_region(vill_img, digit_kernels_gray, out_path=out_path, name=f"{name}_vill")
                    results[f'{name}_vills'] = val_groups_v[0] if val_groups_v else None
                except Exception:
                    results[f'{name}_vills'] = None
            else:
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

    resource_names = ['food', 'wood', 'gold', 'stone', 'silver']
    times = []
    data = {r: [] for r in resource_names}
    vill_data = {r: [] for r in resource_names}

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
    waiting_for_game = True
    
    # Pre-load fonts for waiting screen
    try:
        font_wait = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 40)
    except:
        font_wait = ImageFont.load_default()

    def _append_values(results):
        now = datetime.now()
        times.append(now)
        for r in resource_names:
            # Resource Value
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
            
            # Villager Value
            v_vill = None
            if results and f'{r}_vills' in results:
                v_vill = results[f'{r}_vills']
            if v_vill is None:
                vill_data[r].append(math.nan)
            else:
                sval_v = str(v_vill).replace(',', '').strip()
                try:
                    vill_data[r].append(int(sval_v))
                except Exception:
                    vill_data[r].append(math.nan)
        # Trim history to the scaled window (increase by ~20% by default)
        limit = max(2, int(max_points * time_window_scale))
        if len(times) > limit:
            times.pop(0)
            for r in resource_names:
                if len(data[r]) > len(times):
                    data[r].pop(0)
                if len(vill_data[r]) > len(times):
                    vill_data[r].pop(0)

    # Pre-load fonts
    try:
        font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
        font_path_bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        
        # Increased sizes for better readability
        font_axis = ImageFont.truetype(font_path, 14)
        font_title = ImageFont.truetype(font_path, 18) # Larger title
        font_val = ImageFont.truetype(font_path, 20)   # Larger value
        
        # Bold fonts for Total row
        font_title_bold = ImageFont.truetype(font_path_bold, 18)
        font_val_bold = ImageFont.truetype(font_path_bold, 20)
        
        # Fonts for Villager Stats
        font_vill_count = ImageFont.truetype(font_path, 16)
        font_vill_rate = ImageFont.truetype(font_path, 12)
        
        font_status_label = ImageFont.truetype(font_path, 18)
        font_status_val = ImageFont.truetype(font_path, 36) # Much larger status
        font_footer = ImageFont.truetype(font_path, 12)
    except Exception:
        # Fallback to default if Arial not found
        font_axis = ImageFont.load_default()
        font_title = ImageFont.load_default()
        font_val = ImageFont.load_default()
        font_title_bold = ImageFont.load_default()
        font_val_bold = ImageFont.load_default()
        font_vill_count = ImageFont.load_default()
        font_vill_rate = ImageFont.load_default()
        font_status_label = ImageFont.load_default()
        font_status_val = ImageFont.load_default()
        font_footer = ImageFont.load_default()

    def render_frame():
        # Increase whitespace & margins for a more spacious look
        # Split layout: Left 70% for plots, Right 30% for Villager Status
        
        # Create canvas in OpenCV (BGR) for drawing lines/rectangles
        canvas_cv = 255 * np.ones((win_h, win_w, 3), dtype=np.uint8)
        
        # Define areas
        split_x = int(win_w * 0.7)
        
        # --- Left Side: Resource Plots ---
        pad = 12
        left_margin = 80
        # Increased right margin to accommodate villager stats column
        right_margin = 140 # relative to split_x
        
        plot_w = split_x
        
        # Add "Total" to the list of things to plot locally
        plot_names = resource_names + ['Total']
        plot_h = (win_h - (len(plot_names) + 1) * pad) // len(plot_names)

        # First pass: compute plotting primitives and smoothed rates for each resource
        entries = []  # list of dicts with drawing info per resource
        global_vals = []
        global_rates = []
        
        # Calculate Total Data
        total_vals = []
        if len(times) > 0:
            n_points = len(times)
            for i in range(n_points):
                tot = 0.0
                valid_count = 0
                for r in resource_names:
                    vals = data.get(r, [])
                    if i < len(vals):
                        v = vals[i]
                        if v is not None and not (isinstance(v, float) and math.isnan(v)):
                            tot += float(v)
                            valid_count += 1
                if valid_count > 0:
                    total_vals.append(tot)
                else:
                    total_vals.append(math.nan)
        
        for i, r in enumerate(plot_names):
            if r == 'Total':
                vals = total_vals
                # For Total, we can sum villager counts if we want, or just leave it blank
                # Let's try to sum them for completeness
                current_vill_count = 0
                for rn in resource_names:
                    v_list = vill_data.get(rn, [])
                    if v_list:
                        last_v = v_list[-1]
                        if last_v is not None and not math.isnan(last_v):
                            current_vill_count += last_v
            else:
                vals = data.get(r, [])
                # Get current villager count
                v_list = vill_data.get(r, [])
                current_vill_count = v_list[-1] if v_list else 0
                if current_vill_count is None or (isinstance(current_vill_count, float) and math.isnan(current_vill_count)):
                    current_vill_count = 0
                
            n = len(vals)
            xs = np.linspace(left_margin, plot_w - right_margin, n) if n > 0 else np.array([])

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

            # Only contribute to global scale if NOT Total (Total will likely be much larger)
            if r != 'Total':
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
                'vill_count': current_vill_count
            })

        # Lock left y-axis across subplots (excluding Total)
        if global_vals:
            G_vmin = min(global_vals)
            G_vmax = max(global_vals)
            if G_vmin == G_vmax:
                G_vmax = G_vmin + 1.0
        else:
            G_vmin, G_vmax = 0.0, 1.0

        # Lock right y-axis (rates) across subplots (excluding Total)
        if global_rates:
            G_rmin = min(global_rates)
            G_rmax = max(global_rates)
            if G_rmin == G_rmax:
                G_rmax = G_rmin + 1.0
        else:
            G_rmin, G_rmax = 0.0, 1.0

        # Second pass: draw each subplot using locked axes and a bit more spacing
        # Draw shapes with OpenCV
        for i, ent in enumerate(entries):
            top = pad + i * (plot_h + pad)
            bottom = top + plot_h
            left = left_margin
            right = plot_w - right_margin
            # background box
            cv2.rectangle(canvas_cv, (left - 10, top), (right, bottom), (240, 240, 240), -1)

            xs = ent['xs']
            vals = ent['vals']
            n = ent['n']
            name = ent['name']
            
            # Determine scale for this plot
            if name == 'Total':
                # Use local scale for Total
                p_vmin = ent['vmin']
                p_vmax = ent['vmax']
                if p_vmin == p_vmax: p_vmax += 1.0
                
                srates = ent['smoothed_rates']
                valid_rates = [r for r in srates if not math.isnan(r)]
                if valid_rates:
                    p_rmin = min(valid_rates)
                    p_rmax = max(valid_rates)
                    if p_rmin == p_rmax: p_rmax += 1.0
                else:
                    p_rmin, p_rmax = 0.0, 1.0
            else:
                # Use global scale for others
                p_vmin, p_vmax = G_vmin, G_vmax
                p_rmin, p_rmax = G_rmin, G_rmax

            # Plot resource totals
            pts = []
            vrange = p_vmax - p_vmin if p_vmax != p_vmin else 1.0
            for k in range(n):
                val = vals[k]
                if math.isnan(val):
                    y = bottom - 4
                else:
                    y = int(top + (plot_h - 20) * (1.0 - (float(val) - p_vmin) / vrange)) + 10
                x = int(xs[k])
                pts.append((x, y))
            if pts:
                pts_arr = np.array(pts, dtype=np.int32)
                color = (0, 0, 0) if name == 'Total' else (0, 120, 255) # Black line for Total
                thickness = 3 if name == 'Total' else 2
                cv2.polylines(canvas_cv, [pts_arr], False, color, thickness, lineType=cv2.LINE_AA)
                for (x, y) in pts:
                    cv2.circle(canvas_cv, (x, y), 3, (0, 80, 200), -1)
            
            # Store scale for text rendering
            ent['p_vmin'] = p_vmin
            ent['p_vmax'] = p_vmax
            ent['p_rmin'] = p_rmin
            ent['p_rmax'] = p_rmax

            # Plot smoothed rate on right axis
            srates = ent['smoothed_rates']
            if srates:
                rrange = p_rmax - p_rmin if p_rmax != p_rmin else 1.0
                pts_rate = []
                for k, srate in enumerate(srates):
                    if math.isnan(srate):
                        y_r = bottom - 4
                    else:
                        y_r = int(top + (plot_h - 20) * (1.0 - (srate - p_rmin) / rrange)) + 10
                    x = int(xs[k])
                    pts_rate.append((x, y_r))
                if pts_rate:
                    pts_rate_arr = np.array(pts_rate, dtype=np.int32)
                    try:
                        cv2.polylines(canvas_cv, [pts_rate_arr], False, (34, 139, 34), 1, lineType=cv2.LINE_AA)
                    except Exception:
                        pass
                    for (xr, yr) in pts_rate:
                        cv2.circle(canvas_cv, (xr, yr), 2, (34, 139, 34), -1)

        # --- Right Side: Villager Status ---
        # Draw a large box
        status_color = (100, 255, 100) if current_vill_status else (100, 100, 255) # Light Green or Light Red
        text_color_rgb = (0, 100, 0) if current_vill_status else (100, 0, 0) # Dark Green or Dark Red (RGB for PIL)
        
        box_margin = 20
        box_left = split_x + box_margin
        box_right = win_w - box_margin
        box_top = box_margin
        box_bottom = win_h - box_margin
        
        cv2.rectangle(canvas_cv, (box_left, box_top), (box_right, box_bottom), status_color, -1)
        cv2.rectangle(canvas_cv, (box_left, box_top), (box_right, box_bottom), (0,0,0), 2) # Black border (BGR)

        # --- Convert to PIL for Text Rendering ---
        # OpenCV is BGR, PIL is RGB. Convert color space.
        canvas_pil = Image.fromarray(cv2.cvtColor(canvas_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(canvas_pil)

        # Draw Text for Plots
        for i, ent in enumerate(entries):
            top = pad + i * (plot_h + pad)
            bottom = top + plot_h
            name = ent['name']
            
            p_vmin, p_vmax = ent['p_vmin'], ent['p_vmax']
            p_rmin, p_rmax = ent['p_rmin'], ent['p_rmax']
            
            # Use bold font for Total
            f_title = font_title_bold if name == 'Total' else font_title
            f_val = font_val_bold if name == 'Total' else font_val
            
            # Left axis labels (shifted slightly for larger font)
            draw.text((4, top + 18), f'{int(p_vmax)}', font=font_axis, fill=(0, 0, 0))
            draw.text((4, bottom - 12), f'{int(p_vmin)}', font=font_axis, fill=(0, 0, 0))
            
            # Right axis labels
            draw.text((plot_w - right_margin + 2, top + 18), f'{int(p_rmax)}', font=font_axis, fill=(34, 139, 34))
            draw.text((plot_w - right_margin + 2, bottom - 12), f'{int(p_rmin)}', font=font_axis, fill=(34, 139, 34))
            
            # Title (adjusted y)
            title_y = top + 10
            draw.text((left_margin - 10, title_y), name.title(), font=f_title, fill=(10, 10, 10))
            
            # Current value + inline rate text
            vals = ent['vals']
            cur = vals[-1] if vals else math.nan
            cur_text = str(int(cur)) if (not math.isnan(cur)) else 'NaN'
            
            # Calculate Rate
            latest_rate = 0.0
            try:
                if ent['smoothed_rates']:
                    latest_s = ent['smoothed_rates'][-1]
                    if latest_s is not None and (not math.isnan(latest_s)):
                        latest_rate = latest_s
            except Exception:
                latest_rate = 0.0
            
            latest_rate_text = f' ({latest_rate:.1f}/min)' if latest_rate != 0 else ''
            display_text = f'{cur_text}{latest_rate_text}'
            
            # Right-aligned current value
            bbox = draw.textbbox((0, 0), display_text, font=f_val)
            tw = bbox[2] - bbox[0]
            tx = max(left_margin, plot_w - right_margin - tw - 6)
            ty = top + 12
            draw.text((tx, ty), display_text, font=f_val, fill=(50, 150, 50))
            
            # --- Villager Stats Column ---
            # Centered in the right margin area
            col_center_x = plot_w - (right_margin // 2)
            vill_count = int(ent['vill_count'])
            
            # Per-villager rate
            per_vill_rate = 0.0
            if vill_count > 0:
                per_vill_rate = latest_rate / vill_count
            
            # Draw Villager Count
            v_text = f"{vill_count} vills"
            bbox = draw.textbbox((0, 0), v_text, font=font_vill_count)
            tw = bbox[2] - bbox[0]
            draw.text((col_center_x - tw // 2, top + 10), v_text, font=font_vill_count, fill=(0, 0, 0))
            
            # Draw Per-Vill Rate
            r_text = f"{per_vill_rate:.1f}/min"
            bbox = draw.textbbox((0, 0), r_text, font=font_vill_rate)
            tw = bbox[2] - bbox[0]
            draw.text((col_center_x - tw // 2, top + 30), r_text, font=font_vill_rate, fill=(100, 100, 100))

        # Draw Text for Villager Status
        label_lines = ["VILLAGER", "PRODUCTION"]
        status_text = "ACTIVE" if current_vill_status else "IDLE"
        
        y_cursor = box_top + 80
        for line in label_lines:
            bbox = draw.textbbox((0, 0), line, font=font_status_label)
            tw = bbox[2] - bbox[0]
            tx = box_left + (box_right - box_left - tw) // 2
            draw.text((tx, y_cursor), line, font=font_status_label, fill=text_color_rgb)
            y_cursor += 35
            
        y_cursor += 40
        bbox = draw.textbbox((0, 0), status_text, font=font_status_val)
        tw = bbox[2] - bbox[0]
        tx = box_left + (box_right - box_left - tw) // 2
        draw.text((tx, y_cursor), status_text, font=font_status_val, fill=text_color_rgb)

        # Elapsed time footer
        elapsed = (datetime.now() - start_time).total_seconds()
        draw.text((10, win_h - 15), f'Elapsed: {int(elapsed)}s', font=font_footer, fill=(50, 50, 50))

        # Convert back to OpenCV (BGR)
        return cv2.cvtColor(np.array(canvas_pil), cv2.COLOR_RGB2BGR)

    try:
        _LIVE_ANIMATION = True
        frame = 0
        while _LIVE_ANIMATION:
            try:
                # Capture once
                eco_summary = get_ss.get_bbox('eco_summary')
                screenshot = get_ss.capture_gfn_screen_region(eco_summary)
                
                # Get resource counts
                results = {}
                try:
                    results = summarize_eco(screenshot=screenshot)
                except Exception:
                    logging.exception('summarize_eco() failed')
                _append_values(results)
                
                # Check villager production
                try:
                    current_vill_status = check_villager_production(screenshot, vill_kernel)
                except Exception:
                    current_vill_status = False
                
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
