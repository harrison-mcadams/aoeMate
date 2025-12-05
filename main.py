import os
import math
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional

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



def _parse_number_from_region(image: Image.Image, digit_kernels: dict, out_path: str = None, name: str = "debug") -> List[str]:
    """Parse a number from an image region using Sliding Window Template Matching.
    
    This approach scans the entire image for every digit (0-9), finding all matches
    above a threshold. It then uses Non-Maximum Suppression (NMS) to resolve overlaps.
    This is robust to broken contours and noise.
    """
    if image is None:
        return []

    # Convert to grayscale numpy array
    gray = np.array(image.convert('L'), dtype=np.float32)
    h, w = gray.shape
    
    # Store all candidate matches: (x, digit_str, score, width)
    candidates = []
    
    # Threshold for template matching
    # Since we are matching white text on dark background, correlation should be high.
    MATCH_THRESHOLD = 0.60 
    
    for d_str, k_gray in digit_kernels.items():
        if k_gray is None:
            continue
            
        try:
            kh, kw = k_gray.shape
            if kh > h or kw > w:
                continue
                
            # Match template
            res = cv2.matchTemplate(gray, k_gray, cv2.TM_CCOEFF_NORMED)
            
            # Find all locations above threshold
            locs = np.where(res >= MATCH_THRESHOLD)
            
            for pt in zip(*locs[::-1]): # zip(x_coords, y_coords)
                x = pt[0]
                y = pt[1]
                score = res[y, x]
                candidates.append({'x': x, 'y': y, 'd': str(d_str), 'score': score, 'w': kw})
                
        except Exception:
            continue
            
    if not candidates:
        return []
        
    # --- Non-Maximum Suppression (NMS) ---
    # We want to remove overlapping matches, keeping the highest score.
    # 1. Sort by score descending
    candidates.sort(key=lambda c: c['score'], reverse=True)
    
    final_matches = []
    
    while candidates:
        # Pick best remaining
        best = candidates.pop(0)
        final_matches.append(best)
        
        # Remove any remaining candidates that overlap significantly with 'best'
        # Overlap in X is the main concern for horizontal text.
        # We can define overlap as: |x1 - x2| < (w1 + w2) / 2 * overlap_ratio
        # Digits are close, so we need to be careful.
        # If centers are within ~5 pixels, it's likely the same digit or a mis-match.
        
        # Let's use a strict overlap: if X distance is less than half a digit width.
        min_dist = best['w'] * 0.6
        
        new_candidates = []
        for c in candidates:
            dist = abs(c['x'] - best['x'])
            if dist >= min_dist:
                new_candidates.append(c)
        candidates = new_candidates
        
    # Sort final matches by X to read left-to-right
    final_matches.sort(key=lambda c: c['x'])
    
    # Construct string
    result_str = "".join([m['d'] for m in final_matches])
    
    # Debug output if requested
    if out_path:
        try:
            debug_img = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
            for m in final_matches:
                x, y, w_match = m['x'], m['y'], m['w']
                # Use kh from last template (approximate if varying) or just use h/w from match
                # We stored w, but not h. Let's assume h is image height or close to it.
                cv2.rectangle(debug_img, (x, y), (x + w_match, y + h), (0, 255, 0), 1)
                cv2.putText(debug_img, m['d'], (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            cv2.imwrite(os.path.join(out_path, f"debug_digits_{name}.png"), debug_img)
        except Exception:
            pass

    return [result_str] if result_str else []


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
                # Sanity check: Resource icons should be on the left side
                if c['x'] < 100:
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
        # Generalized detection: crop to top-left ROI (wider to catch offset queue)
        # User requested buffer for ~6 slots. 
        # eco_summary is now 600px wide. Let's search the whole top strip.
        roi = screenshot.crop((0, 0, 600, 60))
        conv = analyze_ss.convolve_ssXkernel(roi, vill_kernel, out_path=out_path)
        
        # DEBUG: Print max score
        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(conv)
        # print(f"DEBUG: Villager Status Max Score: {max_val:.4f}")
        
        binary = analyze_ss.is_target_in_ss(conv, vill_kernel, out_path=out_path, threshold=0.60) # Lowered to 0.60
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
    # Increase default size (User requested 100% bigger)
    # Let's target ~1200x1000 or larger if screen permits
    win_w = int(os.environ.get('AOEMATE_WIN_W', str(min(1600, int(sw * 0.9)))))
    win_h = int(os.environ.get('AOEMATE_WIN_H', str(min(1200, int(sh * 0.9)))))

    cv_win_name = 'AOEMatePlot'
    try:
        cv2.namedWindow(cv_win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cv_win_name, win_w, win_h)
    except Exception:
        pass

    # Load villager kernel for status check
    try:
        vill_kernel = Image.open("/Users/harrisonmcadams/Desktop/villager_icon.png")
    except Exception:
        logging.warning("Could not load villager_icon.png for status check")
        vill_kernel = None

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
    
    # Pause Logic State
    from datetime import timedelta
    total_paused_duration = timedelta(seconds=0)
    pause_start_time = None
    is_paused = False
    
    # Pre-load fonts for waiting screen
    try:
        font_wait = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 40)
    except:
        font_wait = ImageFont.load_default()

    def _append_values(results, current_effective_time):
        # Use the effective time (adjusted for pauses)
        times.append(current_effective_time)
        
        for r in resource_names:
            # Resource Value
            v = None
            if results and r in results:
                v = results[r]
            
            new_val = math.nan
            if v is not None:
                sval = str(v).replace(',', '').strip()
                try:
                    new_val = int(sval)
                except Exception:
                    try:
                        new_val = float(sval)
                    except Exception:
                        new_val = math.nan
            
            # If detection failed (NaN), hold the last valid value (Extrapolation for Pause)
            if math.isnan(new_val):
                if data[r] and not math.isnan(data[r][-1]):
                    new_val = data[r][-1]
            
            data[r].append(new_val)
            
            # Villager Value
            v_vill = None
            if results and f'{r}_vills' in results:
                v_vill = results[f'{r}_vills']
            
            new_vill_val = math.nan
            if v_vill is not None:
                sval_v = str(v_vill).replace(',', '').strip()
                try:
                    new_vill_val = int(sval_v)
                except Exception:
                    new_vill_val = math.nan
            
            # If detection failed (NaN), hold the last valid value
            if math.isnan(new_vill_val):
                if vill_data[r] and not math.isnan(vill_data[r][-1]):
                    new_vill_val = vill_data[r][-1]
            
            vill_data[r].append(new_vill_val)
            
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
        
        # Pause Overlay Font
        font_pause = ImageFont.truetype(font_path_bold, 60)
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
        font_pause = ImageFont.load_default()

    def render_frame(current_w=None, current_h=None):
        # Use current window dimensions if provided, else default
        w = current_w if current_w else win_w
        h = current_h if current_h else win_h
        
        # Increase whitespace & margins for a more spacious look
        # Split layout: Left 70% for plots, Right 30% for Villager Status
        
        # Create canvas in OpenCV (BGR) for drawing lines/rectangles
        canvas_cv = 255 * np.ones((h, w, 3), dtype=np.uint8)
        
        # Define areas
        split_x = int(w * 0.7)
        
        # --- Left Side: Resource Plots ---
        pad = 12
        left_margin = 80
        # Increased right margin to separate axis labels from summary text
        right_margin = 220 
        
        plot_w = split_x
        
        # Add "Total" to the list of things to plot locally
        # Dynamic: Only show resources that have been detected in anchors
        # This allows flexible support for civs with/without Silver
        active_resources = []
        if _CACHED_ANCHORS:
            active_resources = [r for r in resource_names if r in _CACHED_ANCHORS]
        
        # Fallback if no anchors yet (or something went wrong), show core 4
        if not active_resources:
             active_resources = ['food', 'wood', 'gold', 'stone']
             
        plot_names = active_resources + ['Total']
        plot_h = (h - (len(plot_names) + 1) * pad) // len(plot_names)

        # First pass: compute plotting primitives... (unchanged)
        entries = []  # list of dicts with drawing info per resource
        global_vals = []
        global_rates = []
        
        # Calculate Total Data (unchanged)
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
                current_vill_count = 0
                for rn in resource_names:
                    v_list = vill_data.get(rn, [])
                    if v_list:
                        last_v = v_list[-1]
                        if last_v is not None and not math.isnan(last_v):
                            current_vill_count += last_v
            else:
                vals = data.get(r, [])
                v_list = vill_data.get(r, [])
                current_vill_count = v_list[-1] if v_list else 0
                if current_vill_count is None or (isinstance(current_vill_count, float) and math.isnan(current_vill_count)):
                    current_vill_count = 0
                
            n = len(vals)
            xs = np.linspace(left_margin, plot_w - right_margin, n) if n > 0 else np.array([])

            # --- New Rate Calculation: Cumulative Positive Flow ---
            # 1. Calculate diffs, ignoring negatives (spending)
            # 2. Compute cumulative sum of positive flow
            # 3. Calculate rate over a sliding window (e.g. 30s)
            
            smoothed_rates = []
            
            if n >= 2 and len(times) >= 2:
                # Prepare data
                # Forward-fill NaNs for rate computation
                clean_vals = []
                for ii, v in enumerate(vals):
                    if ii == 0:
                        clean_vals.append(0.0 if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v))
                    else:
                        if v is None or (isinstance(v, float) and math.isnan(v)):
                            clean_vals.append(clean_vals[-1])
                        else:
                            clean_vals.append(float(v))
                
                # Calculate Cumulative Positive Flow
                cum_flow = [0.0] * n
                current_cum = 0.0
                for ii in range(1, n):
                    diff = clean_vals[ii] - clean_vals[ii-1]
                    if diff > 0:
                        current_cum += diff
                    cum_flow[ii] = current_cum
                
                # Calculate Rate over Window
                # Use a fixed window of ~30 seconds (or RATE_TAU * 1.5)
                # Since RATE_TAU defaults to 20s, let's use 30s as a good default for "step-like" gathering.
                WINDOW_SEC = 30.0
                
                # Get timestamps in seconds relative to start
                try:
                    t_secs = [(t - times[0]).total_seconds() for t in times[-n:]] if len(times) >= n else [(times[i] - times[0]).total_seconds() for i in range(n)]
                except:
                    t_secs = [float(i) for i in range(n)]

                for ii in range(n):
                    t_curr = t_secs[ii]
                    val_curr = cum_flow[ii]
                    
                    # Find start of window
                    # We want t_curr - t_start >= WINDOW_SEC
                    # Search backwards
                    k = ii
                    while k > 0 and (t_curr - t_secs[k]) < WINDOW_SEC:
                        k -= 1
                    
                    t_start = t_secs[k]
                    val_start = cum_flow[k]
                    
                    dt = t_curr - t_start
                    
                    # Suppress rate calculation for the first few seconds to avoid initial spikes
                    # caused by dividing by small dt (e.g. gathering 10 res in 0.5s => 1200/min).
                    if dt < 5.0: 
                        rate = 0.0
                    else:
                        dv = val_curr - val_start
                        rate = (dv / dt) * 60.0
                        
                    smoothed_rates.append(rate)
            else:
                smoothed_rates = [0.0] * n

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

            # Define Colors (BGR)
            # Food: Red/Orange, Wood: Brown, Gold: Yellow, Stone: Grey, Silver: Cyan/White
            COLORS = {
                'food': (50, 50, 255),   # Red
                'wood': (42, 42, 165),   # Brownish
                'gold': (0, 215, 255),   # Gold
                'stone': (160, 160, 160),# Grey
                'silver': (255, 255, 200),# Light Cyan/Silver
                'Total': (20, 20, 20)    # Dark Grey/Black
            }
            
            base_color = COLORS.get(name, (0, 0, 0))
            
            # Draw Gridlines (Faint Grey)
            grid_color = (220, 220, 220)
            for ratio in [0.25, 0.5, 0.75]:
                y_grid = int(top + (plot_h - 20) * ratio) + 10
                cv2.line(canvas_cv, (left_margin, y_grid), (plot_w - right_margin, y_grid), grid_color, 1)

            # Plot resource totals (Raw Count) -> Solid Thick Line
            pts = []
            vrange = p_vmax - p_vmin if p_vmax != p_vmin else 1.0
            for k in range(n):
                val = vals[k]
                if math.isnan(val):
                    continue
                y = int(top + (plot_h - 20) * (1.0 - (float(val) - p_vmin) / vrange)) + 10
                x = int(xs[k])
                pts.append((x, y))
            
            if pts:
                pts_arr = np.array(pts, dtype=np.int32)
                thickness = 3 if name == 'Total' else 2
                cv2.polylines(canvas_cv, [pts_arr], False, base_color, thickness, lineType=cv2.LINE_AA)
            
            # Store scale for text rendering
            ent['p_vmin'] = p_vmin
            ent['p_vmax'] = p_vmax
            ent['p_rmin'] = p_rmin
            ent['p_rmax'] = p_rmax
            ent['color'] = base_color # Store for text

            # Plot smoothed rate on right axis -> Thin Solid Line
            srates = ent['smoothed_rates']
            if srates:
                rrange = p_rmax - p_rmin if p_rmax != p_rmin else 1.0
                pts_rate = []
                for k, srate in enumerate(srates):
                    if math.isnan(srate):
                        continue
                    y_r = int(top + (plot_h - 20) * (1.0 - (srate - p_rmin) / rrange)) + 10
                    x = int(xs[k])
                    pts_rate.append((x, y_r))
                
                if len(pts_rate) > 1:
                    # Draw Rate as Thin Solid Line
                    # Rate color: Match resource color (base_color)
                    rate_color = base_color
                    pts_rate_arr = np.array(pts_rate, dtype=np.int32)
                    cv2.polylines(canvas_cv, [pts_rate_arr], False, rate_color, 1, lineType=cv2.LINE_AA)

        # --- Right Side: Villager Status ---
        # Draw a large box
        status_color = (100, 255, 100) if current_vill_status else (100, 100, 255) # Light Green or Light Red
        text_color_rgb = (0, 100, 0) if current_vill_status else (100, 0, 0) # Dark Green or Dark Red (RGB for PIL)
        
        box_margin = 20
        box_left = split_x + box_margin
        box_right = w - box_margin
        box_top = box_margin
        box_bottom = h - box_margin
        
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
            
            # Use stored color for title (convert BGR to RGB)
            bgr = ent.get('color', (0, 0, 0))
            rgb_title = (bgr[2], bgr[1], bgr[0])
            
            # Use bold font for Total
            f_title = font_title_bold if name == 'Total' else font_title
            f_val = font_val_bold if name == 'Total' else font_val
            
            # Left Axis Labels (Right-aligned in left margin)
            # Margin is 80px. Align to x=75.
            vmax_str = f'{int(p_vmax)}'
            vmin_str = f'{int(p_vmin)}'
            
            bbox = draw.textbbox((0, 0), vmax_str, font=font_axis)
            tw = bbox[2] - bbox[0]
            draw.text((75 - tw, top + 18), vmax_str, font=font_axis, fill=(0, 0, 0))
            
            bbox = draw.textbbox((0, 0), vmin_str, font=font_axis)
            tw = bbox[2] - bbox[0]
            draw.text((75 - tw, bottom - 12), vmin_str, font=font_axis, fill=(0, 0, 0))
            
            # Right Axis Labels (Left-aligned next to plot)
            # Plot ends at plot_w - right_margin. Draw at +5px.
            rx = plot_w - right_margin + 5
            draw.text((rx, top + 18), f'{int(p_rmax)}', font=font_axis, fill=rgb_title)
            draw.text((rx, bottom - 12), f'{int(p_rmin)}', font=font_axis, fill=rgb_title)
            
            # Title (Left-aligned in left margin, above axis labels?)
            # Or inside plot? Let's put it at x=10 in left margin.
            title_y = top + 10
            draw.text((10, title_y), name.title(), font=f_title, fill=rgb_title)
            
            # --- Summary Data (Centered in remaining Right Margin) ---
            # Right margin starts at `plot_w - right_margin`.
            # Axis labels take ~40px.
            # Summary area: [plot_w - right_margin + 50, plot_w]
            summary_start = plot_w - right_margin + 50
            summary_width = right_margin - 50
            col_center_x = summary_start + (summary_width // 2)
            
            # 1. Resource Count & Rate
            vals = ent['vals']
            cur = vals[-1] if vals else math.nan
            cur_text = str(int(cur)) if (not math.isnan(cur)) else 'NaN'
            
            latest_rate = 0.0
            try:
                if ent['smoothed_rates']:
                    latest_s = ent['smoothed_rates'][-1]
                    if latest_s is not None and (not math.isnan(latest_s)):
                        latest_rate = latest_s
            except Exception:
                latest_rate = 0.0
            
            line1_text = f"{cur_text} ({latest_rate:.0f}/min)"
            
            # Draw Line 1
            bbox = draw.textbbox((0, 0), line1_text, font=f_val)
            tw = bbox[2] - bbox[0]
            draw.text((col_center_x - tw // 2, top + 10), line1_text, font=f_val, fill=rgb_title)
            
            # 2. Villager Count & Per-Vill Rate
            vill_count = int(ent['vill_count'])
            per_vill_rate = 0.0
            if vill_count > 0:
                per_vill_rate = latest_rate / vill_count
            
            line2_text = f"{vill_count} vills ({per_vill_rate:.1f}/min)"
            
            # Draw Line 2
            bbox = draw.textbbox((0, 0), line2_text, font=font_vill_rate)
            tw = bbox[2] - bbox[0]
            draw.text((col_center_x - tw // 2, top + 32), line2_text, font=font_vill_rate, fill=(80, 80, 80))

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
        # Use effective time
        effective_now = datetime.now() - total_paused_duration
        if is_paused and pause_start_time:
             # If currently paused, effective time is frozen at pause start
             effective_now = pause_start_time - total_paused_duration
             
        elapsed = (effective_now - start_time).total_seconds()
        draw.text((10, win_h - 15), f'Elapsed: {int(elapsed)}s', font=font_footer, fill=(50, 50, 50))

        # --- PAUSE OVERLAY ---
        if is_paused:
            # Draw semi-transparent overlay
            # PIL doesn't support alpha composite on RGB easily without converting to RGBA
            # Let's just draw a "PAUSED" text with a background box in the center
            
            pause_text = "PAUSED"
            bbox = draw.textbbox((0, 0), pause_text, font=font_pause)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            
            cx, cy = w // 2, h // 2
            
            # Draw box
            draw.rectangle((cx - tw//2 - 20, cy - th//2 - 20, cx + tw//2 + 20, cy + th//2 + 20), fill=(0, 0, 0))
            draw.text((cx - tw//2, cy - th//2), pause_text, font=font_pause, fill=(255, 255, 255))

        # Convert back to OpenCV (BGR)
        return cv2.cvtColor(np.array(canvas_pil), cv2.COLOR_BGR2RGB)

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
                
                # --- PAUSE DETECTION LOGIC ---
                # If all resources are None, we assume we are NOT in game (Paused/Menu)
                # Check if we have ANY valid data
                has_valid_data = False
                if results:
                    for k, v in results.items():
                        if v is not None:
                            has_valid_data = True
                            break
                
                now = datetime.now()
                
                if not has_valid_data:
                    # ENTER PAUSE STATE
                    if not is_paused:
                        is_paused = True
                        pause_start_time = now
                        logging.info("Game Paused (No resources detected)")
                    
                    # While paused, we do NOT append new values to the history.
                    # This effectively "freezes" the graph.
                    # We also do NOT check villager status to save resources/avoid noise.
                    current_vill_status = False 
                    
                else:
                    # ENTER ACTIVE STATE
                    if is_paused:
                        # Just resumed
                        pause_duration = now - pause_start_time
                        total_paused_duration += pause_duration
                        is_paused = False
                        pause_start_time = None
                        logging.info(f"Game Resumed. Paused for {pause_duration.total_seconds():.1f}s")
                    
                    # Append values using effective time
                    effective_time = now - total_paused_duration
                    _append_values(results, effective_time)
                
                    # Check villager production only when active
                    try:
                        current_vill_status = check_villager_production(screenshot, vill_kernel)
                    except Exception:
                        current_vill_status = False
                
                # Get current window size for responsive layout
                try:
                    rect = cv2.getWindowImageRect(cv_win_name)
                    # rect is (x, y, w, h)
                    # Note: getWindowImageRect might return -1 if window is closed or not ready
                    if rect and rect[2] > 0 and rect[3] > 0:
                        cur_w, cur_h = rect[2], rect[3]
                    else:
                        cur_w, cur_h = win_w, win_h
                except Exception:
                    cur_w, cur_h = win_w, win_h

                canvas = render_frame(cur_w, cur_h)
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
