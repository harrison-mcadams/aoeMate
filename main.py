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


def summarize_eco():
    """Capture the eco_summary region and extract resource counts.

    Returns a dict: {'gold': '100', 'food': '20', 'stone': '0', 'wood': '200'} or None values.
    """
    logging.info('summarize_eco: entering')
    eco_summary = get_ss.get_bbox('eco_summary')
    logging.info('summarize_eco: eco_summary bbox=%s', eco_summary)
    screenshot = get_ss.capture_gfn_screen_region(eco_summary)

    out_path = None
    # Allow kernel path to be overridden with env var AOE_KERNEL_PATH
    kernelPath = _KERNEL_PATH

    resources = [
        ('food', 'food_icon.png'),
        ('wood', 'wood_icon.png'),
        ('gold', 'gold_icon.png'),
        ('stone', 'stone_icon.png'),
    ]

    results = {}
    eco_summary_bbox = eco_summary
    fudge_factor = 10

    # Initialize module-level caches and shared executors (idempotent)
    _init_kernels_and_executors(resources)
    resource_kernels = _RESOURCE_KERNELS
    resource_kernels_gray = _RESOURCE_KERNELS_GRAY
    digit_kernels_gray = _DIGIT_KERNELS_GRAY
    ex_digits = _EX_DIGITS
    ex_resources = _EX_RESOURCES

    # Precompute screenshot grayscale array once and reuse in workers
    ss_gray = np.array(screenshot.convert('L'), dtype=np.float32)

    def process_resource(res_name: str, icon_fname: str):
        # Use precomputed grayscale kernel array to avoid PIL->numpy each call
        k_gray = resource_kernels_gray.get(icon_fname)
        if k_gray is None:
            return None
        res_conv = analyze_ss.match_template_arrays(ss_gray, k_gray, out_path=out_path)
        found, peaks = analyze_ss.is_target_in_ss(res_conv, k_gray, out_path=out_path, return_peaks=True)
        if not found or not peaks:
            return None
        peak_x, peak_y, peak_score = peaks[0]
        top = eco_summary_bbox['top'] + int(peak_y)
        left = eco_summary_bbox['left'] + int(peak_x)
        # kernel array shape is (h, w)
        kh = k_gray.shape[0]
        kw = k_gray.shape[1]
        icon_w, icon_h = kw, kh
        count_width = 100
        count_bbox = {
            'top': int(top - fudge_factor),
            'left': int(left + icon_w + 2),
            'width': int(count_width),
            'height': int(icon_h + fudge_factor * 2)
        }
        try:
            count_img = get_ss.capture_gfn_screen_region(count_bbox, out_path=out_path)
        except Exception:
            return None
        count_img_gray = np.array(count_img.convert('L'), dtype=np.float32)

        per_digit_min_distance = 5
        futures_digits = []
        all_digit_peaks = []

        def detect_digit_worker(dd: int):
            dk = digit_kernels_gray.get(dd)
            if dk is None:
                return []
            conv_local = analyze_ss.match_template_arrays(count_img_gray, dk, out_path=out_path)
            found_d, peaks_d = analyze_ss.is_target_in_ss(conv_local, None, out_path=out_path, return_peaks=True, min_distance=per_digit_min_distance)
            if not found_d:
                return []
            return [(int(px), int(dd), float(pscore)) for (px, py, pscore) in peaks_d]

        # Submit digit detection to shared executor (if available)
        if ex_digits is not None:
            for d in range(10):
                if digit_kernels_gray.get(d) is None:
                    continue
                futures_digits.append(ex_digits.submit(detect_digit_worker, d))
        else:
            # fallback: run sequentially and collect results immediately
            for d in range(10):
                if digit_kernels_gray.get(d) is None:
                    continue
                try:
                    res_list = detect_digit_worker(d)
                    if res_list:
                        all_digit_peaks.extend(res_list)
                except Exception:
                    continue
            # since we've already processed synchronously, skip the futures loop
            if all_digit_peaks:
                all_digit_peaks.sort(key=lambda t: t[0])
                assembled = ''.join(str(int(d)) for (_, d, _) in all_digit_peaks)
                return assembled

        for f in futures.as_completed(futures_digits):
            try:
                res_list = f.result()
                if res_list:
                    all_digit_peaks.extend(res_list)
            except Exception:
                continue

        if not all_digit_peaks:
            return None
        all_digit_peaks.sort(key=lambda t: t[0])
        assembled = ''.join(str(int(d)) for (_, d, _) in all_digit_peaks)
        return assembled

    future_to_res = {}
    try:
        logging.info('Submitting %d resource tasks...', len(resources))
        if ex_resources is not None:
            future_to_res = {ex_resources.submit(process_resource, name, fname): (name, fname) for name, fname in resources}
            for fut in futures.as_completed(future_to_res):
                name, _ = future_to_res[fut]
                try:
                    val = fut.result()
                    results[name] = val
                except Exception:
                    logging.exception('Error processing resource %s', name)
                    results[name] = None
        else:
            # fallback to sequential processing
            for name, fname in resources:
                try:
                    results[name] = process_resource(name, fname)
                except Exception:
                    logging.exception('Error processing resource %s sequentially', name)
                    results[name] = None
    except Exception:
        logging.exception('Parallel submission failed; running sequentially')
        for name, fname in resources:
            try:
                val = process_resource(name, fname)
                results[name] = val
            except Exception:
                logging.exception('Sequential processing failed for %s', name)
                results[name] = None
    finally:
        # Do not shutdown shared executors here; they are global and cleaned up at exit.
        pass

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
        canvas = 255 * np.ones((win_h, win_w, 3), dtype=np.uint8)
        pad = 8
        plot_h = (win_h - (len(resource_names) + 1) * pad) // len(resource_names)
        left_margin = 60
        right_margin = 20
        for i, r in enumerate(resource_names):
            top = pad + i * (plot_h + pad)
            bottom = top + plot_h
            cv2.rectangle(canvas, (left_margin - 10, top), (win_w - right_margin, bottom), (240, 240, 240), -1)
            vals = data.get(r, [])
            n = len(vals)
            # Ensure rates variable exists regardless of data length
            rates = []
            if n >= 2:
                valid = [v for v in vals if not math.isnan(v)]
                if valid:
                    vmin = min(valid)
                    vmax = max(valid)
                    vrange = vmax - vmin if vmax != vmin else 1.0
                else:
                    vmin, vmax, vrange = 0.0, 1.0, 1.0
                xs = np.linspace(left_margin, win_w - right_margin, n)
                pts = []
                for k, val in enumerate(vals):
                    if math.isnan(val):
                        y = bottom - 4
                    else:
                        y = int(top + (plot_h - 20) * (1.0 - (val - vmin) / vrange)) + 10
                    x = int(xs[k])
                    pts.append((x, y))
                pts_arr = np.array(pts, dtype=np.int32)
                cv2.polylines(canvas, [pts_arr], False, (0, 120, 255), 2, lineType=cv2.LINE_AA)
                for (x, y) in pts:
                    cv2.circle(canvas, (x, y), 3, (0, 80, 200), -1)
                cv2.putText(canvas, f'{int(vmax)}', (6, top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(canvas, f'{int(vmin)}', (6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
                # --- compute and plot resource gather rate (units per minute) on right axis ---
                # Compute rates between consecutive points (aligned to same xs). rate[i] = (val[i]-val[i-1]) / dt * 60
                rates = []
                if len(times) >= 2:
                    # Build elapsed times corresponding to data points
                    t_secs = [ (t - times[0]).total_seconds() for t in times[-n:] ] if len(times) >= n else [ (times[i] - times[0]).total_seconds() for i in range(n) ]
                    # If t_secs length mismatches, fallback to uniform spacing of 1 sec
                    if len(t_secs) != n:
                        t_secs = list(range(n))
                    for i in range(n):
                        if i == 0:
                            # Use forward difference for first point
                            dt = t_secs[1] - t_secs[0] if n > 1 else 1.0
                            dv = (vals[1] - vals[0]) if n > 1 and not math.isnan(vals[1]) and not math.isnan(vals[0]) else 0.0
                        else:
                            dt = t_secs[i] - t_secs[i-1] if (t_secs[i] - t_secs[i-1]) != 0 else 1.0
                            dv = 0.0 if math.isnan(vals[i]) or math.isnan(vals[i-1]) else (vals[i] - vals[i-1])
                        if dt == 0:
                            rate = 0.0
                        else:
                            rate = (dv / dt) * 60.0  # units per minute
                        rates.append(rate)
                else:
                    rates = [0.0] * n

                # Apply exponential smoothing (EMA) to rates to average over a longer timescale.
                # RATE_TAU (seconds) controls the smoothing window; it's read from AOE_RATE_TAU_SEC.
                smoothed_rates = []
                if rates:
                    # t_secs should exist and align with rates; ensure fallback dt
                    # compute EMA with time-varying alpha: alpha = 1 - exp(-dt / tau)
                    for i, rate in enumerate(rates):
                        if i == 0:
                            smoothed_rates.append(rate)
                        else:
                            # determine dt between samples; fall back to 1 sec if unknown
                            try:
                                dt = t_secs[i] - t_secs[i - 1]
                                if dt <= 0:
                                    dt = 1.0
                            except Exception:
                                dt = 1.0
                            alpha = 1.0 - math.exp(-dt / RATE_TAU)
                            s_prev = smoothed_rates[-1]
                            s = alpha * rate + (1.0 - alpha) * s_prev
                            smoothed_rates.append(s)
                else:
                    smoothed_rates = []

                # Determine rate axis scale and map to pixel y positions on right using smoothed rates
                valid_rates = [r for r in smoothed_rates if not math.isnan(r)]
                if valid_rates:
                    rmin = min(valid_rates)
                    rmax = max(valid_rates)
                    rrange = rmax - rmin if rmax != rmin else 1.0
                    # Build points for rate curve from smoothed rates
                    pts_rate = []
                    for k, srate in enumerate(smoothed_rates):
                        if math.isnan(srate):
                            y_r = bottom - 4
                        else:
                            y_r = int(top + (plot_h - 20) * (1.0 - (srate - rmin) / rrange)) + 10
                        x = int(xs[k])
                        pts_rate.append((x, y_r))
                    # Draw dotted smoothed rate line
                    pts_rate_arr = np.array(pts_rate, dtype=np.int32)
                    try:
                        cv2.polylines(canvas, [pts_rate_arr], False, (34, 139, 34), 1, lineType=cv2.LINE_AA)
                    except Exception:
                        pass
                    for (xr, yr) in pts_rate:
                        cv2.circle(canvas, (xr, yr), 2, (34, 139, 34), -1)
                    # Right axis labels for smoothed rate
                    cv2.putText(canvas, f'{int(rmax)}', (win_w - right_margin - 2, top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (34, 139, 34), 1, cv2.LINE_AA)
                    cv2.putText(canvas, f'{int(rmin)}', (win_w - right_margin - 2, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (34, 139, 34), 1, cv2.LINE_AA)
            else:
                cv2.putText(canvas, 'waiting for data...', (left_margin + 10, top + plot_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1, cv2.LINE_AA)
            # Current value and inline smoothed rate
            cur = data[r][-1] if data[r] else math.nan
            cur_text = str(int(cur)) if (not math.isnan(cur)) else 'NaN'
            latest_rate_text = ''
            try:
                if smoothed_rates:
                    latest_s = smoothed_rates[-1]
                    if latest_s is not None and (not math.isnan(latest_s)):
                        latest_rate_text = f' ({latest_s:.1f}/min)'
            except Exception:
                latest_rate_text = ''

            display_text = f'{cur_text}{latest_rate_text}'
            # Title
            cv2.putText(canvas, r.title(), (left_margin - 10, top - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2, cv2.LINE_AA)
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
