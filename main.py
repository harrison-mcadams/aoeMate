import os
import time
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
    kernelPath = '/Users/harrisonmcadams/Desktop/'

    resources = [

        ('food', 'food_icon.png'),
        ('wood', 'wood_icon.png'),
        ('gold', 'gold_icon.png'),
        ('stone', 'stone_icon.png'),

    ]

    results = {}
    eco_summary_bbox = eco_summary
    fudge_factor = 10

    # Load resource kernels and digit kernels
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

    max_workers = max(2, min(8, (os.cpu_count() or 4)))
    digit_kernels_gray = {d: (np.array(k.convert('L'), dtype=np.float32) if k is not None else None) for d, k in digit_kernels.items()}

    ex_digits = futures.ThreadPoolExecutor(max_workers=max_workers)
    ex_resources = futures.ThreadPoolExecutor(max_workers=min(len(resources), max_workers))

    def process_resource(res_name: str, icon_fname: str):
        kernel = resource_kernels.get(icon_fname)
        if kernel is None:
            return None
        res_conv = analyze_ss.convolve_ssXkernel(screenshot, kernel, out_path=out_path)
        found, peaks = analyze_ss.is_target_in_ss(res_conv, kernel, out_path=out_path, return_peaks=True)
        if not found or not peaks:
            return None
        peak_x, peak_y, peak_score = peaks[0]
        top = eco_summary_bbox['top'] + int(peak_y)
        left = eco_summary_bbox['left'] + int(peak_x)
        icon_w, icon_h = kernel.size
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

        def detect_digit_worker(dd: int):
            dk = digit_kernels_gray.get(dd)
            if dk is None:
                return []
            conv_local = analyze_ss.match_template_arrays(count_img_gray, dk, out_path=out_path)
            found_d, peaks_d = analyze_ss.is_target_in_ss(conv_local, None, out_path=out_path, return_peaks=True, min_distance=per_digit_min_distance)
            if not found_d:
                return []
            return [(int(px), int(dd), float(pscore)) for (px, py, pscore) in peaks_d]

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
                continue

        if not all_digit_peaks:
            return None
        all_digit_peaks.sort(key=lambda t: t[0])
        assembled = ''.join(str(int(d)) for (_, d, _) in all_digit_peaks)
        return assembled

    future_to_res = {}
    try:
        logging.info('Submitting %d resource tasks...', len(resources))
        future_to_res = {ex_resources.submit(process_resource, name, fname): (name, fname) for name, fname in resources}
        for fut in futures.as_completed(future_to_res):
            name, _ = future_to_res[fut]
            try:
                val = fut.result()
                results[name] = val
            except Exception:
                logging.exception('Error processing resource %s', name)
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
        try:
            ex_digits.shutdown(wait=True)
        except Exception:
            pass
        try:
            ex_resources.shutdown(wait=True)
        except Exception:
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
    times = []q
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
        if len(times) > max_points:
            del times[:-max_points]
            for r in resource_names:
                del data[r][:-max_points]

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
            else:
                cv2.putText(canvas, 'waiting for data...', (left_margin + 10, top + plot_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1, cv2.LINE_AA)
            cur = data[r][-1] if data[r] else math.nan
            cur_text = str(int(cur)) if (not math.isnan(cur)) else 'NaN'
            cv2.putText(canvas, r.title(), (left_margin - 10, top - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2, cv2.LINE_AA)
            cv2.putText(canvas, cur_text, (win_w - right_margin - 60, top + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 150, 50), 2, cv2.LINE_AA)
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
    live_monitor_resources(poll_sec=poll)
