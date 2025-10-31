"""
analyze_ss.py

Utilities for simple template-matching based image analysis.

This module exposes two small, focused functions used by the live monitor:

- convolve_ssXkernel(ss, kernel, *, out_path=None)
    Run OpenCV's matchTemplate (normalized cross-correlation) between a
    screenshot `ss` (PIL Image) and a `kernel` (PIL Image). Returns the
    float32 response map (values typically in [-1, 1] for TM_CCOEFF_NORMED).
    If `out_path` is provided (a directory path), a visualization file
    `convolved.png` will be written there.

- is_target_in_ss(res, target=None, *, out_path=None, threshold=0.8, min_distance=10)
    Given the response map `res` produced by matchTemplate, finds local
    peaks above `threshold`. Enforces a minimum distance between peaks
    by dilating the response map. Returns True if >=1 peak is found.
    If `out_path` is provided the function will also write a heatmap with
    peaks and a histogram (if matplotlib is available).

Notes:
- These functions are intentionally lightweight and avoid heavy external
  dependencies unless requested (matplotlib is optional and only used for
  saving histograms when available).
- All file-writing is optional and only performed when `out_path` is set.
"""

import cv2, numpy as np
from PIL import Image
import os
from typing import Optional, List, Tuple, Union


def convolve_ssXkernel(ss: Image.Image, kernel: Image.Image, *, out_path: Optional[str] = None):
    """Run normalized template matching and optionally save a visualization.

    Inputs:
      - ss: PIL.Image screenshot (any mode; converted to grayscale internally)
      - kernel: PIL.Image template to search for (converted to grayscale)
      - out_path: optional directory path to save diagnostics (if provided)

    Returns:
      - res: numpy.ndarray (float32) response map from cv2.matchTemplate
    """
    # Convert inputs to grayscale float32 arrays for matchTemplate.
    ss_gray = np.array(ss.convert('L'), dtype=np.float32)
    k_gray = np.array(kernel.convert('L'), dtype=np.float32)

    # Use normalized cross-correlation (TM_CCOEFF_NORMED) for robust matching
    # that is typically invariant to constant brightness offsets.
    res = cv2.matchTemplate(ss_gray, k_gray, cv2.TM_CCOEFF_NORMED)

    # If caller requested debugging output, write a normalized visualization.
    if out_path:
        try:
            # Normalize response to 0..255 safely (avoid division by zero)
            rmin = float(res.min())
            rmax = float(res.max())
            denom = (rmax - rmin) if (rmax - rmin) != 0 else 1e-8
            viz = ((res - rmin) / denom * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(viz).save(os.path.join(out_path, 'convolved.png'))
        except Exception:
            # Never crash on debug writes; just continue silently.
            pass

    return res  # float32 map with values typically in [-1, 1]


def match_template_arrays(ss_gray: np.ndarray, k_gray: np.ndarray, *, out_path: Optional[str] = None) -> np.ndarray:
    """Run matchTemplate on precomputed grayscale arrays.

    Inputs:
      - ss_gray: 2D numpy array (float32) representing the screenshot in grayscale
      - k_gray: 2D numpy array (float32) representing the kernel in grayscale
      - out_path: optional directory path to save diagnostics (if provided)

    Returns:
      - res: numpy.ndarray (float32) response map from cv2.matchTemplate

    This function mirrors `convolve_ssXkernel` but operates on arrays to
    avoid repeated PIL.Image -> numpy conversions for callers that can precompute arrays.
    """
    # Ensure dtype float32
    ss_a = ss_gray.astype(np.float32)
    k_a = k_gray.astype(np.float32)

    res = cv2.matchTemplate(ss_a, k_a, cv2.TM_CCOEFF_NORMED)

    if out_path:
        try:
            rmin = float(res.min())
            rmax = float(res.max())
            denom = (rmax - rmin) if (rmax - rmin) != 0 else 1e-8
            viz = ((res - rmin) / denom * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(viz).save(os.path.join(out_path, 'convolved_array.png'))
        except Exception:
            pass

    return res


def is_target_in_ss(res: np.ndarray, target: Image.Image = None, *, out_path: Optional[str] = None, threshold: float = 0.8, min_distance: int = 10, return_peaks: bool = False) -> Union[bool, Tuple[bool, List[Tuple[int,int,float]]]]:
    """Detect peaks in a matchTemplate response map.

    Strategy:
      - Convert `res` to float32 (if not already).
      - Dilate the response map with a circular kernel sized from `min_distance`.
        This produces a map where each pixel contains the local maximum in the
        neighborhood; local peaks are pixels equal to the dilated map.
      - Keep peaks that are >= `threshold` and equal to the dilated value.

    Inputs:
      - res: response map from cv2.matchTemplate (numpy array)
      - target: optional PIL.Image (not used for detection but kept for API symmetry)
      - out_path: optional directory for diagnostic outputs (heatmap + histogram)
      - threshold: float minimum response to consider a peak (0..1 for TM_CCOEFF_NORMED)
      - min_distance: int minimum spacing (in pixels) between reported peaks

    Returns:
      - By default (return_peaks=False): a boolean (True if at least one peak
        meeting the criteria is found, else False) — this preserves backward
        compatibility with existing callers.
      - If return_peaks=True: returns a tuple (found, peaks) where `found` is a
        boolean and `peaks` is a list of (x, y, score) tuples for each detected peak.
    """
    if res is None:
        return False
    if not isinstance(res, np.ndarray):
        # Attempt to coerce to numpy array
        try:
            res = np.array(res, dtype=np.float32)
        except Exception:
            return False

    # Work with float32 for numeric stability
    res_f = res.astype(np.float32)

    # Build a dilation kernel whose size enforces the minimum peak distance.
    ksize = max(1, 2 * min_distance + 1)
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        dilated = cv2.dilate(res_f, kernel)
    except Exception:
        # If the specified kernel is invalid for any reason, fall back to 3x3.
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(res_f, kernel)

    # A pixel is a local maximum if it equals the dilated value (within a small epsilon)
    # and its value exceeds the detection threshold.
    local_max_mask = (res_f >= dilated - 1e-6) & (res_f >= threshold)

    coords = np.argwhere(local_max_mask)
    peaks = []
    # coords are in (row, col) order -> interpret as (y, x)
    for (y, x) in coords:
        peaks.append((int(x), int(y), float(res_f[y, x])))

    # If requested, write diagnostic images: heatmap with peaks and a histogram.
    if out_path:
        try:
            os.makedirs(out_path, exist_ok=True)
            # Heatmap visualization (safe normalization)
            rmin = float(res_f.min())
            rmax = float(res_f.max())
            denom = (rmax - rmin) if (rmax - rmin) != 0 else 1e-8
            viz = ((res_f - rmin) / denom * 255.0).astype(np.uint8)
            viz_color = cv2.applyColorMap(viz, cv2.COLORMAP_JET)
            for (x, y, score) in peaks:
                # Mark peaks in green on the heatmap
                cv2.circle(viz_color, (x, y), max(3, min_distance // 2), (0, 255, 0), 2)
            Image.fromarray(cv2.cvtColor(viz_color, cv2.COLOR_BGR2RGB)).save(os.path.join(out_path, 'convolved_peaks.png'))
        except Exception:
            pass

        # Save a histogram of response values if matplotlib is available. Force a
        # non-GUI backend to avoid macOS/AppKit exceptions when running headless.
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure()
            plt.hist(res_f.ravel(), bins=200)
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(os.path.join(out_path, 'convolve_hist_log.png'))
            plt.close()
        except Exception:
            pass

    # If caller requested peaks, return both boolean and peak list; otherwise
    # return a single boolean to preserve the original function contract.
    found = len(peaks) > 0
    if return_peaks:
        return (found, peaks)
    return found


if __name__ == "__main__":
    # Simple demo (does not alter library behavior): run matching on example files
    from PIL import Image
    try:
        demo_ss = Image.open("/Users/harrisonmcadams/Desktop/debug_screenshot_q.png")
        demo_kernel = Image.open("/Users/harrisonmcadams/Desktop/debug_target.png")
        out_path = '/Users/harrisonmcadams/Desktop/'
        convolved_image = convolve_ssXkernel(demo_ss, demo_kernel, out_path=out_path)
        binary = is_target_in_ss(convolved_image, demo_kernel, out_path=out_path)
        if binary:
            print('Villagers are producing!')
        else:
            print('Villagers are NOT producing! :-(')

    except Exception as e:
        print("Convolution failed:", e)