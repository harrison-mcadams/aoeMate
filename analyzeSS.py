import cv2, numpy as np
from PIL import Image
import os
from typing import Optional

def convolveSSbyKernel(ss: Image.Image, kernel: Image.Image, *, out_path: Optional[str] = None):
    ss_gray = np.array(ss.convert('L'), dtype=np.float32)
    k_gray = np.array(kernel.convert('L'), dtype=np.float32)

    # normalized cross-correlation
    res = cv2.matchTemplate(ss_gray, k_gray, cv2.TM_CCOEFF_NORMED)

    # Optionally save visualization
    if out_path:
        try:
            # normalize safely (avoid division by zero)
            rmin = float(res.min())
            rmax = float(res.max())
            denom = (rmax - rmin) if (rmax - rmin) != 0 else 1e-8
            viz = ((res - rmin) / denom * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(viz).save(os.path.join(out_path, 'convolved.png'))
        except Exception:
            pass

    return res  # float32 map with values ~[-1,1]


def isTargetInSS(res: np.ndarray, target: Image.Image = None, *, out_path: Optional[str] = None, threshold: float = 0.8, min_distance: int = 10) -> bool:
    """Detect peaks in the matchTemplate result `res`.

    Uses a simple local-maximum test by dilating the response map with a
    footprint sized according to `min_distance`, then keeping points that
    are equal to the dilated map and exceed `threshold`.

    If out_path is provided, saves a visualization (convolved heatmap with
    circles marking peaks) and a histogram (if matplotlib is available).
    Returns True if at least one peak is found.
    """
    if res is None:
        return False
    if not isinstance(res, np.ndarray):
        # try to convert
        try:
            res = np.array(res, dtype=np.float32)
        except Exception:
            return False

    # Ensure float32
    res_f = res.astype(np.float32)

    # Create dilation kernel to enforce a minimum peak distance
    ksize = max(1, 2 * min_distance + 1)
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        dilated = cv2.dilate(res_f, kernel)
    except Exception:
        # fallback: use a 3x3 kernel
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(res_f, kernel)

    # Local maxima mask: equal to dilated and above threshold
    local_max_mask = (res_f >= dilated - 1e-6) & (res_f >= threshold)

    coords = np.argwhere(local_max_mask)
    peaks = []
    # coords are (row, col) -> (y, x)
    for (y, x) in coords:
        peaks.append((int(x), int(y), float(res_f[y, x])))

    # optionally save visualization and histogram
    if out_path:
        try:
            os.makedirs(out_path, exist_ok=True)
            # heatmap viz
            # guard against zero range
            rmin = float(res_f.min())
            rmax = float(res_f.max())
            denom = (rmax - rmin) if (rmax - rmin) != 0 else 1e-8
            viz = ((res_f - rmin) / denom * 255.0).astype(np.uint8)
            # convert to color for drawing
            viz_color = cv2.applyColorMap(viz, cv2.COLORMAP_JET)
            for (x, y, score) in peaks:
                cv2.circle(viz_color, (x, y), max(3, min_distance//2), (0, 255, 0), 2)
            Image.fromarray(cv2.cvtColor(viz_color, cv2.COLOR_BGR2RGB)).save(os.path.join(out_path, 'convolved_peaks.png'))
        except Exception:
            pass

        # try to save histogram if matplotlib is present
        try:
            # force non-GUI backend to avoid macOS NSException (Cocoa) when no display is available
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

    return len(peaks) > 0

if __name__ == "__main__":
    from PIL import Image
    # Basic self-check when executed as a script: convolve a demo screenshot with a demo kernel.
    try:

        demo_ss = Image.open("/Users/harrisonmcadams/Desktop/debug_screenshot_q.png")
        demo_kernel = Image.open("/Users/harrisonmcadams/Desktop/debug_target.png")

        out_path = '/Users/harrisonmcadams/Desktop/'

        convolved_image = convolveSSbyKernel(demo_ss, demo_kernel, out_path=out_path)

        binary = isTargetInSS(convolved_image, demo_kernel, out_path=out_path)

        if binary:
            print('Villagers are producing!')
        else:
            print('Villagers are NOT producing! :-(')

    except Exception as e:
        print("Convolution failed:", e)