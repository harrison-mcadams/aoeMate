
import cv2, numpy as np
from PIL import Image

def convolveSSbyKernel(ss: Image.Image, kernel: Image.Image, out_path=None):
    ss_gray = np.array(ss.convert('L'), dtype=np.float32)
    k_gray = np.array(kernel.convert('L'), dtype=np.float32)

    # normalized cross-correlation
    res = cv2.matchTemplate(ss_gray, k_gray, cv2.TM_CCOEFF_NORMED)

    return res  # float32 map with values ~[-1,1]

def isTargetInSS(res: Image.Image, target: Image.Image, out_path = None):
    # Binarize
    # Make a mask of res that is 1 where res > 0.8, else 0
    threshold = 0.8
    res_mask = np.where(res > threshold, res, 0)

    # Find peaks in the masked result
    min_distance = 10  # Minimum distance between peaks
    coordinates = np.argwhere(res_mask > 0)
    peaks = []
    if coordinates.size > 0:
        from scipy.spatial import cKDTree
        tree = cKDTree(coordinates)

        for coord in coordinates:
            if all(np.linalg.norm(coord - np.array(p)) >= min_distance for p in peaks):
                peaks.append(coord)
        for peak in peaks:
            y, x = peak
            cv2.circle(res, (x, y), 5, (1.0,), thickness=-1)  # Mark peak on res

    # Save a visualization (scale to 0..255)
    viz = ((res - res.min()) / (res.max() - res.min()) * 255).astype(np.uint8)

    if out_path:
        Image.fromarray(viz).save(out_path + 'convolved.png')

        import matplotlib.pyplot as plt
        plt.hist(res.ravel(), bins=200);
        plt.yscale('log');
        plt.savefig(out_path + 'convolve_hist_log.png')

    if len(peaks) > 0:
        binary = True
    else:
        binary = False

    return binary


def areVillsProducing(ss: Image.Image):
    # Load the village icon kernel
    kernelPath = '/Users/harrisonmcadams/Desktop/villager_icon.png'
    villKernel = Image.open("village_icon.png")  # Ensure this path is correct

    convolved_image(ss, villKernel)


if __name__ == "__main__":
    from PIL import Image
    # Basic self-check when executed as a script: convolve a demo screenshot with a demo kernel.
    try:

        demo_ss = Image.open("/Users/harrisonmcadams/Desktop/debug_screenshot_q.png")
        demo_kernel = Image.open("/Users/harrisonmcadams/Desktop/debug_target.png")

        out_path = '/Users/harrisonmcadams/Desktop/'

        convolved_image = convolveSSbyKernel(demo_ss, demo_kernel, out_path=out_path)

        binary = isTargetInSS(convolved_image, demo_kernel)

        if binary:
            print('Villagers are producing!')
        else:
            print('Villagers are NOT producing! :-(')

    except Exception as e:
        print("Convolution failed:", e)