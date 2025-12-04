
import cv2
import numpy as np
from PIL import Image
import main
import analyze_ss
import os

# Load debug panel
img_path = os.path.expanduser("~/Desktop/debug_panel.png")
img = Image.open(img_path)
ss_gray = np.array(img.convert('L'), dtype=np.float32)

# Load kernels
resources = [
    ('food', 'food_icon.png'),
    ('wood', 'wood_icon.png'),
    ('gold', 'gold_icon.png'),
    ('stone', 'stone_icon.png'),
]
main._init_kernels_and_executors(resources)

# Find anchors
anchors = main._find_anchors(ss_gray, main._RESOURCE_KERNELS_GRAY)
print("Anchors:", anchors)

# Draw anchors
resource_map = {
    'food': 'food_icon.png',
    'wood': 'wood_icon.png',
    'gold': 'gold_icon.png',
    'stone': 'stone_icon.png',
}
debug_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
for name, (x, y) in anchors.items():
    fname = resource_map.get(name)
    if not fname: continue
    h, w = main._RESOURCE_KERNELS_GRAY[fname].shape
    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.putText(debug_img, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

cv2.imwrite(os.path.expanduser("~/Desktop/debug_panel_anchors.png"), debug_img)
