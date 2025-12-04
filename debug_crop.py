
from PIL import Image
import os

img_path = os.path.expanduser("~/Desktop/macedonia_debug.jpg")
img = Image.open(img_path)
# Crop bottom left 300x400
# Image size is 1024x576
w, h = img.size
crop = img.crop((0, h-400, 300, h))
crop.save(os.path.expanduser("~/Desktop/debug_panel.png"))
print(f"Saved debug_panel.png size: {crop.size}")
