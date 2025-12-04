
from PIL import Image
import os

img_path = os.path.expanduser("~/Desktop/debug_panel.png")
img = Image.open(img_path)

# Anchor based on Gold (Silver) at (5, 221)
# Assuming 30px spacing
icons = {
    'silver_extracted.png': (5, 221),
    'stone_extracted.png': (5, 251),
    'wood_extracted.png': (5, 191),
    'food_extracted.png': (5, 161),
}

for name, (x, y) in icons.items():
    # Crop 30x30 (approx icon size)
    crop = img.crop((x, y, x+30, y+30))
    crop.save(os.path.expanduser(f"~/Desktop/{name}"))
    print(f"Saved {name}")
