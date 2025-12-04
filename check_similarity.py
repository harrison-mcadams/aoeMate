
import cv2
import numpy as np
import os

def load_gray(name):
    path = os.path.expanduser(f"~/Desktop/{name}")
    if not os.path.exists(path):
        print(f"Missing {path}")
        return None
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

gold = load_gray("gold_icon.png")
silver = load_gray("silver_icon.png")

if gold is not None and silver is not None:
    # Resize silver to match gold if needed
    if gold.shape != silver.shape:
        silver = cv2.resize(silver, (gold.shape[1], gold.shape[0]))
        
    res = cv2.matchTemplate(gold, silver, cv2.TM_CCOEFF_NORMED)
    print(f"Correlation between Gold and Silver: {res[0][0]:.4f}")
else:
    print("Could not load icons.")
