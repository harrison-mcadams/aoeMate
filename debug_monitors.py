from mss import mss
import json

with mss() as sct:
    monitors = sct.monitors
    print("Available monitors:")
    for i, m in enumerate(monitors):
        print(f"Monitor {i}: {m}")
