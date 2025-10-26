# aoeMate

aoeMate is a small Python toolkit that captures screen regions, runs simple image matching (template matching) or OCR, and provides a live visual status window to monitor in-game events (for example: detecting whether villagers are producing in AOE4).

This README documents setup, running the live loop, debugging tips, and where to find key settings.

---

Quick overview
--------------
- `getSS.py` - capture a region of the screen and return a PIL Image (optionally save to disk).
- `analyzeSS.py` - analysis helper functions (template matching using OpenCV's matchTemplate, peak detection, optional debug visualizations).
- `areVillsProducing.py` / `main.py` - interactive monitoring loops that repeatedly capture a region and display a large green/red status window to indicate detection.

Goals of this repository
------------------------
- Be able to capture a small screen region consistently.
- Run a fast, local template-matching pipeline to detect a given UI icon.
- Provide a simple visual indicator (large centered window) so you can run the monitor while playing.
- Keep disk I/O optional (debug-only) and provide clear debug outputs when requested.

Requirements
------------
- Python 3.10+ recommended
- Required Python packages (basic): `opencv-python`, `pillow`, `numpy`
- Optional packages for extra features and debugging: `matplotlib`, `pyautogui`, `google-cloud-vision` (if using cloud OCR)

Install (recommended in virtualenv)
----------------------------------
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the monitor
---------------
Start the live monitor (will open a centered status window):

```bash
python3 main.py
```

Controls
--------
- Press `q` or `Esc` while the status window has focus to quit the monitoring loop.
- Use `AOEMATE_POLL_MS` environment variable to change the polling interval (ms). Example: `AOEMATE_POLL_MS=250 python3 main.py`.

Debugging and examples
----------------------
- By default, the capture/analysis routines do not write debug files. To collect debug outputs (heatmaps, annotated images, crops), pass an `out_path` directory to the analysis/capture functions or enable debugging in the module demos.
- If the window isn't centered correctly on your machine, set `AOEMATE_SCREEN_W` and `AOEMATE_SCREEN_H` to your monitor resolution or install `pyautogui` in your environment so the code can detect screen size automatically.

API notes for developers
------------------------
- `getSS.capture_gfn_screen_region(bbox, *, out_path: Optional[str] = None) -> PIL.Image` : captures and returns an image, saves to `out_path` only if provided.
- `analyzeSS.convolveSSbyKernel(ss, kernel, *, out_path: Optional[str] = None)` : returns the matchTemplate response map; saves a visualization if `out_path` is provided.
- `analyzeSS.isTargetInSS(res, target=None, *, out_path: Optional[str] = None, threshold=0.8, min_distance=10)` : returns True if peaks are detected in response map; optionally saves heatmap/histogram to `out_path`.

Troubleshooting notes
---------------------
- On macOS, GUI-related libraries (tkinter, certain matplotlib backends) can raise Objective-C exceptions in some environments. The code attempts to avoid that by using non-GUI matplotlib backends (`Agg`) and by detecting screen size using safe methods.
- If you encounter crashes related to GUI toolkits, prefer running the analysis functions directly (they do not require a display) or set the environment variables mentioned above to bypass auto-detection.

Contributing
------------
- Keep `out_path` optional. Avoid writing to disk by default.
- Add unit tests or small synthetic-image scripts for vision routines so parameter changes can be validated automatically.

License
-------
This project is private; add a LICENSE file if you plan to open-source.
