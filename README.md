# aoeMate

**aoeMate** is a real-time Age of Empires 4 (AOE4) economy monitor. It captures a specific region of your screen (the resource panel), parses the resource counts using computer vision (template matching), and tracks your villager production queue.

It provides a live **visual overlay** of your economy and an **audio alert** if your Town Center is idle (no villagers queuing).

## Features

- **Real-time Resource Tracking**: Graphs Food, Wood, Gold, Stone, and Silver/Olive Oil.
- **Idle Queue Alert**: Plays a subtle "pulse" sound when your villager queue is empty.
- **Robust Digit Recognition**: Uses custom template matching with digit recognition tuned for the AOE4 UI font.
- **Multi-Monitor Support**: Specifically configured to capture from a secondary (left) monitor.

---

## Setup & Installation

1.  **Prerequisites**:

    - Python 3.10+ installed.
    - Windows OS (recommended for audio alerts and screen capture).

2.  **Install**:

    ```powershell
    # Create virtual environment
    python -m venv venv

    # Activate (PowerShell)
    .\venv\Scripts\Activate.ps1

    # Install dependencies
    pip install -r requirements.txt
    ```

3.  **Template Configuration**:
    The system relies on a set of reference icons in the `templates/` directory.

    - **Required Files**: `food.png`, `wood.png`, `gold.png`, `stone.png`, `villager_separator.png`.
    - **Villager Icons**: At least one `villager_*.png` file (e.g., `villager_generic.png`).
      - **Multi-Civ Support**: The system automatically detects ALL `villager_*.png` files (except `villager_separator.png`) and tries to match against each. This means you can add civ-specific villager templates like `villager_delhi.png`, `villager_english.png`, etc. for better detection accuracy.
    - **Digits**: `0.png` through `9.png`.

    **To Add/Update Templates**:

    1.  Take a high-resolution screenshot of the game UI (press `PrtScn`).
    2.  Open in an image editor (e.g., Paint, Photoshop).
    3.  Crop significantly tightly around the icon or number.
    4.  Save as a `.png` file in the `templates/` folder.
        _Note: The system is sensitive to the exact pixel structure, so ensure screenshots are from the same resolution used for playing (e.g., 2560x1600)._

    **Adding Civ-Specific Villager Templates**:

    If villager detection isn't working for a specific civilization:

    1.  Play a game as that civ and take a screenshot showing the villager queue icon in the top-left.
    2.  Crop the villager icon tightly (it shows the villager queuing in your TC).
    3.  Save as `villager_<civname>.png` (e.g., `villager_mongols.png`, `villager_abbasid.png`).
    4.  Restart aoeMate - it will automatically load and use the new template.

---

## Usage

### 1. Configure Screen Capture (`get_ss.py`)

By default, the script captures from the **Left-most monitor** (Index 2 in `mss`).

- If you play on your primary monitor, you may need to edit `get_ss.py`:
  ```python
  # Change this line in capture_gfn_screen_region:
  monitor = monitors[2]  # Left monitor
  # To:
  monitor = monitors[1]  # Primary monitor
  ```

### 2. Run the Monitor

```powershell
python main.py
```

- A window `AOEMatePlot` will open, showing live graphs.
- The console will log the current Villager Queue Score and detected resources.
- **Audio Alert**: You will hear a soft pulse if the Queue Score drops below `0.70`.

### 3. Controls

- **Exit**: Click the OpenCV window and press `q` or `Esc`.
- **Pause**: The system automatically detects pauses (black screens/menus) and stops logging.

---

## Troubleshooting

### "Extra Numbers" in Resource Counts

If you see resource counts like `10500` instead of `500`, the system might be detecting noise as digits.

- **Fix**: The `main.py` uses a high match threshold (`0.70`) and a "horizontal gap check" (>15px). Ensure your `templates/` digits are clean and do not include background noise.

### Villager Queue Not Detecting

- **Add your civ's template**: Different civilizations have different villager icons. If detection isn't working, add a `villager_<yourciv>.png` template (see Template Configuration above).
- **Check existing templates**: Ensure at least one `villager_*.png` in `templates/` roughly matches your in-game icon.
- **Check the logs**: When aoeMate starts, it logs which villager templates were loaded. Look for lines like `Loaded villager template(s): [...]`.
- **Threshold**: The detection threshold is set in `main.py` (currently `0.70`). If it fails to trigger, you might need to lower this, but `0.70` is tuned for high-confidence matching.

### Capture Region is Wrong

- The capture region is defined in `get_ss.py` under the `'eco_summary'` bounding box.
- Use a tool like ShareX or MSpaint to find the `top`, `left`, `width`, and `height` of your resource panel relative to the monitor, and update the values in `get_ss.py`.
