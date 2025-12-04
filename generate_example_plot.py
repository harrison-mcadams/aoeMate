
import main
import cv2
import numpy as np
import os
from PIL import Image, ImageFont, ImageDraw
import math
from datetime import datetime

# Load debug screenshot
img_path = os.path.expanduser("~/Desktop/debug_screenshot_new.png")
try:
    img = Image.open(img_path)
except Exception as e:
    print(f"Error loading image: {e}")
    exit(1)

# Extract data
print("Extracting data...")
results = main.summarize_eco(screenshot=img)
print("Results:", results)

# Mock data structure for plotting
resource_names = ['food', 'wood', 'gold', 'stone', 'silver']
data = {r: [] for r in resource_names}

# Populate with the single data point
for r in resource_names:
    v = results.get(r)
    if v is not None:
        try:
            data[r].append(int(str(v).replace(',', '')))
        except:
            data[r].append(math.nan)
    else:
        data[r].append(math.nan)

# Render frame (adapted from main.py)
sw = 1280
sh = 800
win_w = min(1000, int(sw * 0.6))
win_h = min(900, int(sh * 0.6))

# Increase width for split layout
win_w = int(win_w * 1.3)

canvas_cv = 255 * np.ones((win_h, win_w, 3), dtype=np.uint8)

# Define areas
split_x = int(win_w * 0.7)

# --- Left Side: Resource Plots ---
pad = 12
left_margin = 80
right_margin = 140 # relative to split_x

plot_w = split_x
plot_h = (win_h - (len(resource_names) + 1) * pad) // len(resource_names)





# Pre-load fonts
try:
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    font_path_bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    
    # Increased sizes for better readability
    font_axis = ImageFont.truetype(font_path, 14)
    font_title = ImageFont.truetype(font_path, 18) # Larger title
    font_val = ImageFont.truetype(font_path, 20)   # Larger value
    
    # Bold fonts for Total row
    font_title_bold = ImageFont.truetype(font_path_bold, 18)
    font_val_bold = ImageFont.truetype(font_path_bold, 20)
    
    # Fonts for Villager Stats
    font_vill_count = ImageFont.truetype(font_path, 16)
    font_vill_rate = ImageFont.truetype(font_path, 12)
    
    font_status_label = ImageFont.truetype(font_path, 18)
    font_status_val = ImageFont.truetype(font_path, 36) # Much larger status
except Exception:
    # Fallback to default if Arial not found
    font_axis = ImageFont.load_default()
    font_title = ImageFont.load_default()
    font_val = ImageFont.load_default()
    font_title_bold = ImageFont.load_default()
    font_val_bold = ImageFont.load_default()
    font_vill_count = ImageFont.load_default()
    font_vill_rate = ImageFont.load_default()
    font_status_label = ImageFont.load_default()
    font_status_val = ImageFont.load_default()

# Add "Total" to the list of things to plot locally
plot_names = resource_names + ['Total']
plot_h = (win_h - (len(plot_names) + 1) * pad) // len(plot_names)

entries = []
global_vals = []

# Calculate Total Data (mocking it for single point since we don't have time series in this script)
# But wait, data dict has lists.
total_vals = []
# Assuming all lists are same length for this example script
n_points = 0
if resource_names:
    n_points = len(data[resource_names[0]])

for i in range(n_points):
    tot = 0.0
    valid_count = 0
    for r in resource_names:
        vals = data.get(r, [])
        if i < len(vals):
            v = vals[i]
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                tot += float(v)
                valid_count += 1
    if valid_count > 0:
        total_vals.append(tot)
    else:
        total_vals.append(math.nan)

for i, r in enumerate(plot_names):
    if r == 'Total':
        vals = total_vals
        # Mock villager count sum for Total
        current_vill_count = 0
        for rn in resource_names:
            v_res = results.get(f'{rn}_vills')
            if v_res:
                try:
                    current_vill_count += int(str(v_res).replace(',', ''))
                except: pass
    else:
        vals = data.get(r, [])
        # Get villager count from results
        v_res = results.get(f'{r}_vills')
        try:
            current_vill_count = int(str(v_res).replace(',', '')) if v_res else 0
        except:
            current_vill_count = 0
        
    n = len(vals)
    xs = np.linspace(left_margin, plot_w - right_margin, n) if n > 0 else np.array([])
    
    valid = [v for v in vals if not math.isnan(v)]
    if valid:
        vmin = min(valid)
        vmax = max(valid)
    else:
        vmin, vmax = 0.0, 1.0
        
    # Only contribute to global scale if NOT Total
    if r != 'Total':
        global_vals.extend([v for v in valid])
    
    entries.append({
        'name': r,
        'xs': xs,
        'vals': vals,
        'n': n,
        'vmin': vmin,
        'vmax': vmax,
        'smoothed_rates': [], # No rates for single point example
        'vill_count': current_vill_count
    })

if global_vals:
    G_vmin = min(global_vals)
    G_vmax = max(global_vals)
    if G_vmin == G_vmax:
        G_vmax = G_vmin + 1.0
else:
    G_vmin, G_vmax = 0.0, 1.0

# Draw shapes with OpenCV
for i, ent in enumerate(entries):
    top = pad + i * (plot_h + pad)
    bottom = top + plot_h
    left = left_margin
    right = plot_w - right_margin
    
    cv2.rectangle(canvas_cv, (left - 10, top), (right, bottom), (240, 240, 240), -1)
    
    xs = ent['xs']
    vals = ent['vals']
    n = ent['n']
    name = ent['name']
    
    # Determine scale for this plot
    if name == 'Total':
        p_vmin = ent['vmin']
        p_vmax = ent['vmax']
        if p_vmin == p_vmax: p_vmax += 1.0
    else:
        p_vmin, p_vmax = G_vmin, G_vmax
    
    # Store scale for text rendering
    ent['p_vmin'] = p_vmin
    ent['p_vmax'] = p_vmax
    
    pts = []
    vrange = p_vmax - p_vmin if p_vmax != p_vmin else 1.0
    for k in range(n):
        val = vals[k]
        if math.isnan(val):
            y = bottom - 4
        else:
            y = int(top + (plot_h - 20) * (1.0 - (float(val) - p_vmin) / vrange)) + 10
        x = int(xs[k])
        pts.append((x, y))
        
    if pts:
        pts_arr = np.array(pts, dtype=np.int32)
        color = (0, 0, 0) if name == 'Total' else (0, 120, 255)
        thickness = 3 if name == 'Total' else 2
        cv2.polylines(canvas_cv, [pts_arr], False, color, thickness, lineType=cv2.LINE_AA)
        for (x, y) in pts:
            cv2.circle(canvas_cv, (x, y), 3, (0, 80, 200), -1)

# Check villager production
print("Checking villager production...")
vills_producing = main.check_villager_production(img)
print(f"Villager production: {vills_producing}")

# --- Right Side: Villager Status ---
# Draw a large box
status_color = (100, 255, 100) if vills_producing else (100, 100, 255) # Light Green or Light Red
text_color_rgb = (0, 100, 0) if vills_producing else (100, 0, 0) # Dark Green or Dark Red (RGB for PIL)

box_margin = 20
box_left = split_x + box_margin
box_right = win_w - box_margin
box_top = box_margin
box_bottom = win_h - box_margin

cv2.rectangle(canvas_cv, (box_left, box_top), (box_right, box_bottom), status_color, -1)
cv2.rectangle(canvas_cv, (box_left, box_top), (box_right, box_bottom), (0,0,0), 2) # Black border (BGR)

# --- Convert to PIL for Text Rendering ---
# OpenCV is BGR, PIL is RGB. Convert color space.
canvas_pil = Image.fromarray(cv2.cvtColor(canvas_cv, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(canvas_pil)

# Draw Text for Plots
for i, ent in enumerate(entries):
    top = pad + i * (plot_h + pad)
    bottom = top + plot_h
    name = ent['name']
    
    p_vmin, p_vmax = ent['p_vmin'], ent['p_vmax']
    
    # Use bold font for Total
    f_title = font_title_bold if name == 'Total' else font_title
    f_val = font_val_bold if name == 'Total' else font_val
    
    # Left axis labels (shifted slightly for larger font)
    draw.text((4, top + 18), f'{int(p_vmax)}', font=font_axis, fill=(0, 0, 0))
    draw.text((4, bottom - 12), f'{int(p_vmin)}', font=font_axis, fill=(0, 0, 0))
    
    # Title (adjusted y)
    title_y = top + 10
    draw.text((left_margin - 10, title_y), name.title(), font=f_title, fill=(10, 10, 10))
    
    # Current value
    vals = ent['vals']
    cur = vals[-1] if vals else math.nan
    cur_text = str(int(cur)) if (not math.isnan(cur)) else 'NaN'
    display_text = f'{cur_text}'
    
    # Right-aligned current value
    bbox = draw.textbbox((0, 0), display_text, font=f_val)
    tw = bbox[2] - bbox[0]
    tx = max(left_margin, plot_w - right_margin - tw - 6)
    ty = top + 12
    draw.text((tx, ty), display_text, font=f_val, fill=(50, 150, 50))
    
    # --- Villager Stats Column ---
    # Centered in the right margin area
    col_center_x = plot_w - (right_margin // 2)
    vill_count = int(ent['vill_count'])
    
    # Per-villager rate (mocked as 0 for this example since we don't have rate history)
    per_vill_rate = 0.0
    
    # Draw Villager Count
    v_text = f"{vill_count} vills"
    bbox = draw.textbbox((0, 0), v_text, font=font_vill_count)
    tw = bbox[2] - bbox[0]
    draw.text((col_center_x - tw // 2, top + 10), v_text, font=font_vill_count, fill=(0, 0, 0))
    
    # Draw Per-Vill Rate
    r_text = f"{per_vill_rate:.1f}/min"
    bbox = draw.textbbox((0, 0), r_text, font=font_vill_rate)
    tw = bbox[2] - bbox[0]
    draw.text((col_center_x - tw // 2, top + 30), r_text, font=font_vill_rate, fill=(100, 100, 100))

# Draw Text for Villager Status
label_lines = ["VILLAGER", "PRODUCTION"]
status_text = "ACTIVE" if vills_producing else "IDLE"

y_cursor = box_top + 80
for line in label_lines:
    bbox = draw.textbbox((0, 0), line, font=font_status_label)
    tw = bbox[2] - bbox[0]
    tx = box_left + (box_right - box_left - tw) // 2
    draw.text((tx, y_cursor), line, font=font_status_label, fill=text_color_rgb)
    y_cursor += 35
    
y_cursor += 40
bbox = draw.textbbox((0, 0), status_text, font=font_status_val)
tw = bbox[2] - bbox[0]
tx = box_left + (box_right - box_left - tw) // 2
draw.text((tx, y_cursor), status_text, font=font_status_val, fill=text_color_rgb)

# Convert back to OpenCV (BGR) for saving
canvas_final = cv2.cvtColor(np.array(canvas_pil), cv2.COLOR_RGB2BGR)

out_path = os.path.expanduser("~/Desktop/example_analysis_plot.png")
cv2.imwrite(out_path, canvas_final)
print(f"Saved plot to {out_path}")
