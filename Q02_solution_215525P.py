import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- 1. CAMERA PARAMETERS ---
f_mm           = 8.0
Z_mm           = 720.0
pixel_pitch_mm = 0.0022
mm_per_pixel   = (pixel_pitch_mm * Z_mm) / f_mm 

# --- 2. LOAD AND PROCESS ---
img_path = "Assets/earrings.jpg"
img = cv2.imread(img_path)
if img is None:
    print("Error: Image not found.")
    exit()

img_annotated = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold to isolate gold from background
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# --- THE FIX: Morphological Closing ---
# This bridges the gap at the top of the earring so the hole becomes enclosed.
kernel = np.ones((15, 15), np.uint8)
closed_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Find Contours on the CLOSED mask
contours, hierarchy = cv2.findContours(closed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# --- 3. MEASUREMENTS ---
outer_data = {'w': 0, 'h': 0}
inner_data = {'w': 0, 'h': 0}
found_outer = False
found_inner = False

for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area < 2000: continue # Filter noise

    parent_idx = hierarchy[0][i][3]
    is_outer = (parent_idx == -1)
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(contour)
    color = (0, 255, 0) if is_outer else (255, 140, 0) # Green / Orange

    # Store first valid pair for report
    if is_outer and not found_outer:
        outer_data = {'w': w, 'h': h}
        found_outer = True
    elif not is_outer and not found_inner:
        inner_data = {'w': w, 'h': h}
        found_inner = True

    # Draw result
    cv2.rectangle(img_annotated, (x, y), (x + w, y + h), color, 3)
    cv2.line(img_annotated, (x, y + h//2), (x + w, y + h//2), color, 1)
    cv2.line(img_annotated, (x + w//2, y), (x + w//2, y + h), color, 1)

# --- 4. VISUALIZATION ---
fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={'width_ratios': [2.5, 1]})

h_px, w_px = img.shape[:2]
ax_img.imshow(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB), extent=[0, w_px*mm_per_pixel, h_px*mm_per_pixel, 0])
ax_img.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax_img.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax_img.grid(True, color='black', alpha=0.3)
ax_img.set_title("Corrected Dimensional Analysis (with Morphological Closing)", fontweight='bold')

# Result Box
ax_text.axis('off')
res = (
    f"CAMERA GEOMETRY\n{'='*25}\n"
    f"Scale: {mm_per_pixel:.4f} mm/px\n\n"
    f"OUTER HOOP (GREEN)\n{'─'*25}\n"
    f"Width : {outer_data['w']} px = {outer_data['w']*mm_per_pixel:.2f} mm\n"
    f"Height: {outer_data['h']} px = {outer_data['h']*mm_per_pixel:.2f} mm\n\n"
    f"INNER HOLE (ORANGE)\n{'─'*25}\n"
    f"Width : {inner_data['w']} px = {inner_data['w']*mm_per_pixel:.2f} mm\n"
    f"Height: {inner_data['h']} px = {inner_data['h']*mm_per_pixel:.2f} mm"
)
ax_text.text(0.05, 0.95, res, transform=ax_text.transAxes, fontsize=11, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#FDFEFE', edgecolor='#2E4053'))

plt.tight_layout()
plt.show()