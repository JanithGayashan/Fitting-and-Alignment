import cv2
import numpy as np

points = []

def mouse_callback(event, x, y, flags, param):
    global points, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")
            cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Image", img_display)
        if len(points) == 4:
            print("\nFour points selected. Press any key to continue.")

# Load images
turf_img = cv2.imread("Assets/turf.jpg")
flag_img = cv2.imread("Assets/sri_lanka_flag.png")

if turf_img is None or flag_img is None:
    raise FileNotFoundError("Could not find the images. Check your paths.")

img_display = turf_img.copy()
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

print("Click 4 corners on the cricket turf.")
print("IMPORTANT: Click them in this order: Top-Left, Top-Right, Bottom-Right, Bottom-Left")
cv2.imshow("Image", img_display)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(points) != 4:
    print("You didn't select exactly 4 points. Exiting.")
    exit()

# Set up source and destination points
pts_dst = np.array(points, dtype=np.float32)
h, w, _ = flag_img.shape
pts_src = np.array([
    [0, 0],          # Top-Left
    [w - 1, 0],      # Top-Right
    [w - 1, h - 1],  # Bottom-Right
    [0, h - 1]       # Bottom-Left
], dtype=np.float32)

# Calculate Homography
H, status = cv2.findHomography(pts_src, pts_dst)

# Warp the flag
warped_flag = cv2.warpPerspective(flag_img, H, (turf_img.shape[1], turf_img.shape[0]))

# --- BETTER BLENDING METHOD (Alpha Blending) ---

# 1. Create a pure white float mask exactly the size of the flag
mask = np.ones_like(flag_img, dtype=np.float32)

# 2. Warp the mask using the same Homography
warped_mask = cv2.warpPerspective(mask, H, (turf_img.shape[1], turf_img.shape[0]))

# 3. Apply the desired opacity to the warped mask (e.g., 0.6 for 60% visibility)
opacity = 0.6 
alpha = warped_mask * opacity

# 4. Convert images to float for smooth mathematical blending
turf_float = turf_img.astype(float)
flag_float = warped_flag.astype(float)

# 5. Blend using formula: Output = Flag * Alpha + Turf * (1 - Alpha)
blended = flag_float * alpha + turf_float * (1.0 - alpha)

# Convert back to uint8 for saving/displaying
final_result = blended.astype(np.uint8)

# Show and save
cv2.imshow("Superimposed Flag", final_result)
print("\nSuccess! Press any key to close the final image.")
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("Results/superimposed_result.jpg", final_result)
print("Image saved as 'superimposed_result.jpg'")