import cv2
import numpy as np
import math

def get_color_name(hsv_color):
    """Lookup array for color ranges."""
    h, s, v = hsv_color
    if v < 40: return "Black"
    if s < 40: return "White" if v > 200 else "Gray"

    # (Lower_Hue, Upper_Hue, Color_Name)
    color_ranges = [
        (0, 10, "Red"), (11, 25, "Orange"), (26, 35, "Yellow"),
        (36, 85, "Green"), (86, 130, "Blue"), (131, 155, "Violet"),
        (156, 170, "Pink"), (171, 180, "Red")
    ]

    for lower, upper, name in color_ranges:
        if lower <= h <= upper:
            return name
    return "Unknown"

def process_scene(image_path):
    img = cv2.imread(image_path)
    if img is None: return
    
    # 1. Preprocessing for Shape Detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)[1]
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return

    # 2. Isolate the Largest Contour
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    # --- SHAPE ANALYSIS ---
    approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)
    
    shape_label = "Unknown Shape"
    defect_count = 0
    if defects is not None:
        for i in range(defects.shape[0]):
            if defects[i, 0][3] > 1000: defect_count += 1
    
    if defect_count >= 4:
        shape_label = "X-Shape"
    elif len(approx) == 4:
        shape_label = "Rectangle"
    elif len(cnt) >= 5:
        _, (w, h), _ = cv2.fitEllipse(cnt)
        ratio = max(w, h) / min(w, h)
        shape_label = "Circle" if ratio <= 1.15 else "Ellipse"

    # --- COLOR ANALYSIS (Focused on the Object Only) ---
    # Create a mask of the largest contour
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    
    # Get pixels belonging only to the object
    object_pixels = img[mask == 255]
    
    # Use K-Means on the object's pixels to find the dominant color
    pixels = np.float32(object_pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    dominant_bgr = np.uint8(centers[0])
    hsv_color = cv2.cvtColor(np.uint8([[dominant_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    color_label = get_color_name(hsv_color)

    # 3. Final Visualization
    full_label = f"{color_label} {shape_label}"
    cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3)
    
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
        cv2.putText(img, full_label, (cX - 60, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Object Identified", img)
    cv2.waitKey(0)

process_scene('images/Rock.jpg')