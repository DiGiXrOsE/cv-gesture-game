import cv2
import numpy as np
import os

# --- SETTINGS ---
LABEL = "punch" # Change to "block" then "idle"
SAVE_PATH = f"data_v2/{LABEL}"
os.makedirs(SAVE_PATH, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

def get_clean_mask(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Optimized Dual-Range
    lower1, upper1 = np.array([0, 40, 50]), np.array([20, 255, 255])
    lower2, upper2 = np.array([0, 50, 25]), np.array([20, 255, 100])
    
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    
    # Contour Cleaning: Keep only the hand
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)
    return clean

print(f"Recording for: {LABEL}. Press 's' to start/pause, 'q' to quit.")
recording = False

while count < 400: # We'll take 400 per gesture for extra "brain power"
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    x1, y1, x2, y2 = 200, 100, 450, 350
    roi = frame[y1:y2, x1:x2]
    mask = get_clean_mask(roi)
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    cv2.imshow("What the AI Sees", mask)
    
    key = cv2.waitKey(1)
    if key == ord('s'): recording = not recording
    if key == ord('q'): break
    
    if recording:
        cv2.imwrite(f"{SAVE_PATH}/{count}.jpg", mask)
        count += 1
        print(f"Saved {count}/400", end="\r")

cap.release()
cv2.destroyAllWindows()