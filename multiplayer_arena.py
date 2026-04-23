import cv2
import numpy as np
import pickle
import pyautogui
from collections import deque, Counter

# --- CONFIG ---
pyautogui.PAUSE = 0
IMG_SIZE = 64
CATEGORIES = ["punch", "block", "idle"]

# Load your model (Make sure this model was trained on the simple skin mask!)
with open('gesture_model.pkl', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

# Memory for stability (keeping this because it stops the keys from spamming)
p1_buffer = deque(maxlen=8)
p2_buffer = deque(maxlen=8)
last_p1, last_p2 = "idle", "idle"

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)

    # 1. SIMPLE SKIN MASK (The old reliable settings)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    full_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # 2. DEFINE THE TWO HITBOXES
    # Player 1 (Blue)
    p1_roi_mask = full_mask[100:350, 50:300]
    # Player 2 (Red)
    p2_roi_mask = full_mask[100:350, 340:590]

    # 3. PREDICT
    # P1
    res_p1 = cv2.resize(p1_roi_mask, (IMG_SIZE, IMG_SIZE)).flatten().reshape(1, -1)
    p1_now = CATEGORIES[model.predict(res_p1)[0]]
    p1_buffer.append(p1_now)

    # P2
    res_p2 = cv2.resize(p2_roi_mask, (IMG_SIZE, IMG_SIZE)).flatten().reshape(1, -1)
    p2_now = CATEGORIES[model.predict(res_p2)[0]]
    p2_buffer.append(p2_now)

    # 4. ACTION LOGIC
    if len(p1_buffer) == 8:
        # P1 Voting
        m1 = Counter(p1_buffer).most_common(1)[0][0]
        if m1 != last_p1:
            if m1 == "punch": pyautogui.press('f')
            elif m1 == "block": pyautogui.press('g')
            last_p1 = m1

        # P2 Voting
        m2 = Counter(p2_buffer).most_common(1)[0][0]
        if m2 != last_p2:
            if m2 == "punch": pyautogui.press('k')
            elif m2 == "block": pyautogui.press('l')
            last_p2 = m2

    # 5. DRAW HUD
    cv2.rectangle(frame, (50, 100), (300, 350), (255, 0, 0), 2)
    cv2.putText(frame, f"P1: {last_p1}", (50, 90), 1, 1.5, (255, 0, 0), 2)
    
    cv2.rectangle(frame, (340, 100), (590, 350), (0, 0, 255), 2)
    cv2.putText(frame, f"P2: {last_p2}", (340, 90), 1, 1.5, (0, 0, 255), 2)

    cv2.imshow("Simplified Multiplayer", frame)
    cv2.imshow("Mask View", np.hstack((p1_roi_mask, p2_roi_mask)))

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()