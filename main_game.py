import cv2
import numpy as np
import pickle
import pyautogui

print("Loading AI Model...")
with open('gesture_model.pkl', 'rb') as f:
    model = pickle.load(f)

CATEGORIES = ["punch", "block", "idle"]
IMG_SIZE = 64
cap = cv2.VideoCapture(0)
last_move = "idle"

print("SYSTEM READY. Aim for the green box!")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Flip the frame to act like a mirror
    frame = cv2.flip(frame, 1)

    # EXACT same Hitbox coordinates
    x1, y1, x2, y2 = 350, 100, 600, 350 
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Crop the live feed to the Hitbox
    roi = frame[y1:y2, x1:x2]

    # --- THE UPGRADE: HSV SKIN MASK ---
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Preprocess the mask for the AI
    resized = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    flattened = resized.flatten().reshape(1, -1) 

    # Predict the gesture
    prediction_index = model.predict(flattened)[0]
    current_move = CATEGORIES[prediction_index]
    # -----------------------------------

    # Execute Keypresses
    if current_move == "punch" and last_move != "punch":
        print("BAM! (Pressing 'x')")
        pyautogui.press('x')  
    elif current_move == "block" and last_move != "block":
        print("SHIELD! (Pressing 'z')")
        pyautogui.press('z')  

    last_move = current_move

    # Heads Up Display
    cv2.putText(frame, f"ACTION: {current_move.upper()}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.imshow('Live Combat Feed', frame)
    
    # Debug Window: Look at this one to see what the AI sees!
    cv2.imshow('AI Vision (Skin Mask)', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()