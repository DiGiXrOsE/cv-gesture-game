import cv2
import os

# 1. Setup the vault: Create folders for our classes if they don't exist
for gesture in ['punch', 'block', 'idle']:
    os.makedirs(f'dataset/{gesture}', exist_ok=True)

# 2. Open the eye: Initialize the webcam
cap = cv2.VideoCapture(0)

# Track how many photos we've taken
counts = {'punch': 0, 'block': 0, 'idle': 0}

print("SYSTEM ONLINE.")
print("Controls: Hold 'p' for Punch, 'b' for Block, 'i' for Idle. Press 'q' to Quit.")

while True:
    ret, frame = cap.read()
    if not ret: 
        print("Error: Could not read from webcam.")
        break
    
    # Show the live feed
    cv2.imshow('Gesture Data Collector', frame)
    
    # Listen for key presses
    key = cv2.waitKey(1) & 0xFF
    
    # 3. The triggers: Save the frame based on what key you press
    if key == ord('p'):
        cv2.imwrite(f"dataset/punch/punch_{counts['punch']}.jpg", frame)
        counts['punch'] += 1
        print(f"Captured PUNCH: {counts['punch']}")
        
    elif key == ord('b'):
        cv2.imwrite(f"dataset/block/block_{counts['block']}.jpg", frame)
        counts['block'] += 1
        print(f"Captured BLOCK: {counts['block']}")
        
    elif key == ord('i'):
        cv2.imwrite(f"dataset/idle/idle_{counts['idle']}.jpg", frame)
        counts['idle'] += 1
        print(f"Captured IDLE: {counts['idle']}")
        
    elif key == ord('q'):
        print("Shutting down collector...")
        break

# Clean up and close the window
cap.release()
cv2.destroyAllWindows()