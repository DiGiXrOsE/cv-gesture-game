import cv2
import os

for gesture in ['punch', 'block', 'idle']:
    os.makedirs(f'dataset/{gesture}', exist_ok=True)

cap = cv2.VideoCapture(0)
counts = {'punch': 0, 'block': 0, 'idle': 0}

print("SYSTEM ONLINE. Place your hand INSIDE the green box.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Flip the frame so it acts like a mirror (easier for you to aim)
    frame = cv2.flip(frame, 1)
    
    # Define the Hitbox (Region of Interest) coordinates
    # Top-Left (x1, y1) to Bottom-Right (x2, y2)
    x1, y1, x2, y2 = 350, 100, 600, 350 
    
    # Draw the green box on the screen
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # CROP THE IMAGE to only include what is inside the green box
    roi = frame[y1:y2, x1:x2]
    
    cv2.imshow('Gesture Data Collector', frame)
    # Show exactly what the AI sees
    cv2.imshow('AI Vision (Hitbox)', roi) 
    
    key = cv2.waitKey(1) & 0xFF
    
    # IMPORTANT: We now save the 'roi', not the full 'frame'
    if key == ord('p'):
        cv2.imwrite(f"dataset/punch/punch_{counts['punch']}.jpg", roi)
        counts['punch'] += 1
        print(f"Captured PUNCH: {counts['punch']}")
    elif key == ord('b'):
        cv2.imwrite(f"dataset/block/block_{counts['block']}.jpg", roi)
        counts['block'] += 1
        print(f"Captured BLOCK: {counts['block']}")
    elif key == ord('i'):
        cv2.imwrite(f"dataset/idle/idle_{counts['idle']}.jpg", roi)
        counts['idle'] += 1
        print(f"Captured IDLE: {counts['idle']}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()