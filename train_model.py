import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Configuration
IMG_SIZE = 64 # We will compress the images to 64x64 pixels to speed up training
DATA_DIR = "dataset"
CATEGORIES = ["punch", "block", "idle"]

data = []
labels = []

print("Loading and flattening images from the vault...")

# 2. Preprocessing Pipeline
for category in CATEGORIES:
    path = os.path.join(DATA_DIR, category)
    class_num = CATEGORIES.index(category) # 0 for punch, 1 for block, 2 for idle
    
    for img_name in os.listdir(path):
        try:
            # 1. Read in FULL COLOR
            img_array = cv2.imread(os.path.join(path, img_name))
            
            # 2. Convert from BGR (OpenCV's default) to HSV (Hue, Saturation, Value)
            hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
            
            # 3. Define the universal color range for human skin in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # 4. Create the Mask: Skin becomes pure white, everything else becomes pure black
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # 5. Resize and Flatten
            resized_array = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
            flattened_array = resized_array.flatten()
            
            data.append(flattened_array)
            labels.append(class_num)
        except Exception as e:
            pass

# Convert lists to NumPy arrays for Scikit-Learn
X = np.array(data)
y = np.array(labels)

print(f"Total images processed: {len(X)}")

# 3. The Bake-Off (Train/Test Split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Model
print("Training Random Forest Model (this might take a few seconds)...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy on unseen test data: {accuracy * 100:.2f}%")

# 6. Save the Brain
with open('gesture_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved successfully as 'gesture_model.pkl'!")