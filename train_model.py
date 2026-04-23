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
            # Read the image in Grayscale (color doesn't matter for shapes)
            img_array = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE)
            
            # Resize to 64x64
            resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            
            # Flatten the 2D grid into a 1D array of numbers
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