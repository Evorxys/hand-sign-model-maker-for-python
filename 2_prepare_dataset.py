import os
import json

# Step 1: Load the Dataset
X = []  # This will hold the raw landmark data
y = []  # This will hold the corresponding gesture labels

base_dir = 'sign_language_dataset/'  # Define the base directory where your dataset is stored
gestures = os.listdir(base_dir)  # Get a list of all the gesture subdirectories

for gesture in gestures:
    gesture_dir = os.path.join(base_dir, gesture)  # Create the full path to the gesture directory
    for file_name in os.listdir(gesture_dir):
        file_path = os.path.join(gesture_dir, file_name)  # Create the full path to the JSON file
        
        # Open and read the JSON file
        with open(file_path, 'r') as f:
            landmarks = json.load(f)  # Load the landmarks from the JSON file
            
            X.append(landmarks)  # Append the landmarks to X
            y.append(gesture)  # Append the corresponding label to y

# Step 2: Flatten the Landmark Data
X_flat = []  # Initialize a new list to hold the flattened data

for landmarks in X:
    # Flatten the landmark data
    flattened_landmarks = [coord for lm in landmarks for coord in (lm['x'], lm['y'], lm['z'])]
    
    X_flat.append(flattened_landmarks)  # Append the flattened data to X_flat

# Step 3: Verify the Data
print(f"Number of samples: {len(X_flat)}")
print(f"Number of features per sample: {len(X_flat[0])}")

print("First sample's flattened landmarks:", X_flat[0])
print("First sample's label:", y[0])








# (Optional) Step 4: Prepare the Data for Model Training
import numpy as np
from sklearn.model_selection import train_test_split

# Convert the lists into numpy arrays
X_flat = np.array(X_flat)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)
