import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Load and Flatten the Dataset
X = []
y = []
base_dir = 'sign_language_dataset/'
gestures = os.listdir(base_dir)

for gesture in gestures:
    gesture_dir = os.path.join(base_dir, gesture)
    for file_name in os.listdir(gesture_dir):
        file_path = os.path.join(gesture_dir, file_name)
        with open(file_path, 'r') as f:
            landmarks = json.load(f)
            X.append(landmarks)
            y.append(gesture)

# Step 1a: Visualize the Skeletal Data (Optional)
def plot_skeleton(landmarks, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Example connections for a hand skeleton
    connections = [(0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                   (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                   (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                   (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                   (0, 17), (17, 18), (18, 19), (19, 20)]  # Pinky

    # Extract coordinates
    xs = [lm['x'] for lm in landmarks]
    ys = [lm['y'] for lm in landmarks]
    zs = [lm['z'] for lm in landmarks]

    # Plot the points
    ax.scatter(xs, ys, zs, c='r', marker='o')

    # Plot the connections
    for (i, j) in connections:
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]], 'b-')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

# Plot the first gesture's skeleton as an example
plot_skeleton(X[0])
plt.show()

# Step 2: Flatten the Dataset for Model Input
X_flat = []
for landmarks in X:
    X_flat.append([coord for lm in landmarks for coord in (lm['x'], lm['y'], lm['z'])])

# Convert to numpy arrays
X_flat = np.array(X_flat)
y = np.array(y)

# Step 3: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

# Step 4: Initialize and Train the Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# (Optional) Step 7: Save the Model
joblib.dump(model, 'sign_language_model.pkl')

# (Optional) Step 8: Load the Model
model = joblib.load('sign_language_model.pkl')
    