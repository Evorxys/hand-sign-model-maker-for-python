import cv2
import mediapipe as mp
import json
import os

# Initialize MediaPipe Hands and OpenCV
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
cap = cv2.VideoCapture(0)

gesture_name = 'LOVE'  # Change this for each gesture
save_dir = f'sign_language_dataset/{gesture_name}/'

# Create directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Start MediaPipe hands processing
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmarks
                landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in hand_landmarks.landmark]

                # Save the landmarks to a JSON file
                file_path = os.path.join(save_dir, f'{gesture_name}_{count}.json')
                with open(file_path, 'w') as f:
                    json.dump(landmarks, f)

                print(f'Saved {file_path}')
                count += 1

                # Draw hand landmarks with connections
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Display the frame with hand skeleton
        cv2.imshow('Capture Gesture with Skeleton', image)

        # Press 'q' to quit or 'c' to capture the next frame
        key = cv2.waitKey(10)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('c'):
            continue

cap.release()
cv2.destroyAllWindows()
