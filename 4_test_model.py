import cv2
import joblib
import mediapipe as mp

# Load your pre-trained model
model = joblib.load(r"C:\Users\Administrator\Documents\WorkProfile\Learning_Python\sign_language_model.pkl")

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

def recognize_hand_gesture():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hand landmarks
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the hand landmarks on the frame (optional for visualization)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract the landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])

                # Use the landmarks as input to the model to predict the letter
                prediction = model.predict([landmarks])
                predicted_letter = prediction[0]

                # Display the predicted letter in the text widget
                text_output_camera.insert("1.0", f"Predicted letter: {predicted_letter}\n")

        # Show the frame with landmarks (optional for visualization)
        cv2.imshow('Hand Gesture Recognition', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Update the command of button_camera
button_camera.config(command=recognize_hand_gesture)
