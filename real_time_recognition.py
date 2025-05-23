import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load trained SVM model
svm_model = joblib.load("gesture_svm.pkl")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Open webcam
cap = cv2.VideoCapture(0)

print("🚀 Real-Time Gesture Recognition Started. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract 21 landmark points
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])  # Store x, y, z

            # Convert landmarks to NumPy array
            landmarks = np.array(landmarks).reshape(1, -1)

            # Predict gesture
            prediction = svm_model.predict(landmarks)[0]

            # Determine gesture name
            gesture_name = "Wave (👋)" if prediction == 1 else "Fist (✊)"

            # Display gesture
            cv2.putText(frame, f"Gesture: {gesture_name}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw landmarks on screen
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow("Real-Time Gesture Recognition", frame)

    # Stop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("✅ Real-Time Recognition Stopped.")
