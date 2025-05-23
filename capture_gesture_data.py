import cv2
import mediapipe as mp
import numpy as np
import csv
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Open webcam
cap = cv2.VideoCapture(0)

# CSV file to store landmarks
gesture_name = input("Enter gesture name (wave/fist): ").strip().lower()
file_name = f"{gesture_name}_data.csv"

# Open CSV file for writing
with open(file_name, mode="w", newline="") as file:
    writer = csv.writer(file)
    
    print(f"Recording {gesture_name} gesture... Press 'q' to stop.")
    time.sleep(2)  # Wait before starting

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

                # Save landmarks to CSV
                writer.writerow(landmarks)

                # Draw landmarks on screen
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the frame
        cv2.imshow("Gesture Capture", frame)
        # Stop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
print(f"Data saved in {file_name}")
