import cv2
import mediapipe as mp
import numpy as np
import joblib
import geocoder
import time
import os
import sys
import requests

try:
    import winsound  # Windows beep sound
    def beep():
        winsound.Beep(1000, 500)  # 1000 Hz for 500ms
except ImportError:
    def beep():
        os.system("echo -e '\a'")  # Mac/Linux beep sound

# Load trained SVM model
svm_model = joblib.load("gesture_svm.pkl")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Function to get GPS location
def get_location():
    try:
        location = geocoder.ip("me")
        return f"Latitude: {location.latlng[0]}, Longitude: {location.latlng[1]}"
    except:
        return "Location not available"

# Function to send SOS via SMS
def send_sos_sms(location):
    api_key = "JRMwfUtEX5lHZ4d2cBku8WFCQYVPNhGLvSTrqO6bnK137maixANZltCmvcTxdsVieUanFg8jq6wE9pkA"  # Replace with Fast2SMS API Key
    phone_number = "8585924194"  # Replace with recipient's phone number

    message = f"🚨 SOS ALERT! 🚨\nLocation: {location}\nNeed immediate help!"
    
    url = "https://www.fast2sms.com/dev/bulkV2"
    headers = {
        "authorization": api_key,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "route": "q",  # Quick SMS
        "message": message,
        "language": "english",
        "flash": 0,
        "numbers": phone_number
    }

    response = requests.post(url, headers=headers, data=data)
    
    if response.status_code == 200:
        print("✅ SOS SMS Sent Successfully!")
    else:
        print("❌ Failed to Send SOS SMS:", response.text)

# Open webcam
cap = cv2.VideoCapture(0)

print("🚀 Emergency SOS System Active. Wave (👋) to start countdown. Fist (✊) to cancel & quit.")

sos_triggered = False
cooldown_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    results = hands.process(rgb_frame)

    current_time = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract 21 landmark points
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks).reshape(1, -1)
            prediction = svm_model.predict(landmarks)[0]

            # Wave (👋) detected & cooldown over
            if prediction == 1 and not sos_triggered and current_time > cooldown_time:
                print("⏳ SOS Countdown Started (5s). Show Fist (✊) to cancel.")
                sos_triggered = True  
                
                start_time = time.time()

                # Start countdown
                while time.time() - start_time < 5:
                    remaining_time = 5 - int(time.time() - start_time)
                    print(f"SOS in {remaining_time}s... Show Fist (✊) to cancel.")
                    beep()  # Play beep sound

                    # Capture new frame during countdown
                    ret, frame = cap.read()
                    if not ret:
                        break

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb_frame)

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            landmarks = []
                            for lm in hand_landmarks.landmark:
                                landmarks.extend([lm.x, lm.y, lm.z])
                            landmarks = np.array(landmarks).reshape(1, -1)

                            new_prediction = svm_model.predict(landmarks)[0]
                            if new_prediction == 0:  # Fist (✊) detected
                                print("❌ SOS Canceled! Exiting...")
                                cap.release()
                                cv2.destroyAllWindows()
                                sys.exit()  # Quit program

                    time.sleep(1)  # Reduce CPU usage

                if sos_triggered:  # If not canceled, send SOS
                    location = get_location()
                    send_sos_sms(location)
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit()  # Auto-close after SOS is sent

    # Display the frame
    cv2.imshow("Emergency SOS System", frame)

    # Stop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("✅ Emergency SOS System Stopped.")


