import cv2
import mediapipe as mp
import numpy as np
import joblib

model = joblib.load("gesture_model.pkl")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

sequence = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    frame_data = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            for lm in hand_landmarks.landmark:
                frame_data.extend([lm.x, lm.y, lm.z])

    # если одна рука — дополняем нулями
    while len(frame_data) < 126:
        frame_data.append(0)

    sequence.append(frame_data)

    if len(sequence) > 30:
        sequence.pop(0)

    gesture = ""

    if len(sequence) == 30:
        input_data = np.array(sequence).flatten().reshape(1, -1)
        gesture = model.predict(input_data)[0]

    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()