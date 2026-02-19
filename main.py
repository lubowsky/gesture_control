import cv2
import mediapipe as mp

# ---------------- ФУНКЦИЯ ОПРЕДЕЛЕНИЯ ПАЛЬЦЕВ ----------------
def fingers_up(hand_landmarks):
    fingers = []

    lm = hand_landmarks.landmark

    # Большой палец (сравниваем по X)
    if lm[4].x < lm[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Остальные пальцы (сравниваем по Y)
    tips = [8, 12, 16, 20]

    for tip in tips:
        if lm[tip].y < lm[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers
# ---------------------------------------------------------------

def detect_gesture(fingers):
    if fingers == [0, 0, 0, 0, 0]:
        return "PAUSE"

    if fingers == [1, 1, 1, 1, 1]:
        return "PLAY"

    if fingers == [0, 1, 1, 0, 0]:
        return "NEXT"

    return ""


# Инициализация MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Рисуем скелет руки
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # ---------------- ВОТ ЗДЕСЬ МЫ ВЫЗЫВАЕМ ФУНКЦИЮ ----------------
            fingers = fingers_up(hand_landmarks)

            gesture = detect_gesture(fingers)

            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"Fingers: {fingers}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            # ---------------------------------------------------------------

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
