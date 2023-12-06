import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

finger_tips = [8, 12, 16, 20]
thumb_tip = 4

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            # Accessing the landmarks by their position
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            # Draw circles around fingertips
            for tip_id in finger_tips:
                x, y = int(lm_list[tip_id].x * w), int(lm_list[tip_id].y * h)
                cv2.circle(img, (x, y), 10, (255, 0, 0), cv2.FILLED)

            # Check if fingers are folded
            folded_fingers = all(lm_list[tip_id].x < lm_list[tip_id - 2].x for tip_id in finger_tips)

            # Check if the thumb is raised up
            thumb_up = lm_list[thumb_tip].y < lm_list[thumb_tip - 1].y

            # Check if the thumb is raised down (DISLIKE)
            thumb_down = lm_list[thumb_tip].y > lm_list[thumb_tip - 1].y

            # Display "LIKE" if all fingers are folded and the thumb is raised up
            if folded_fingers and thumb_up:
                print("LIKE")
                cv2.putText(img, "LIKE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display "DISLIKE" if all fingers are folded and the thumb is raised down
            elif folded_fingers and thumb_down:
                print("DISLIKE")
                cv2.putText(img, "DISLIKE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            mp_draw.draw_landmarks(img, hand_landmark,
                                    mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec((0, 0, 255), 2, 2),
                                    mp_draw.DrawingSpec((0, 255, 0), 4, 2))

    cv2.imshow("hand tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
