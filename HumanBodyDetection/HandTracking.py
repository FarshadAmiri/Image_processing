import cv2
import mediapipe as mp

cap = cv2.VideoCapture(1)
while True:
    ret, image = cap.read()
    if ret == True:
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        # Run MediaPipe Hands.
        with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7) as hands:
            # Convert the BGR image to RGB, flip the image around y-axis for correct
            # handedness output and process it with MediaPipe Hands.
            results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))

            # Print handedness (left v.s. right hand).
            print(results.multi_handedness)

            if not results.multi_hand_landmarks:
                continue
            # Draw hand landmarks of each hand.
            image_hight, image_width, _ = image.shape
            annotatedh_image = cv2.flip(image.copy(), 1)
            for hand_landmarks in results.multi_hand_landmarks:
                # Print index finger tip coordinates.
                print(
                    f'Index finger tip coordinate: (',
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
                )
                mp_drawing.draw_landmarks(
                    annotatedh_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('Webcam', annotatedh_image)
        # cv2.imshow('Webcam', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()