import cv2
import mediapipe as mp

cap = cv2.VideoCapture(1)
while True:
    ret, image = cap.read()
    if ret == True:
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        # Run MediaPipe Hands.
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.2, model_complexity=2) as pose:
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print nose landmark.
            image_height, image_width, _ = image.shape
            if not results.pose_landmarks:
                continue
            print(
                f'Nose coordinates: ('
                f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
                f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
            )

            # Draw pose landmarks.
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=results.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
        cv2.imshow('Webcam',annotated_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()