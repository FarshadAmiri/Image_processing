# Run the code; then press "h" to capture your photo (from webcam)
# and it will show you your image with "body pose alignments"
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt


cap = cv2.VideoCapture(1)
while True:
    ret, image = cap.read()
    if ret == True:
        cv2.imshow('Webcam', image)
        if cv2.waitKey(1) & 0xFF == ord('h'):
            cv2.waitKey(4000)
            ret, image = cap.read()
            my_image = image
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
plt.ion()
my_image = cv2.cvtColor(my_image, cv2.COLOR_BGR2RGB)
plt.imshow(my_image)
plt.show()

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_pose.Pose(
        static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:
    # Convert the BGR image to RGB and process it with MediaPipe Pose.
    results = pose.process(cv2.cvtColor(my_image, cv2.COLOR_BGR2RGB))

    # Print nose landmark.
    image_hight, image_width, _ = my_image.shape
    # if not results.pose_landmarks:
    #     continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_hight})'
    )

    # Draw pose landmarks.
    annotated_image = my_image.copy()
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=results.pose_landmarks,
        connections=mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec)
    # cv2_imshow(annotated_image)
# cv2.imshow('Webcam', annotated_image)
# cv2.imshow('Webcam', image)
# cv2.waitKey(10000)
# if cv2.waitKey(10000) & 0xFF == ord('q'):
#     break

plt.imshow(annotated_image)
plt.show()