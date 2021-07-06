import cv2
import mediapipe as mp

cap = cv2.VideoCapture(1)
while True:
    ret, image = cap.read()
    if ret == True:
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=2,
                min_detection_confidence=0.5) as face_mesh:
            # Convert the BGR image to RGB and process it with MediaPipe Face Mesh.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Draw face landmarks of each face.
            if not results.multi_face_landmarks:
                continue
            annotated_image = image.copy()
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
            # cv2_imshow(annotated_image)
        cv2.imshow('Webcam', annotated_image)
        # cv2.imshow('Webcam', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()