import cv2
import numpy as np

from facial_landmarks import FaceLandmarks

fl = FaceLandmarks()

cap = cv2.VideoCapture("example.mp4")
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    frame_copy = frame.copy()
    height, width, _ = frame.shape

    landmarks = fl.get_facial_landmarks(frame)
    ch = cv2.convexHull(landmarks)

    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [ch], True, 255, 3)
    cv2.fillConvexPoly(mask, ch, 255)

    frame_copy = cv2.blur(frame_copy, (27, 27))

    face_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask = mask)

    blurred_face = cv2.GaussianBlur(face_extracted, (19,19), 0)

    mask_bg = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(frame, frame, mask = mask_bg)

    result = cv2.add(background, face_extracted)

    cv2.imshow("Result", result)
    key = cv2.waitKey(30)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
