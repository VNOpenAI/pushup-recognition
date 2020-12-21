from ispushingup import IsPushingUp
import cv2
from myutils import padding
import numpy as np 

model = IsPushingUp()
cap = cv2.VideoCapture(0)
cv2.namedWindow('a', cv2.WINDOW_NORMAL)
while(1):
    ret, frame = cap.read()
    frame_ = frame.copy()
    frame_ = padding(frame_)
    frame_ = cv2.resize(frame_, dsize=(224, 224))
    frame_ = (frame_ - 127.0) / 255.
    frame_ = np.expand_dims(frame_, axis=0)
    out = model.predict(frame)
    if out > 0.9:
        color = (255, 0, 0)
    else:
        color = (0, 0, 255)
    frame = cv2.putText(frame, str(out), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 2, cv2.LINE_AA)
    cv2.imshow('a', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break