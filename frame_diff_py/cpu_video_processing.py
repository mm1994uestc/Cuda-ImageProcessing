import cv2 as cv
import numpy as np

cv.namedWindow("Moving Detecting!")
cap = cv.VideoCapture(0)

W = 320
H = 240

cap.set(cv.CAP_PROP_FRAME_HEIGHT,H)
cap.set(cv.CAP_PROP_FRAME_WIDTH,W)

ret, frame = cap.read()
gray_a = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    gray_buff = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    for row in range(H):
        for clo in range(1,W):
            if abs(int(gray_buff[row][clo]) - int(gray_a[row][clo])) > 25:
                frame[row][clo] = [0,0,0]
    gray_a = gray_buff
    cv.imshow("Moving Detecting!",frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
