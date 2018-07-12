import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
ret, img = cap.read()

fgbg = cv.createBackgroundSubtractorMOG2(0, varThreshold=150)

while cap.isOpened():
    ret, img = cap.read()
    img = cv.flip(img, 1)
    img = cv.rectangle(img, (381, 120), (381+28*9, 120+28*9), (255, 255, 255), 2)
    cv.imshow("image", img)
    cut = img[122:122+28*9, 382:382+28*9]
    gray = cv.cvtColor(cut, cv.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(cut)
    res = cv.bitwise_and(cut, cut, mask=fgmask)
    cv.imshow("Gray", res)
    k = cv.waitKey(10)
    if k == 27:
        break

cv.destroyAllWindows()


