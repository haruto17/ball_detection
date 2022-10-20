import cv2 as cv
import numpy as np

capture = cv.VideoCapture(0)

while(True):
    ret,frame = capture.read()

    if not ret: break

    gray = cv.cvtColor(frame,cv.COLOR_RGB2GRAY)
    gray = cv.GaussianBlur(gray,(33,33),1)

    colimg = frame.copy()

    gray = cv.cvtColor(frame,cv.COLOR_RGB2GRAY)
    gray = cv.GaussianBlur(gray,(33,33),1)

    colimg = frame.copy()

    circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1.10,100,param1=100,param2=30,minRadius=30,maxRadius=70)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv.circle(colimg,(i[0],i[1]),i[2],(0,255,255),2)
            cv.circle(colimg,(i[0],i[1]),2,(0,0,255),3)


    cv.imshow('frame',colimg)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()