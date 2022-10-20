from pdb import lasti2lineno
import cv2 as cv
import numpy as np

# color definition
RED = 1
YELLOW = 2
BLUE = 3

capture = cv.VideoCapture(0)

def find_rect_of_target_color(image, color_type):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV_FULL)
    h = hsv[:,:,0]
    s = hsv[:,:,1]

    # Red detection
    if color_type == RED:
        mask = np.zeros(h.shape, dtype = np.uint8)
        mask[((h < 20) | (h > 200)) & (s > 128)] = 255

    # Yellow detection
    if color_type == YELLOW:
        lower_yellow = np.array([22,93,0])
        upper_yellow = np.array([45,255,255])
        mask = cv.inRange(hsv, lower_yellow, upper_yellow)

    # Blue detection
    if color_type == BLUE:
        lower_blue = np.array([130,50,50])
        upper_blue = np.array([200,255,255])
        mask = cv.inRange(hsv, lower_blue,upper_blue)

    neiborhood = np.array([[0,1,0],
                           [1,1,1],
                           [0,1,0]],np.uint8)

    mask = cv.dilate(mask,
                     neiborhood,
                     iterations=2)

    mask = cv.erode(mask,
                    neiborhood,
                    iterations=2)

    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    rects = []
    for contour in contours:
        approx = cv.convexHull(contour)
        rect = cv.boundingRect(approx)
        rects.append(np.array(rect))
    return rects


while(True):
    ret,frame = capture.read()

    if not ret: break

    # red
    rects = find_rect_of_target_color(frame, RED)
    if len(rects) > 0:
        rect = max(rects, key=(lambda x: x[2] * x[3]))
        if rect[3] > 10:
            cv.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), (0,0,255),thickness=2)

    # yellow
    rects = find_rect_of_target_color(frame, YELLOW)
    if len(rects) > 0:
        rect = max(rects, key=(lambda x: x[2] * x[3]))
        if rect[3] > 10:
            cv.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), (0, 255, 255), thickness=2)

    # blue
    rects = find_rect_of_target_color(frame, BLUE)
    if len(rects) > 0:
        rect = max(rects, key=(lambda x: x[2] * x[3]))
        if rect[3] > 10:
            cv.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), (255,0,0), thickness=2)

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