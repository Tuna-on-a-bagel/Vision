import cv2 as cv
import numpy as np
from vid_cap import vid_capture


if __name__ == "__main__":
    vis = vid_capture("ssh")
    #cap = cv.VideoCapture(0)
    count = 0
    #time.sleep(2)
    connect = False
    while True:
        #ret, frame = cap.read()
        ret, frame = vis.vid_feed()
        if ret:
            connect = True
            #print('ret')
            cv.imshow('frame', frame)
            cv.waitKey(1)
        if connect:
            count += 1
            if count > 500:
                break
    cv.destroyAllWindows()
    vis.cap.release()
