import cv2
import imagezmq
from pickle import NONE
import sys
import time
from Vision.feature_detection import ORB_FLANN_2 as ob


image_hub = imagezmq.ImageHub()
#connected = False
count = 0
while True:  # show streamed images until Ctrl-C
    jetson, image = image_hub.recv_image()
    if jetson:
        connected = True
        cv2.imshow(jetson, image) # 1 window for each RPi
        cv2.waitKey(1)
        image_hub.send_reply(b'OK')

    if connected:
        count += 1
        
        if count > 500:
            break

cv2.destroyAllWindows()