import cv2 as cv
import imagezmq
import numpy as np
import time

class vid_capture:

    source = ''
    filepath = ''

    def __init__(self, source, filepath=None):
        self.source = source
        self.filepath = filepath
        if (self.source == 'ssh') or (self.source == 'SSH'):
            print('ssh hub created')
            self.cap = imagezmq.ImageHub()
        elif (self.source == 'onboard') or (self.source == 'onBoard'):
            print('onboard capture initialized')
            self.cap = cv.VideoCapture(0)
        elif (self.source == 'fromfile') or (self.source == 'fromFile'):
            print('file capture intialized')
            self.cap = cv.VideoCapture(self.filepath)

    def vid_feed(self):
        '''A function to switch between ssh feed, webcam feed, and a video file read
        -------------------
        source = 'ssh' or 'onboard cam' or 'fromFile'
        '''
        if (self.source == 'ssh') or (self.source == 'SSH'):
            self.jetson, self.frame = self.cap.recv_image()
            if self.jetson:
                self.cap.send_reply(b'OK')
                return self.jetson, self.frame
            else: pass
        
        else:
            self.ret, self.frame = self.cap.read()
            if self.ret:
                return self.ret, self.frame
            else: print('no returned image')

'''
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

cv.destroyAllWindows()
'''

if __name__ == "__main__":
    vis = vid_capture("onBoard")
    #cap = cv.VideoCapture(0)
    
    #time.sleep(2)
    
    for i in range(0, 50):
        #ret, frame = cap.read()
        ret,frame = vis.vid_feed()
        if ret:
            print('ret')
            cv.imshow('frame', frame)
            cv.waitKey(1)
    cv.destroyAllWindows()
    vis.cap.release()
