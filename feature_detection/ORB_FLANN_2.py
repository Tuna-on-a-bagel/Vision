import cv2 as cv
import numpy as np
from pickle import NONE
import pymagsac
from copy import deepcopy

#some setup:
#video_capture = cv.VideoCapture(simple_CSI.gstreamer_pipeline(flip_method=0), cv.CAP_GSTREAMER)

#This should all be setup in host file, this is only here for the purpose of showing how setup should work, should remain commented out
#video_path = 'lanes/lanes_1.mp4'
#video_capture = cv.VideoCapture(video_path)

#values for IMX219 cameras on jetson nano (hand tuned)
#cameraMatrix = np.array([[470.0, 0.0, 206.0], [0.0, 193.0, 156.0], [0.0, 0.0, 1.0]])
#distortionCoeffs = np.array([[-0.141, 0.012, 0.0, 0.006, 0.018]])

#counter = 0
#timer = 300

#ORB stuff:
#max_features = 400  #max number of features, will stop if this value is hit
#scale_factor = 1.2  #default = 1.2
#nlevels = 8        #default = 8, num of pyramid levels
#first_level = 0     #default = 0, the pyramid level at which input image is  located, previous layers will be filled with upscaled source image
#edge_threshold = 31 #default = 31, should nearly match patchsize
#patch_size = 31    #default = 31, should nearly match edge threshold
#score_type = 'HARRIS_SCORE' #default value = 0 (HARRIS_SCORE), optional value =  FAST_SCORE which will be faster but less stable points
#fast_threshold = 20 #default = 20

#orb_1 = cv.ORB_create(nfeatures=max_features, 
#                    scaleFactor=scale_factor, 
#                    nlevels=nlevels, 
#                    edgeThreshold=edge_threshold, 
#                    firstLevel=first_level, 
#                    patchSize=patch_size,
#                    fastThreshold=fast_threshold)

#flann stuff:
#FLANN_INDEX_LSH = 6
#index_params = dict(algorithm=FLANN_INDEX_LSH, table_number =6, key_size=12, multi_probe_level=1)
#search_params = dict(checks=50)


def get_ORB(frame, ORB_ID, drawMarker=False, distortionFix=None):

    #convert to gray, saves cost
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #distortion:
    if distortionFix:
        frame = cv.undistort(frame, distortionFix[0], distortionFix[1], None, distortionFix[0])

    #Compute
    keyp, des = ORB_ID.detectAndCompute(frame, None) #this is faster than orb.detect(), orb.compute()
     
    #Draw the orb markers (only for visualizing, does not affect algorithm)
    if drawMarker == True:
        frame = cv.drawKeypoints(frame, keyp, None, color=(255, 0, 255), flags=0)
    
        return frame, keyp, des

    else: return keyp, des

  
def get_FLANN(keyp_cur, des_cur, keyp_prev, des_prev, flan_ID, threshold=0.7):

    tentatives = []
    ratios = []
    probabilities = []
    prev_match_keyp = []
    cur_matched_keyp = []
    #only run flann if minimum num keypt for orientation reached
    if len(keyp_cur) > 6 and len(keyp_prev) > 6:
        matches = flan_ID.knnMatch(des_cur, des_prev, k=2)
   
        #make sure we have enough matches, then build list of best matches
        try:
            for m, n in matches:
                match = True       #Debuggin
                if m.distance < threshold * n.distance:                
                    tentatives.append(m)
                    ratios.append(m.distance / n.distance)

            sorted_indices = np.argsort(ratios)
            tentatives = list(np.array(tentatives)[sorted_indices])
            print('tent', tentatives[0])
            
           
            cur_matched_keyp = np.float32([ keyp_cur[m.queryIdx].pt for m in tentatives ])
            prev_matched_keyp = np.float32([ keyp_prev[m.trainIdx].pt for m in tentatives ])

            # matches_test =  [keyp_prev[m.queryIdx].pt,
            
            # Since the correspondences are assumed to be ordered by their SNN ratio a priori,
            # we just assign a probability according to their order.
            for i in range(len(tentatives)):
                probabilities.append(1.0 - i / len(tentatives))


            #q1 = np.float32([keypA[m.queryIdx].pt for m in good_matches])
            #q2 = np.float32([keypB[m.trainIdx].pt for m in good_matches])

            #matched_keypA = [keypA[m.queryIdx] for m in good_matchesA]
            #matched_keypB = [keypA[n.queryIdx] for n in good_matchesB]
            
            
            return tentatives, ratios, probabilities, prev_matched_keyp, cur_matched_keyp 

        except ValueError:
            match = False          #Debuggin
            print("match list insufficient")
            return None, None, None, None, None
            pass


def draw_matches_hconcat(prev_frame, prev_matched_keyp, cur_frame, cur_matched_keyp, match_limit=None):
    
    #draws matches between two images and retuns the horizontally concatinated output with match lines
    image1 = prev_frame.copy()
    image2 = cur_frame.copy()

    out = cv.hconcat([image1, image2])

    if match_limit is None: match_limit = len(cur_matched_keyp)
    
    for i in range(0, match_limit):
        
        # Coordinates of a point on t frame
        p1 = (int(prev_matched_keyp[i][0]), int(prev_matched_keyp[i][1]))
        # Coordinates of the same point on t+1 frame
        p2 = (int(cur_matched_keyp[i][0] + image1.shape[1]), int(cur_matched_keyp[i][1]))          

        #generate random colors
        color = (list(np.random.choice(range(256), size=3)))  
        color =[int(color[0]), int(color[1]), int(color[2])]     

        cv.circle(out, p1, 5, color, 1)
        cv.line(out, p1, p2, color, 1)
        cv.circle(out, p2, 5, color, 1)

    return out

def draw_matches_flow(prev_frame, prev_matched_keyp, cur_frame, cur_matched_keyp, match_limit=None, window_length=2, window=None):

    out = cur_frame.copy()
    #window is the list of previous frame lines
    #window length is how many frames until color fade is black

    if match_limit is None: match_limit = len(cur_matched_keyp)

    #line points, color: line[0] = [(x1, y1), (x2, y2), (b, g, r)]
    lines = []
    
    for i in range(0, match_limit):
        
        # Coordinates of a point on t frame
        p1 = (int(prev_matched_keyp[i][0]), int(prev_matched_keyp[i][1]))
        # Coordinates of the same point on t+1 frame
        p2 = (int(cur_matched_keyp[i][0]), int(cur_matched_keyp[i][1]))          

        #generate random colors:
        #color = (list(np.random.choice(range(256), size=3)))  
        #color = (int(color[0]), int(color[1]), int(color[2])) 

        #or just static color:   
        color = (255, 0, 255)

        lines.append([p1, p2, color])
        print('line:', lines[i]) 
    window.insert(0, lines)

    if len(window) > window_length:
        window.pop(-1)  #remove last item   

    for i in range(len(window)):
        intensity =(window_length - i)/ window_length
        
        for j in range(0, match_limit):
            try:
                b, g, r = window[i][j][2]
                color = (int(b*intensity), int(g*intensity), int(r*intensity))
                #this is a tuned section, this is only drawing lines that are short enough for a specific scenario, remove this for general use
                x1, y1 = window[i][j][0]
                x2, y2 = window[i][j][1]

                if np.sqrt((int((x2-x1)**2) + int((y2-y1)**2))) < 50:
                    cv.line(out, window[i][j][0], window[i][j][1], color=color, thickness=2)

                #uncomment this to run with out specific tuning:
                #cv.line(out, window[i][j][0], window[i][j][1], color=window[i][j][2], thickness=2)
            except:
                pass
            
    return out, window


def get_probabilities(tentatives):
    probabilities = []
    for i in range(len(tentatives)):
                probabilities.append(1.0 - i / len(tentatives))

    return probabilities

def verify_pymagsac_fundam(kps1, kps2, tentatives, use_magsac_plus_plus, h1, w1, h2, w2, sampler_id):
    correspondences = np.float32([ (kps1[m.queryIdx].pt + kps2[m.trainIdx].pt) for m in tentatives ]).reshape(-1,4)
    probabilities = []
    
    # NG-RANSAC and AR-Sampler require an inlier probability to be provided for each point.
    # Since deep learning-based prediction is not applied here, we calculate the probabilities
    # from the SNN ratio ranks.  
    if sampler_id == 3 or sampler_id == 4:
        probabilities = get_probabilities(tentatives)

    F, mask = pymagsac.findFundamentalMatrix(
        np.ascontiguousarray(correspondences), 
        w1, h1, w2, h2,
        probabilities = probabilities,
        sampler = sampler_id,
        use_magsac_plus_plus = use_magsac_plus_plus,
        sigma_th = 1.0)
    print (deepcopy(mask).astype(np.float32).sum(), 'inliers found')
    return F, mask



