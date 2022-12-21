import numpy as np
import cv2 as cv


def compute_left_disparity_map(img_left, img_right):
    
    # Parameters
    num_disparities = 5*16
    block_size = 11
    
    min_disparity = 0
    window_size = 6
    
    img_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
    img_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

    
    # Stereo BM matcher
    #left_matcher_BM = cv.StereoBM_create(
    #    numDisparities=num_disparities,
    #    blockSize=block_size
    #)

    ##################
    # Should look into left right consistency test, can be effective for cleaning up disparity map, but requires disparity
    # to be computed for right image as well so additional cost

    # Stereo SGBM matcher
    left_matcher_SGBM = cv.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Compute the left disparity map
    disp_left = left_matcher_SGBM.compute(img_left, img_right).astype(np.float32)/16
    
    return disp_left



def compute_depth_map(disparity_mapL, projection_matL, projection_matR):


    Kmat_L, Rmat_L, Trans, _, _, _, _ = cv.decomposeProjectionMatrix(projection_matL)
    Trans_L = Trans / Trans[3]

    Kmat_R, Rmat_R, Trans, _, _, _, _ = cv.decomposeProjectionMatrix(projection_matR)
    Trans_R = Trans / Trans[3]

    # Get the focal length from the K matrix
    f = Kmat_L[0, 0]
    #print('f from Kmat:', f)

    # Get the distance between the cameras from the t matrices (baseline)
    b = Trans_L[1] - Trans_R[1]

    # Replace all instances of 0 and -1 disparity with a small minimum value (to avoid div by 0 or negatives)
    disparity_mapL[disparity_mapL == 0] = 0.01
    disparity_mapL[disparity_mapL == -1] = 0.01

    # Initialize the depth map to match the size of the disparity map
    depth_map = np.ones(disparity_mapL.shape, np.single)

    # Calculate the depths 
    depth_map[:] = f * b / disparity_mapL[:]
    
    ### END CODE HERE ###
    
    return depth_map

def compute_depth_map2(disparity_mapL, projection_matL, projection_matR):


    Kmat_L, Rmat_L, Trans, _, _, _, _ = cv.decomposeProjectionMatrix(projection_matL)
    Trans_L = Trans / Trans[3]

    Kmat_R, Rmat_R, Trans, _, _, _, _ = cv.decomposeProjectionMatrix(projection_matR)
    Trans_R = Trans / Trans[3]

    # Get the focal length from the K matrix
    f = Kmat_L[0, 0]
    

    # Get the distance between the cameras from the t matrices (baseline)
    b = Trans_L[1] - Trans_R[1]
    print('f from Kmat:', f, 'b from transmat:', b)

    #Paul's forced params for Gryphon stereo dataset: b = 0.6
    #I think there is some issue with the formatting of the matricies, f is being pulled at 501mm, which seems like a very large focal length.
    #    because of this, I believe it is necessary to have the constrain 0 < b < 1, else we frequently exceed octaves of our 0-255 range
    #Maybe when it's time to implement this on my camera, I will not need this constraint, as the focal length of my IMX cameras is listed at 3mm
    b = 0.6

    # Replace all instances of 0 and -1 disparity with a small minimum value (to avoid div by 0 or negatives)
    disparity_mapL[disparity_mapL == 0] = 0.1
    disparity_mapL[disparity_mapL == -1] = 0.1

    # Initialize the depth map to match the size of the disparity map
    depth_map = np.ones(disparity_mapL.shape, np.uint8)

    # Calculate the depths 
    depth_map[:] = (f * b) / disparity_mapL[:]
    
    ### END CODE HERE ###
    
    return depth_map



#img_left = cv.imread('ball-left.png', 1)
#img_right = cv.imread('ball-right.png', 1)


# Compute the disparity map using the fuction above
#disp_left = compute_left_disparity_map(img_left, img_right)

#convert the disparity map to a format that OpenCV accepts and normalize it
#img_n = cv.normalize(src=disp_left, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

#cv.imshow('meh', img_n)
#cv.waitKey()
#cv.destroyAllWindows()


