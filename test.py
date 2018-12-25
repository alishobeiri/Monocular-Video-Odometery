import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


cap = cv.VideoCapture('./videos/test_bronx_trim_2.mp4')
fast = cv.FastFeatureDetector_create(threshold=150)


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 4,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes

n_features = p0.shape[0]

mask = np.zeros_like(old_frame)
count = 0
flag = True
while(1):
    if n_features < 50:
        p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    ret,frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    
    # draw the tracks
    p_mask = np.zeros_like(good_new)
    for i, (new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()

        if np.linalg.norm(new - old) < 10:
            p_mask[i] = 1

            if flag:
                mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
    
    # Only use good points that don't move too much, ensure points are stable
    good_new = good_new[p_mask == 1]

    img = cv.add(frame, mask)
    cv.imshow('frame',img)
    k = cv.waitKey(1)
    
    if k == 27:
        break

    if k == 121:
        flag = not flag
        toggle_out = lambda flag: "On" if flag else "Off"
        print("Flow lines turned ", toggle_out(flag))
        mask = np.zeros_like(old_frame)

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    n_features = p0.shape[0]
    count += 1
cv.destroyAllWindows()
