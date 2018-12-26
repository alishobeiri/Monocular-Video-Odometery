import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


cap = cv.VideoCapture('./videos/test_countryroad.mp4')


focal = 718.8560
pp = (607.1928, 185.2157)
R = np.zeros((3, 3))
t = np.zeros((3, 3))
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
avg = np.array([old_frame.shape[0]//2, old_frame.shape[1]//2], dtype=np.float32)
mask = np.zeros_like(old_frame)
traj = np.zeros_like(old_frame)
t_mask = np.zeros_like(traj)
count = 1
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
    

    dom_vector = np.array([0.0, 0.0])
    # draw the tracks
    p_mask = np.zeros_like(good_new)
    for i, (new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()

        if np.linalg.norm(new - old) < 10:
            p_mask[i] = 1

            dom_vector[0] += (a - c)
            dom_vector[1] += (b - d)
            if flag:
                mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)

    dom_vector = dom_vector/np.linalg.norm(good_new - good_old)
    print(dom_vector)

    first = (int(round(avg[0])), int(round(avg[1])))
    second = (int(round(avg[0] + dom_vector[0])), int(round(avg[1] + dom_vector[1])))
    t_mask = cv.line(t_mask, first, second, color[i].tolist(), 2)

    traj = cv.circle(traj, (1000, 1000),5,color[i].tolist(),-1)

    # t_mask = cv.line(t_mask, (0,0),(500,500), color[i].tolist(), 2)
    traj = cv.add(traj, t_mask)

    avg[0] += dom_vector[0]
    avg[1] += dom_vector[1]
    
    print(avg)

    # Only use good points that don't move too much, ensure points are stable
    # print("p0: ", p0)
    # good_new = good_new[p_mask == 1].reshape(-1, 2)
    # good_old = good_old[p_mask == 1].reshape(-1, 2)
    E, _ = cv.findEssentialMat(good_new, good_old, focal, pp, cv.RANSAC, 0.999, 1.0, mask)
    # print("E: ", E)

    _, R, t, _ = cv.recoverPose(E, good_old, good_new, R, t, focal, pp, mask)
    print("R: ", R)
    print("t: ", t)
    img = cv.add(frame, mask)
    # cv.imshow('frame', img)
    # cv.imshow('traj', traj)
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
