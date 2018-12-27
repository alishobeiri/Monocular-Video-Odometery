import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

cap = cv.VideoCapture('./videos/test_bronx_trim_1.mp4')

file_path = 'C:\\Users\\Ali\\Desktop\\Projects\\mono-vo\\data_odometry_gray\\dataset\\sequences\\00\\image_0\\'
annot_path = 'C:\\Users\\Ali\\Desktop\\Projects\\mono-vo\\data_odometry_poses\\dataset\\poses\\00.txt'

with open(annot_path) as f:
    pose = f.readlines()

def detect(img):
    p0 = fast.detect(img)
    p0 = [x.pt for x in p0]
    
    return np.array(p0, dtype=np.float32).reshape(-1, 1, 2)


def get_absolute_scale(id):
    ss = pose[id - 1].strip().split()
    x_prev = float(ss[3])
    y_prev = float(ss[7])
    z_prev = float(ss[11])
    ss = pose[id].strip().split()
    x = float(ss[3])
    y = float(ss[7])
    z = float(ss[11])

    true_vect = np.array([[x], [y], [z]])
    prev_vect = np.array([[x_prev], [y_prev], [z_prev]])
    
    return np.linalg.norm(true_vect - prev_vect)


focal = 718.8560
pp = (607.1928, 185.2157)
R_total = np.zeros((3, 3))
t_total = np.empty(shape=(3, 1))

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (21,21),
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

fast = cv.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)


# Create some random colors
color = np.random.randint(0,255,(5000,3))
# Take first frame and find corners in it
old_gray = cv.imread(file_path +str().zfill(6)+'.png', 0)
p0 = detect(old_gray)
frame = cv.imread(file_path + str(1).zfill(6)+'.png', 0)
frame_gray = frame.copy()


p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

good_new = p1[st==1]
good_old = p0[st==1]

E, _ = cv.findEssentialMat(good_new, good_old, focal, pp, cv.RANSAC, 0.999, 1.0, None)

_, R_total, t_total, _ = cv.recoverPose(E, good_old, good_new, R_total, t_total, focal, pp, None)


# color = np.random.randint(0,255,(100,3))
# print(color)


good_old = good_new 

n_features = p0.shape[0]
traj = np.zeros(shape=(600, 600, 3))

true_total = np.array([400, 500])
mask = np.zeros_like(old_gray)
count = 2
flag = True
for img_id in range(2, len(os.listdir(file_path))):
    if n_features < 1000:
        p0 = detect(old_gray)
        # p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    frame = cv.imread(file_path + str(img_id).zfill(6)+'.png', 0)
    frame_gray = frame.copy()
    
    # frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    print("p0.shape: ", p0.shape)
    print("old_gray.shape: ", old_gray.shape)
    print("frame_gray.shape: ", frame_gray.shape)

    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    
    # draw the tracks
    color = np.random.randint(0,255,(good_old.shape[0],3))
    p_mask = np.zeros_like(good_new)
    for i, (new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()

        if np.linalg.norm(new - old) < 10:
            p_mask[i] = 1

            if flag:
                mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
    
    print(good_old.shape)
    # Only use good points that don't move too much, ensure points are stable
    # print("p0: ", p0)
    # good_new = good_new[p_mask == 1].reshape(-1, 2)
    # good_old = good_old[p_mask == 1].reshape(-1, 2)
    E, _ = cv.findEssentialMat(good_new, good_old, focal, pp, cv.RANSAC, 0.999, 1.0, None)
    # print("E: ", E)

    _, R, t, _ = cv.recoverPose(E, good_old, good_new, R_total.copy(), t_total.copy(), focal, pp, None)
    # print("R: ", R)
    print("t: ", t)
    img = cv.add(frame, mask)
    cv.imshow('frame', img)

    k = cv.waitKey(1)
    
    absolute_scale = get_absolute_scale(img_id)
    print("absolute_scale: ", absolute_scale)
    if absolute_scale > 0.1:
        print("t before: ", t_total)
        t_total = t_total + absolute_scale*R_total.dot(t)
        print("t after: ", t_total)
        R_total = R.dot(R_total)
        print("R_total: ", R_total)
    

    draw_x, draw_y = int(round(t_total[0][0]) + 400), int(round(t_total[2][0]) + 500)


    ss = pose[img_id].strip().split()

    true_x, true_y = int(round(-float(ss[3])) + 400), int(round(-float(ss[11])) + 500)
    true_total[0] += true_x
    true_total[1] += true_y
    print("True x: ", true_x)
    print("True z: ", true_y)
    # if img_id == 4:
    #     exit()
    traj = cv.circle(traj, (draw_x, draw_y), 1, list((255, 255, 255)), 1)
    traj = cv.circle(traj, (true_x, true_y), 1, list((0, 0, 255)), 1)
    # cv.circle(traj, (true_x,true_y), 1, (0,0,255), 2)



    print("True total: ", true_x, true_y)
    # first = (int(round(avg[0])), int(round(avg[1])))
    # second = (int(round(avg[0] + t[0][0])), int(round(avg[1] + t[1][0])))
    # t_mask = cv.line(t_mask, first, second, color[i].tolist(), 2)

    # avg[0] += t[0][0]
    # avg[1] += t[1][0]
    # avg[2] += t[2][0]

    # # t_mask = cv.line(t_mask, (0,0),(500,500), color[i].tolist(), 2)
    # traj = cv.add(traj, t_mask)
    cv.imshow('traj', traj)
    if k == 27:
        break

    if k == 121:
        flag = not flag
        toggle_out = lambda flag: "On" if flag else "Off"
        print("Flow lines turned ", toggle_out(flag))
        mask = np.zeros_like(frame)

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    n_features = p0.shape[0]
    count += 1
cv.destroyAllWindows()
