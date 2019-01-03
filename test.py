import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from monovideoodometery import MonoVideoOdometery
import os


img_path = 'C:\\Users\\Ali\\Desktop\\Projects\\SLAM\\videos\\data_odometry_gray\\dataset\\sequences\\00\\image_0\\'
pose_path = 'C:\\Users\\Ali\\Desktop\\Projects\\SLAM\\videos\\data_odometry_poses\\dataset\\poses\\00.txt'

focal = 718.8560
pp = (607.1928, 185.2157)
R_total = np.zeros((3, 3))
t_total = np.empty(shape=(3, 1))

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (21,21),
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))


# Create some random colors
color = np.random.randint(0,255,(5000,3))

vo = MonoVideoOdometery(img_path, pose_path, focal, pp, lk_params)
traj = np.zeros(shape=(600, 800, 3))

# mask = np.zeros_like(vo.current_frame)
# flag = False
while(vo.hasNextFrame()):

    frame = vo.current_frame

    # for i, (new,old) in enumerate(zip(vo.good_new, vo.good_old)):
    #     a,b = new.ravel()    
    #     c,d = old.ravel()
        
    #     if np.linalg.norm(new - old) < 10:
    #         if flag:
    #             mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    #             frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)


    # cv.add(frame, mask)
    cv.imshow('frame', frame)
    k = cv.waitKey(1)
    if k == 27:
        break

    if k == 121:
        flag = not flag
        toggle_out = lambda flag: "On" if flag else "Off"
        print("Flow lines turned ", toggle_out(flag))
        mask = np.zeros_like(vo.old_frame)
        mask = np.zeros_like(vo.current_frame)

    vo.process_frame()

    print(vo.get_mono_coordinates())

    mono_coord = vo.get_mono_coordinates()
    true_coord = vo.get_true_coordinates()

    print("MSE Error: ", np.linalg.norm(mono_coord - true_coord))
    print("x: {}, y: {}, z: {}".format(*[str(pt) for pt in mono_coord]))
    print("true_x: {}, true_y: {}, true_z: {}".format(*[str(pt) for pt in true_coord]))

    draw_x, draw_y, draw_z = [int(round(x)) for x in mono_coord]
    true_x, true_y, true_z = [int(round(x)) for x in true_coord]

    traj = cv.circle(traj, (true_x + 400, true_z + 100), 1, list((0, 0, 255)), 4)
    traj = cv.circle(traj, (draw_x + 400, draw_z + 100), 1, list((0, 255, 0)), 4)

    cv.putText(traj, 'Actual Position:', (140, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1)
    cv.putText(traj, 'Red', (270, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 1)
    cv.putText(traj, 'Estimated Odometry Position:', (30, 120), cv.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1)
    cv.putText(traj, 'Green', (270, 120), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 1)

    cv.imshow('trajectory', traj)
cv.imwrite("./images/trajectory.png", traj)

cv.destroyAllWindows()