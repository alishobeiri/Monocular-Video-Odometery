from utils import video_edit
import cv2


video_edit.vid_to_frames("./Calgary/IMG_2694.mp4", "test", 
                        "./Calgary/Frames/")


img = cv2.imread("./Calgary/Frames/test_frame_1")

cv2.imshow('image', img)