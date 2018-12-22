import cv2
import os


def vid_to_frames(vid_path, out_prefix, out_path, file_type=".jpg"):    
    """ This method is used to generate frames from a video of 
        any length. 
    
    Arguments:
        vid_path {str} -- [Path to video]
        out_prefix {str} -- [Prefix for each video frame]
        out_path {str} -- [Path for saved frames]
    
    Keyword Arguments:
        file_type {str} -- [The image type to save frames as] (default: {".jpg"})
    
    Raises:
        ValueError -- [Raised when specified paths are incorrect or do not exist]
    """

    # if not os.path.exists(vid_path):
    #     raise ValueError("The video path does not exist")

    # if not os.path.exists(out_path):
    #     raise ValueError("The output path does not exist")

    file_name = out_prefix + "_frame"
    if len(out_prefix) == 0:
        file_name = "frame"

    vidcap = cv2.VideoCapture(vid_path)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(out_path + file_name + "_" + str(count) + file_type, image)    
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
    
    