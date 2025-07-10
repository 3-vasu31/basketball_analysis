import cv2
import os

def read_video(video_path):
    cap=cv2.VideoCapture(video_path)
    frames=[]
    while True:
        ret,frame=cap.read() # it returns frame and ret: refers if something is returned or not.
        if not ret:
            break

        frames.append(frame)

    return frames

def save_video(output_video_frames,output_video_path):
    if not os.path.exists(os.path.dirname(output_video_path)):
        os.mkdir(os.path.dirname(output_video_path))

    fourcc=cv2.VideoWriter_fourcc(*"XVID")
    #24 fps
    #(output_video_frames[0].shape[1],...) is the width and height of the video
    out=cv2.VideoWriter(output_video_path,fourcc,24.0,(output_video_frames[0].shape[1],output_video_frames[0].shape[0]))

    #loop over each frame and write it out
    for  frame in output_video_frames:
        out.write(frame)
    out.release()