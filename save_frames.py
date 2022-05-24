## 
# Title: Extract frames from a video and save them as png
# Author: Krishna Manaswi Digumarti
# Version: 1.0 (adapted from 3.0 of MATLAB code)
# Date: May 2022
#
# I would appreciate it if you cite the following paper when using this code 
# Digumarti KM, Trimmer B, Conn AT, Rossiter J. Quantifying Dynamic Shapes in Soft Morphologies. Soft Robotics. 2019 Dec 1;6(6):733-44.

import os
import cv2

## construct subfolder to save frames from movie
folderForFrames = 'frames'
recordingName = 'movie'         # set this to the name of the video file
subFolder = folderForFrames + '/' + recordingName

## create sub folder if it does not exist
if not os.path.exists(subFolder):
    os.makedirs(subFolder)

## save frames from a video
v = cv2.VideoCapture(recordingName + '.mp4')
frameNum = 0

while v.isOpened():
    success, frame = v.read()
    
    if success:
        frameNum += 1
        cv2.imwrite(os.path.join(subFolder, "frame{:d}.png".format(frameNum)), frame)
    else:
        break

v.release()
cv2.destroyAllWindows()