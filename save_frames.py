## 
# Title: Extract frames from a video and save them as png
# Author: Krishna Manaswi Digumarti
# Version: 1.0 (adapted from 3.0 of MATLAB code)
# Date: May 2022
#
# I would appreciate it if you cite the following paper when using this code 
# Digumarti KM, Trimmer B, Conn AT, Rossiter J. 
# "Quantifying Dynamic Shapes in Soft Morphologies."
# Soft Robotics. 6(6), pp.733-744. 2019

import os
import cv2

## construct subfolder to save frames from movie
folderForFrames = 'frames'
recordingName = 'movie'           

## create sub folder
subFolder = folderForFrames + '/' + recordingName

if not os.path.exists(subFolder):
    os.makedirs(subFolder)

## save frames from a video
v = cv2.VideoCapture(recordingName + '.mp4')
frameNum = 0

while v.isOpened():
    ret, frame = v.read()

    if ret == True:
        frameNum += 1
        cv2.imwrite(os.path.join(subfolder + '/' + frameNum), frame)
    else:
        break
end

v.release()
cv2.destroyAllWindows()