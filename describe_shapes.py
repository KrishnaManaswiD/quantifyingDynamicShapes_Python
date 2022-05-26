## 
# Title: Describe shapes in terms of elliptic Fourier descriptors
# Author: Krishna Manaswi Digumarti
# Version: 1.0 (adapted from 3.0 of MATLAB code)
# Date: May 2022
#
# I would appreciate it if you cite the following paper when using this code 
# Digumarti KM, Trimmer B, Conn AT, Rossiter J. Quantifying Dynamic Shapes in Soft Morphologies. Soft Robotics. 2019 Dec 1;6(6):733-44.
#
# Based on: 
# Kuhl FP, Giardina CR. Elliptic Fourier features of a closed contour. Computer graphics and image processing. 1982 Mar 1;18(3):236-58.

import os
import cv2

## some settable properties
nHarmonics = 6      # number of harmonics to use for estimation
nSynthesis = 100    # num of pts to reconstruct contour from estimate

shouldNormalize = 0 # 1 = yes / 0 = no: use 0 for visualization, 1 for pca
shouldVisualize = 1 # 1 = yes / 0 = no: to visualised estimated contour
# Note: only normalised coefficients are used in pca. 
# To speed up processing, avoid visualisation when computing coefficients 
# for further analysis. Set shouldNormalize to 1 and shouldVisualize to 0

contourColor = (0, 0, 255) # B, G, R
lineWidth = 2

# specify folder to read image frames from
folderForFrames = 'frames'
recordingName = 'movie'
subFolder = folderForFrames + '/' + recordingName

# figure out how many frames there are in total
numberOfFrames = len(os.listdir(subFolder))

## compute chain code and harmonic coeffieints for each frame
coeffs_Mat = [] # a variable that stores the Fourier coefficients

for frameNum in range(1,numberOfFrames+1):
    # read image and binarize it
    img = cv2.imread(os.path.join(subFolder, "frame{:d}.png".format(frameNum)))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh, img_binary = cv2.threshold(img_gray, 128, 255, cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)

    # if there are multiple contours choose the longest one - assumption is that smaller ones are from noise or islands
    contourLength = 0
    longestContour = []
    for cnt in contours:
        if len(cnt) > contourLength:
            contourLength = len(cnt)
            longestContour = cnt

    cv2.drawContours(img, [longestContour], 0, contourColor, lineWidth)

    cv2.imshow('img', img)
    cv2.waitKey(0)

    # assign the Freeman chain code
    chainCode = []
    for i in range(len(longestContour)-1):
        dx = longestContour[i+1][0][0] - longestContour[i][0][0]
        dy = longestContour[i+1][0][1] - longestContour[i][0][1]

        if dx == 1 and dy == 0:
            chainCode.append(0)
        if dx == 1 and dy == 1:
            chainCode.append(1)
        if dx == 0 and dy == 1:
            chainCode.append(2)
        if dx == -1 and dy == 1:
            chainCode.append(3)
        if dx == -1 and dy == 0:
            chainCode.append(4)
        if dx == -1 and dy == -1:
            chainCode.append(5)
        if dx == 0 and dy == -1:
            chainCode.append(6)
        if dx == 1 and dy == -1:
            chainCode.append(7)

#     # get harmonic coefficients
#     coefficients = fourier_approx(transpose(chainCode), nHarmonics, shouldNormalize);

#     A0 = coefficients(1,1);
#     C0 = coefficients(1,3);
#     a = coefficients(2:end,1);
#     b = coefficients(2:end,2);
#     c = coefficients(2:end,3);
#     d = coefficients(2:end,4);
    
#     # collect coefficients from each frame in a matrix
#     coeffs = reshape(coefficients', [1, size(coefficients,1)*size(coefficients,2)]);
#     coeffs_Mat = [coeffs_Mat; coeffs];
    
#     # Optional - visualize the estimated contour
#     if shouldVisualize
#         # synthesize the estimated contour
#         coordinates = zeros(nSynthesis,2);
#         for j = 1 : nSynthesis
#             x_ = 0.0;
#             y_ = 0.0;

#             for i = 1 : nHarmonics
#                 x_ = x_ + (a(i) * cos(2 * i * pi * j / nSynthesis) + b(i) * sin(2 * i * pi * j / nSynthesis));
#                 y_ = y_ + (c(i) * cos(2 * i * pi * j / nSynthesis) + d(i) * sin(2 * i * pi * j / nSynthesis));
#             end

#             coordinates(j,1) = A0 + x_;
#             coordinates(j,2) = C0 + y_;
#         end

#         # correct location
#         coordinates = coordinates + [startCol*ones(size(x_,1),1) startRow*ones(size(x_,1),1)];

#         # Make it closed contour
#         contour = [coordinates; coordinates(1,1) coordinates(1,2)];

#         # draw the synthesized contour
#         figure(1)
#         plot(contour(:,1), contour(:,2), 'color', color, 'linewidth', lineWidth);
#         drawnow
#     end

# # save coefficients - Note: only normalized coefficients are used in pca
# if shouldNormalize
#     save('coefficients.mat','coeffs_Mat')
# end