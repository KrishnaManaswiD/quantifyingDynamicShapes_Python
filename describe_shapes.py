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

from json.tool import main
import os
from turtle import st
from typing import List
import cv2
import numpy
import math



def calc_harmonic_coefficients(chainCode, nHarmonics):

    # This function returns the n-th set of four harmonic coefficients.
    # Input: 
    #   chain code (chainCode)
    #   number of harmonics (nHarmonics)
    # Output:
    #   coeffients of harmonics [an bn cn dn]

    k = len(chainCode) # length of chain code
    
    l = calc_traversal_length(chainCode) # traversal length for each link
        
    L = l[k-1] # basic period = traversal length for entire chain (perimeter)
    
    # Store this value to make computation faster
    two_n_pi = 2 * nHarmonics * math.pi
    
    ## Compute Harmonic cofficients: an, bn, cn, dn
    sigma_a = 0
    sigma_b = 0
    sigma_c = 0
    sigma_d = 0
        
    for p in range(0,k):
        if (p > 0):
            lp_prev = l[p - 1]          
        else:
            lp_prev = 0
        
        delta_d = calc_change_in_projections([chainCode[p]])
        delta_x = delta_d[0][0]
        delta_y = delta_d[0][1]
        delta_l = calc_traversal_length([chainCode[p]])[0]
        
        q_x = delta_x / delta_l
        q_y = delta_y / delta_l
        
        sigma_a = sigma_a + q_x * (math.cos(two_n_pi * l[p] / L) - math.cos(two_n_pi * lp_prev / L))
        sigma_b = sigma_b + q_x * (math.sin(two_n_pi * l[p] / L) - math.sin(two_n_pi * lp_prev / L))
        sigma_c = sigma_c + q_y * (math.cos(two_n_pi * l[p] / L) - math.cos(two_n_pi * lp_prev / L))
        sigma_d = sigma_d + q_y * (math.sin(two_n_pi * l[p] / L) - math.sin(two_n_pi * lp_prev / L))   
    
    r = L/(2 * pow(nHarmonics,2) * pow(math.pi,2))
    
    an = r * sigma_a
    bn = r * sigma_b
    cn = r * sigma_c
    dn = r * sigma_d
    
    ## Assign  to output
    return [an, bn, cn, dn]


def calc_traversal_length(chainCode):

    # The length of each link is either 1 or sqrt(2) depending on the orientation of the link.
    #
    # Input: chain code (chainCode)
    # Output:
    #   if size(chainCode) is 1, returns length of that link
    #   if size(chainCode) > 1, ith element is accumulated length up to link i. 

    sum_deltaLength = 0
    l = [0] * len(chainCode)
    for i in range(0,len(chainCode)):
        sum_deltaLength = sum_deltaLength + 1 + ((math.sqrt(2) - 1)/2) * (1 - (-1)**chainCode[i])
        l[i] = sum_deltaLength
    
    return l


def calc_change_in_projections(chainCode):

    # returns the changes in x and y projections of the chain, as a link in the chain code is traversed
    #
    # input: chain code (chainCode)
    # output: 
    #    if size(chainCode) is 1, [Dx, Dy]
    #    if size(chainCode) > 1, ith row of ouput is [Sum_1toi(Dx), Sum_1toi(Dy)]

    sum_Dx = 0
    sum_Dy = 0
    
    p = [[0,0] for _ in range(len(chainCode))]
    
    for i in range(0, len(chainCode)):
        sum_Dx = sum_Dx + numpy.sign(6 - chainCode[i]) * numpy.sign(2 - chainCode[i])
        sum_Dy = sum_Dy + numpy.sign(4 - chainCode[i]) * numpy.sign(chainCode[i])
        p[i][0] = sum_Dx
        p[i][1] = sum_Dy
    
    return p


def calc_dc_components(chainCode):

    # Calculate DC components.
    # Input: 
    #   chain code (chain code)
    # Output: 
    #   A0 and C0 are bias coefficeis, corresponding to a frequency of zero.

    ## Maximum length of chain code
    k = len(chainCode)
    
    ## Traversal length and change in projection length
    l = calc_traversal_length(chainCode)
    s = calc_change_in_projections(chainCode)
    
    ## Basic period of the chain code
    L = l[k-1]
    
    ## DC Components: A0, C0
    sum_a0 = 0
    sum_c0 = 0
    
    for p in range(0,k):
        delta_d = calc_change_in_projections([chainCode[p]])
        delta_x = delta_d[0][0]
        delta_y = delta_d[0][1]
        delta_l = calc_traversal_length([chainCode[p]])[0]

        if (p > 0):
            zeta = s[p - 1][0] - delta_x / delta_l * l[p - 1]
            delta = s[p - 1][1] - delta_y / delta_l * l[p - 1]
        else:
            zeta = 0
            delta = 0
        

        if (p > 0):
            sum_a0 = sum_a0 + delta_x / (2 * delta_l) * ((l[p])**2 - (l[p-1])**2) + zeta * (l[p] - l[p-1])
            sum_c0 = sum_c0 + delta_y / (2 * delta_l) * ((l[p])**2 - (l[p-1])**2) + delta * (l[p] - l[p-1])
        else:
            sum_a0 = sum_a0 + delta_x / (2 * delta_l) * (l[p])**2 + zeta * l[p]
            sum_c0 = sum_c0 + delta_y / (2 * delta_l) * (l[p])**2 + delta * l[p]
    
    ## Assign  to output
    A0 = sum_a0 / L
    C0 = sum_c0 / L

    return A0, C0


def fourier_approx(chainCode, nHarmonics, shouldNormalize):

    # This function generates coefficients of fourier approximation, given a chain code.
    # Input: 
    #   chain code (chainCode)
    #   number of harmonics (nHarmonics)
    #   whether to normalise or not (shouldNormalize)
    # Output:
    #   coeffients of harmonics n+1x4 matrix
    #   first row is [A0 0 C0 0]

    a = [0] * nHarmonics
    b = [0] * nHarmonics
    c = [0] * nHarmonics
    d = [0] * nHarmonics

    for i in range(1,nHarmonics+1):       # loop over each harmonic
        harmonic_coeff = calc_harmonic_coefficients(chainCode, i)
        a[i-1] = harmonic_coeff[0]
        b[i-1] = harmonic_coeff[1]
        c[i-1] = harmonic_coeff[2]
        d[i-1] = harmonic_coeff[3]

    A0, C0 = calc_dc_components(chainCode) # bias components corresponding to zero frequency

    # Normalization procedure
    if shouldNormalize == 1:
        # Remove DC components
        A0 = 0
        C0 = 0
        
        # Compute theta1
        theta1 = 0.5 * math.atan(2 * (a[0] * b[0] + c[0] * d[0]) / (a[0]**2 + c[0]**2 - b[0]**2 - d[0]**2))
       
        costh1 = math.cos(theta1)
        sinth1 = math.sin(theta1)
             	 
        a_star_1 = costh1 * a[0] + sinth1 * b[0]
        b_star_1 = -sinth1 * b[0] + costh1 * b[0]
        c_star_1 = costh1 * c[0] + sinth1 * d[0]
        d_star_1 = -sinth1 * c[0] + costh1 * d[0]
       
        # Compute psi1 
        psi1 = math.atan(c_star_1 / a_star_1)
        
        # Compute E
        E = math.sqrt(a_star_1**2 + c_star_1**2)
        
        cospsi1 = math.cos(psi1)
        sinpsi1 = math.sin(psi1)
        
        for i in range(0,nHarmonics):
            m1 = [[cospsi1, sinpsi1], [-sinpsi1, cospsi1]]
            m2 = [[a[i], b[i]], [c[i], d[i]]] 
            m3 = [[math.cos(theta1 * i), -math.sin(theta1 * i)], [math.sin(theta1 * i), math.cos(theta1 * i)]]
            product = numpy.dot(m1, m2)
            product = numpy.dot(product, m3)
            normalised = product.tolist()

            a[i] = normalised[0][0] / E
            b[i] = normalised[0][1] / E
            c[i] = normalised[1][0] / E
            d[i] = normalised[1][1] / E
        
    coefficients = [[A0, 0, C0, 0]]
    for ai,bi,ci,di in zip(a,b,c,d):
        coefficients.append([ai, bi, ci, di])

    return coefficients


if __name__=="__main__":

    ## some settable properties
    nHarmonics = 3      # number of harmonics to use for estimation
    nSynthesis = 100    # num of pts to reconstruct contour from estimate

    shouldNormalize = 1 # 1 = yes / 0 = no: use 0 for visualization, 1 for pca
    shouldVisualize = 0 # 1 = yes / 0 = no: to visualised estimated contour
    shouldNormalize = 0 if shouldVisualize == 1 else shouldNormalize
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
                longestContour = numpy.squeeze(cnt, axis=1)

        startingPoint = longestContour[0]

        # assign the Freeman chain code
        chainCode = []
        for i in range(len(longestContour)-1):
            dx = longestContour[i+1][0] - longestContour[i][0]
            dy = longestContour[i+1][1] - longestContour[i][1]

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

        
        # get harmonic coefficients
        coefficients = fourier_approx(chainCode, nHarmonics, shouldNormalize)
        
        # collect the coefficients from each frame for saving
        coeffs_Mat.append(coefficients)

        # Optional - visualize the estimated contour

        if shouldVisualize == 1:
            A0 = coefficients[0][0]
            C0 = coefficients[0][2]
            a = [i[0] for i in coefficients]
            a.pop(0) # 1:end, 0
            b = [i[1] for i in coefficients]
            b.pop(0) # 1:end, 1
            c = [i[2] for i in coefficients]
            c.pop(0) # 1:end, 2
            d = [i[3] for i in coefficients]
            d.pop(0) # 1:end, 3

            # synthesize the estimated contour
            coordinates = [[0,0] for _ in range(nSynthesis)]
            for j in range(0,nSynthesis):
                x_ = 0.0
                y_ = 0.0

                for i in range(0,nHarmonics):
                    x_ = x_ + (a[i] * math.cos(2 * (i+1) * math.pi * (j+1) / nSynthesis) + b[i] * math.sin(2 * (i+1) * math.pi * (j+1) / nSynthesis))
                    y_ = y_ + (c[i] * math.cos(2 * (i+1) * math.pi * (j+1) / nSynthesis) + d[i] * math.sin(2 * (i+1) * math.pi * (j+1) / nSynthesis))

                coordinates[j][0] = A0 + x_
                coordinates[j][1] = C0 + y_

            # correct location
            for i,c in enumerate(coordinates):
                coordinates[i][0] += startingPoint[0]
                coordinates[i][1] += startingPoint[1]
            
            # Make it closed contour
            estimatedContour = numpy.array(coordinates)
            numpy.append(estimatedContour, [coordinates[0][0],coordinates[0][1]])

            # draw the synthesized contour
            if shouldVisualize == 1:
                cv2.drawContours(img, [estimatedContour.astype(int)], 0, contourColor, lineWidth)
                cv2.imshow('img', img)
                cv2.waitKey(0)

    # save coefficients - Note: only normalized coefficients are used in pca
    if shouldNormalize:
        with open("coefficients.txt", "w") as f:
            for frame in coeffs_Mat:
                for harmonic in frame:
                    for coefficient in harmonic:
                        f.write((str(coefficient) + " " ))
                    f.write("\n")
                f.write("\n\n")
