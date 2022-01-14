#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:21:22 2019

@author: yoonahshin
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import finger_analysis as fa

def detect_lines(filename, threshold, minLineLength, maxLineGap,\
                 data_bar_top=690, show_image=False, save_image=False): 
    """
    For a given SEM image, this function crops the data bar area and detects lines
    in the image using probabilistic Hough Transform (PHT) function in OpenCV.
    The PHT function returns endpoints of the detected lines (x1,y1,x2,y2).
    The detected lines can be drawn on the image with labels. PHT is an optimization
    of Hough Transform. It doesn't take all the points into consideration, but instead
    takes only a random subset of points which is sufficient for line detection.    
    """
    # Image pre-processing
    img = cv2.imread(filename+'.tif').copy()[0:data_bar_top,:,:] # crop data bar 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    blur = cv2.threshold(cv2.GaussianBlur(gray,(7,7),0),0,255,\
                         cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] # Gaussian blur
    edges = cv2.Canny(blur, 50, 200, apertureSize=5) # find edges in the image
    # Detect lines (outputs end points (x1,y1,x2,y2) of the detected lines)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=threshold,\
                            minLineLength=minLineLength, maxLineGap=maxLineGap)
    # Draw the detected lines on the image and put labels for the detected lines 
    n = len(lines) # number of detected lines
    for i in range(n):
        for x1,y1,x2,y2 in lines[i]:
            cv2.line(img, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(img, text=str(i), org=(x1,y1), fontFace=1,\
                        fontScale=2, color=(255,255,255), thickness=2)
    # Show image with the lines and labels drawn
    if show_image == True: 
        plt.figure()
        plt.imshow(img)
    # Save image with the lines and labels drawn
    if save_image == True: 
        fa.output.save_fig(directory_name='Finger_Direction_Images',\
                           save_name='{}_detected_lines_{}_{}_{}.tif'.format(filename,\
                                                                             threshold,\
                                                                             minLineLength,\
                                                                             maxLineGap))
    return lines

def compute_angles(lines, col_name):
    """
    Given a numpy array of lines, which consists of rows of end points of lines
    (x1,y1,x2,y2), this function calculates the angle of each of the lines from
    (x1,y1,x2,y2) with respect to the x-axis.   
    """
    # Calculate angles of lines 
    n = len(lines) # number of detected lines 
    df = np.zeros((n,1)) # initialization of data frame
    for i in range(n):
        x1 = lines[i][0][0]  
        y1 = lines[i][0][1]
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]
        angle= np.arctan((y2-y1)/(x2-x1))
        angle *= 180/np.pi # convert unit from rad to deg 
        df[i] = angle
    df = pd.DataFrame(df, columns=col_name)
    return df

def get_finger_orientation(filename, threshold, minLineLength, maxLineGap,\
                           data_bar_top=690, show_image=False, save_image=False,\
                           save_df=False):
    """
    For a given SEM image, this function detects lines in the image and computes
    the orientation of the detected lines in degrees with respect to the x-axis.
    The number of lines detected will vary while you change the three parameters:
    threshold, minLineLength, and maxLineGap. 
    """
    lines = detect_lines(filename, threshold, minLineLength,\
                         maxLineGap, data_bar_top, show_image, save_image)
    df = compute_angles(lines, col_name=['Finger orientation (deg)'])
    if save_df == True:
        excel_name = '{}_{}_{}_{}'.format(filename, threshold, minLineLength, maxLineGap)
        fa.output.make_dir_and_output_df_to_excel('Finger_Direction', df, excel_name, '')    
    return df
    
def get_red_line_orientation(filename, suffix, threshold_l, minLineLength_l,\
                             maxLineGap_l, data_bar_top=690, img_show=False):
    """
    For a given SEM image with a red line drawn, this function returns
    orientation of the red line in degrees with respect to the x-axis. 
    The counterclockwise rotation is positive.
    """
    img = cv2.imread(filename+suffix+'.tif').copy()[0:data_bar_top,:,:]
    red_line = np.multiply(img, img==[0,0,255]) # get red line in the image
    red_line_edges = cv2.Canny(red_line, 100, 200, apertureSize=7)    
    lines = cv2.HoughLinesP(red_line_edges, rho=1, theta=np.pi/180,\
                            threshold=threshold_l, minLineLength=minLineLength_l,\
                            maxLineGap=maxLineGap_l)
    # Draw detected lines on the image and put label for each of them 
    n = len(lines) # number of detected lines
    for i in range(n):
        for x1,y1,x2,y2 in lines[i]:
            cv2.line(img, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(img, text=str(i), org=(x1,y1), fontFace=1,\
                        fontScale=2, color=(0,0,0), thickness=2)
    # Show image with the lines and labels drawn
    if img_show == True: 
        plt.figure()
        plt.imshow(img)
    # Compute angles of the lines 
    df = compute_angles(lines, col_name=['Red line orientation (deg)'])
    return df

def get_black_line_orientation(filename, suffix, threshold_l, minLineLength_l,\
                               maxLineGap_l, data_bar_top=690, show_image=False):
    """
    For a given SEM image with a black line drawn, this function returns
    orientation of the black line in degrees with respect to the x-axis. 
    The counterclockwise rotation is positive.
    """
    img = cv2.imread(filename+suffix+'.tif').copy()[0:data_bar_top,:,:]
    is_black = img.copy()
    line = np.multiply(is_black==[0,0,0], np.ones(is_black.shape))
    line_copy = np.uint8(line)
    edge = cv2.Canny(line_copy,0,0)
    lines = cv2.HoughLinesP(edge, rho=1, theta=np.pi/180, threshold=threshold_l,\
                            minLineLength=minLineLength_l, maxLineGap=maxLineGap_l)
    # Draw detected lines on the image and put labels 
    n = len(lines)
    for i in range(n):
        for x1,y1,x2,y2 in lines[i]:
            cv2.line(img, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(img, text=str(i), org=(x1,y1),\
                        fontFace=1, fontScale=2, color=(0,0,0), thickness=2)
    # Show image with labels
    if show_image == True:
        plt.figure()
        plt.imshow(img)
    # Compute angles of the lines
    df = compute_angles(lines, col_name=['Black line orientation (deg)'])
    return df 