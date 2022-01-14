#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 08:44:35 2021

@author: yoonahshin
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import measure
from skimage import color
from skimage import feature
import pandas as pd

def get_edges_in_img(filename, data_bar_top=690):
    """
    Parameters
    ----------
    filename : str
        File name of image without extension. For example, if the file name was
        '33deg_029.tif', then filename='33deg_029'.
    data_bar_top : int, optional
        y coordinate of top of data bar area. The default is 690.

    Returns
    -------
    Processed image - edges in the oringinal image

    """
    img = cv2.imread(filename+'.tif',0).copy()[:data_bar_top,:] # Read image in grayscale and crop the data bar area
    blur = cv2.GaussianBlur(img,(7,7),0) # Apply Gaussian Filtering to denoise the image
    img_bin = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] # Get a binarized image
    edges = feature.canny(img_bin) # Use the Canny filter to detect edges in the binarized image 
    return edges

def get_line_drawn_in_img(filename, suffix, line_color='r', data_bar_top=690):
    """
    Parameters
    ----------
    filename : str
        File name of image without extension. For example, if the file name was
        '33deg_029.tif', then filename='33deg_029'.
    suffix : str
        Suffix after the file name without extension. 
    line_color : str, optional
        Color of the line drawn in the image. The default is 'r'. 
        line_color='r' when the line is red.
        line_color='k' when the line is black.
    data_bar_top : int, optional
        y coordinate of top of data bar area. The default is 690.

    Returns
    -------
    Line in the image

    """
    if line_color == 'r':
        img_red_line = cv2.imread(filename+suffix+'.tif').copy()[:data_bar_top,:,2] # Read image with a red line drawn
        line = np.multiply(img_red_line==255, np.ones(img_red_line.shape)) # Image with only the red line
    elif line_color == 'k':
        img_black_line = cv2.imread(filename+suffix+'.tif').copy()[:data_bar_top,:,:]
        line = np.multiply(img_black_line==[0,0,0], np.ones(img_black_line.shape)) # Image with only the black line
        line = line[:,:,2] # The channel of line can be one of amongst [:,:,0], [:,:,1] and [:,:,2]
    return line

def get_coords_intersections(img_1, img_2, show_overlay=False):
    """
    Get regions where the two input images (i.e. ndarrays) cross, label the 
    regions, and returns a dataframe of sorted coordinates of the crossed regions.

    Parameters
    ----------
    img_1 : ndarray
        Input image 1.
    img_2 : ndarray
        Input image 2.
    show_overlay : bool, optional
        If True, shows the overlay of labels of crossed regions on the image.

    Returns
    -------
    A dataframe of sorted coordinates of the crossed regions

    """
    intersections = np.multiply(img_1, img_2)
    column_labels = ['label','x', 'y']
    label_image = measure.label(intersections)
    image_label_overlay = color.label2rgb(label_image, image=intersections)
    props = measure.regionprops(label_image)
    # Initialize the data frame for storing the coordinates of intersections
    coords = np.zeros([len(props),len(column_labels)])
    i = 0
    for prop in props:
        image_label_overlay = cv2.putText(img=np.copy(image_label_overlay),\
            text=str(prop.label), org=(int(prop.centroid[1]),\
            int(prop.centroid[0])), fontFace=2, fontScale=1,\
            color=(1,1,1), thickness=1)
        coords[i,:] = [prop.label, prop.centroid[1], prop.centroid[0]]
        i+=1
    image_df = pd.DataFrame(coords, columns=column_labels)    
    sorted_df = image_df.sort_values('y').reset_index(drop=True) 
    if show_overlay==True:
        plt.figure()
        plt.imshow(image_label_overlay)
    return sorted_df

def get_wire_widths_along_line(sorted_df, pixel_size):
    """
    Calculates width of wires along the line. 
    Returns a dataframe of the wire widths.
    """
    n = int(sorted_df.shape[0]/2)
    a_along_line = np.zeros(n)
    for i in range(n):
        x1 = sorted_df.x.iloc[2*i]
        x2 = sorted_df.x.iloc[2*i+1]
        y1 = sorted_df.y.iloc[2*i]
        y2 = sorted_df.y.iloc[2*i+1]
        a_along_line[i] = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    a_along_line *= pixel_size
    a_along_line = pd.DataFrame(a_along_line, columns=['Wire width (\u03BCm)'])
    return a_along_line

def get_finger_widths_along_line(sorted_df, pixel_size):
    """
    Calculates width of fingers along the line. 
    Returns a dataframe of the finger widths. 
    """
    n = int(sorted_df.shape[0]/2)-1
    b_along_line = np.zeros(n)
    for i in range(n):
        x1 = sorted_df.x.iloc[2*i+1]
        x2 = sorted_df.x.iloc[2*i+2]
        y1 = sorted_df.y.iloc[2*i+1]
        y2 = sorted_df.y.iloc[2*i+2]
        b_along_line[i] = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    b_along_line *= pixel_size 
    b_along_line = pd.DataFrame(b_along_line, columns=['Finger width (\u03BCm)'])
    return b_along_line

def get_finger_periods_along_line(sorted_df, pixel_size):
    """
    Calculates period of fingers along the line. 
    Returns a dataframe of the finger periods. 
    """
    n = int(sorted_df.shape[0]/2)-1
    p_along_line = np.zeros(n)
    for i in range(n):
        x1 = sorted_df.x.iloc[2*i]
        x2 = sorted_df.x.iloc[2*i+2]
        y1 = sorted_df.y.iloc[2*i]
        y2 = sorted_df.y.iloc[2*i+2]
        p_along_line[i] = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    p_along_line *= pixel_size 
    p_along_line = pd.DataFrame(p_along_line, columns=['Finger period (\u03BCm)'])
    return p_along_line

def propagation_distance_of_fingers(sorted_df_1, sorted_df_2, pixel_size):
    """
    Calculates propagation distances of fingers.
    Returns a dataframe of the propagation distances of fingers. 
    """
    n = sorted_df_1.shape[0]
    distance = np.zeros(n)
    for i in range(n):
        x1 = sorted_df_1.x.iloc[i]
        y1 = sorted_df_1.y.iloc[i]
        x2 = sorted_df_2.x.iloc[i]
        y2 = sorted_df_2.y.iloc[i]
        distance[i] = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    distance *= pixel_size 
    distance = pd.DataFrame(distance, columns=['Finger propagation distance (\u03BCm)'])
    return distance

def new_method_propagation_distance_of_fingers(alpha, beta, sorted_df, pixel_size):
    """
    Calculates propagation distances of fingers based on a new method.
    Returns a dataframe of the propagation distances of fingers.
    """
    n = len(sorted_df)
    q = np.zeros(n)
    for i in range(1,n):
        x1 = sorted_df.x.iloc[0]
        y1 = sorted_df.y.iloc[0]
        x2 = sorted_df.x.iloc[i]
        y2 = sorted_df.y.iloc[i]
        q[i] = np.sqrt((x2-x1)**2 + (y2-y1)**2)*pixel_size
    q = pd.DataFrame(q, columns=['distance'])
    q.distance.iloc[0] = float('nan')
    m = np.sin(np.deg2rad(abs(alpha)+abs(beta)))
    d = q/m
    return d  