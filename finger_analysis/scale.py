#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 00:31:40 2021

@author: yoonahshin
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
import re
pytesseract.pytesseract.tesseract_cmd = r'/usr/local/Cellar/tesseract/4.1.0/bin/tesseract'
# One can find full path to their tesseract executable by typing command as 
# written below on terminal on Mac OS:
# brew list tesseract

# The function "get_pixel_size()" is copied from Maxwell A. L'Etoile's 
# get_pixel_size() in scale.py. 
# Here, I added comments that describe each line of the code. 

def get_pixel_size(filename, data_bar_min=690, data_bar_max=760):
    """
    returns the stated pixel size in the data bar.
    To be specific, this function returns a tuple of
    pixel_size and the prefix of the unit, for example: 
    (97.85, n)
    """
    full_img = cv2.imread(filename+'.tif',0) # read in image file as a grayscale 
    data_bar = full_img.copy()[data_bar_min:data_bar_max,:] # take the data bar area from the original image
    
    # convert the data_bar(type: ndarray) to text(type: str)
    # Since data_bar is an numpy array, use Image.fromarray() function
    # to read in the data_bar image 
    bar_text = pytesseract.image_to_string(Image.fromarray(data_bar))
    # remove all the spacings in the text, and then convert to lower-case letters
    bar_text = bar_text.replace(' ','').lower()
    # find index of starting point of the word 'pixelsize'
    p_loc = bar_text.find('pixelsize')
    # text length of the 'pixelsize' is 9. 
    l = 9
    # find relative index of the letter 'm' from the index of starting point of 'pixelsize'
    m_rel_loc = bar_text[p_loc:].find('m')
    # find the prefix of unit; if the unit is nm, then prefix will be 'n'
    # while, if the unit is um, then the prefix will be 'u'. 
    # Note: 'u' is 'micro' symbol. 
    prefix = bar_text[p_loc + m_rel_loc - 1]
    # Get the 'str' of pixel size and convert its type to 'float'
    pixel_size = float(bar_text[p_loc+l+1:p_loc+m_rel_loc-1]) 
    
    return (pixel_size, prefix)

def read_number_above_scale_bar(filename, y_min=713, y_max=750,\
                                x_min=0, x_max=190, show_img=False):
    """
    Parameters
    ----------
    filename : str
        Image file name without extension. 
    y_min : int, optional
        y coordinate of the top of data bar. The default is 713.
    y_max : int, optional
        y coordinate of the bottom of data bar. The default is 750.
    x_min : int, optional
        x coordinate of the start of data bar. The default is 0.
    x_max : int, optional
        x coordinate of the end data bar. The default is 190.
    show_img : boolean, optional
        if True, the function shows scale bar.
    
    Returns 
    -------
    Number on the scale bar. TYPE 'int'
    
    """
    img = cv2.imread(filename+'.tif',0) # Read image in grayscale
    data_bar = img.copy()[y_min:y_max,x_min:x_max]
    if show_img == True:
        plt.imshow(data_bar)
    # Read text on the scale bar using pytesseract    
    bar_text = pytesseract.image_to_string(data_bar,\
                                           config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789')
    bar_text = re.findall(r'\d+', bar_text) # extracts numbers from str
    bar_text = int(bar_text[0])
    
    return bar_text
    
def get_number_of_pixels_in_scale_bar(filename, y_min_bar=730, y_max_bar=760,\
                                      x_min_bar=5, x_max_bar=190, e_w=1, s_w=1,\
                                      threshold=25, minLineLength=25, maxLineGap=10,\
                                      show_edge=False, show_scale_bar=False):
    """
    Parameters
    ----------
    filename : str
        Image file name without extension. 
        example: '20deg_001'
    y_min_bar : int, optional
        y coordinate of top of the scale bar area. The default is 730.
    y_max_bar : int, optional
        y coordinate of bottom of the scale bar area. The default is 760.
    x_min_bar : int, optional
        x coordinate of start of the scale bar area. The default is 5.
    x_max : int, optional
        x coordinate of end of the scale bar area. The default is 190.
    e_w : int, optional
        pixel number of the edge. It was found that the value is 1. 
    s_w: int, optional
        pixel number of the width of scale bar. It was found that the value is 1. 
    threshold : int, optional
        argument needed for detecting lines in the data bar area. 
    minLineLength : int, optional
        another argument needed for detecting lines in the data bar area.
    maxLineGap : int, optional
        another argument needed for detecting lines in the data bar area.
    show_edge : boolean, optional
        if True, the function shows edge image of the scale bar.
    show_scale_bar : boolean, optional
        if True, the function shows scale bar with lines drawn. 
        The end points of the lines give number of pixels of the scale bar.

    Returns 
    -------
    Number of pixels in the scale bar. TYPE 'int'
    
    """
    # Image processing
    img = cv2.imread(filename+'.tif',0)
    scale_bar = img.copy()[y_min_bar:y_max_bar,x_min_bar:x_max_bar]
    edges = cv2.Canny(scale_bar, threshold1=125, threshold2=255, apertureSize=5)
    
    # Show edge image of scale bar
    if show_edge == True: 
        plt.imshow(edges)
    
    # Probabilistic Hough Transform directly returns the two endpoints of lines
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,\
                            threshold=threshold, minLineLength=minLineLength,\
                            maxLineGap=maxLineGap)
    # Draw detected lines and get their pixel numbers
    n = len(lines) # number of detect lines from the edge image
    df = np.zeros(n) # dataframe for storing pixel numbers of the detected lines
    for i in range(n):
        for x1,y1,x2,y2 in lines[i]:
            pixel_nums = x2 - x1 -(2*e_w + s_w) 
            df[i] = pixel_nums
            cv2.line(scale_bar,(x1+e_w+int(s_w/2),y1),(x2-e_w-int(s_w/2),y2),(100,0,0),1) # draw the detected lines
    
    # Show scale bar
    if show_scale_bar == True:
        plt.figure()
        plt.imshow(scale_bar)
    
    idx = np.argmax(df)
    num_pixels_scale_bar = int(df[idx])
    
    return num_pixels_scale_bar

def extract_pixel_size(filename, y_min=713, y_max=750, x_min=5, x_max=190,\
                       y_min_bar=730, y_max_bar=760, x_min_bar=5, x_max_bar=190,\
                       e_w=1, s_w=1, threshold=25, minLineLength=25, maxLineGap=10,\
                       show_img=False, show_edge=False, show_scale_bar=False):
    """
    Reads number above the scale bar, gets number of pixels in the scale bar, 
    calculates the pixel size, and returns a tuple of 
    (number above the scale bar, number of pixels in the scale bar, pixel size)
    """
    # Read number on the scale bar
    text = read_number_above_scale_bar(filename, y_min, y_max, x_min, x_max, show_img)
    # Get number of pixels in the scale bar
    num_pix = get_number_of_pixels_in_scale_bar(filename, y_min_bar, y_max_bar,\
                                                x_min_bar, x_max_bar, e_w, s_w,\
                                                threshold, minLineLength, maxLineGap,\
                                                show_edge, show_scale_bar)
    pix_size = text/num_pix
    
    return (text, num_pix, pix_size)