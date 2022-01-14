#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 10:39:21 2021

@author: yoonahshin
"""
import finger_analysis as fa
import numpy as np
import pandas as pd
import os
import glob
import datetime
import matplotlib.pyplot as plt
# Plot settings
#plt.rcParams.update({'font.size': 18}) 
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams["font.family"] = 'Arial'

def create_filename_list(base_name, img_num_list, zeropad=3):
    """
    Creates a list of file names for a given 
    base name and a list of image numbers.

    Parameters
    ----------
    base_name : str
        Base name of image list.
    img_num_list : list
        List of image numbers.
    zeropad : int, optional
        Number of zeros pad to a string. The default is 3.

    Returns
    -------
    List of file names. TYPE 'list'

    """
    filename_list=[] # Create an empty list to store list of file names
    for i in range(len(img_num_list)):
        filename_list.append(base_name+str(img_num_list[i]).zfill(zeropad))
    return filename_list

def create_filename_list_using_wildcard(global_name, base_name='',\
                                        img_num_list=[], zeropad=3,\
                                        if_remove_ele=False):
    """
    Create a list of file names without extension.
    If you want to selectively remove certain elements in the list,
    you can do it so by providing base_name and img_num_list that needs to be removed.
    ----------
    global_name : str
        global name of files using wildcard, such as *,?.
        # '???' matches exactly three characters in length
        example: '20deg_0??.tif'
    
    Returns
    -------
    list of file names without extension. TYPE 'list'

    """
    filename_list = glob.glob(global_name)
    filename_list.sort()
    filename_w_o_ext = [] # list of file names without extension
    i=0
    for name in filename_list:
        filename_w_o_ext.append(os.path.splitext(name)[0])
        i+=1
    if if_remove_ele==True:
        unwanted_ele = create_filename_list(base_name, img_num_list, zeropad)
    else:
        unwanted_ele = []
        
    return [ele for ele in filename_w_o_ext if ele not in unwanted_ele]

def batch_get_propagation_direction(base_name, img_num_list, zeropad=3,\
                                    threshold=100, minLineLength=100,\
                                    maxLineGap=5, data_bar_top=690,\
                                    show_image=False, save_image=False,\
                                    save_df_indiv=False, ori_l=0, ori_u=50,\
                                    save_df=False, h=120, alpha=33, t=6,\
                                    l='2_p1', dir_name='propagation_direction'):
    # 1. create file name list
    filename_list = create_filename_list(base_name, img_num_list, zeropad)
    # 2. batch process of calculating finger proagation direction
    concat_df = [] # Create an empty list to store concatenated data frames
    for name in filename_list:
        df = fa.line_orientation.get_finger_orientation(name, threshold, minLineLength,\
                                                        maxLineGap, data_bar_top,\
                                                        show_image, save_image, save_df_indiv)
        df = df[df>ori_l].dropna()
        df = df[df<ori_u].dropna()
        concat_df.append(df) # create list of data frames to be concatenated
    df_combined = pd.concat(concat_df, keys=filename_list)
    # 3. save the result
    if save_df==True:
        # saving the dataframe to an excel file
        output_name = '{}nm_{}deg_{}h_{}'.format(h,alpha,t,l)
        suffix_name = ''
        fa.output.make_dir_and_output_df_to_excel(dir_name, df_combined,\
                                                  output_name, suffix_name)
        # saving the local variables to a text file
        text_file = open("{}/{}_variables.txt".format(dir_name, output_name), "w")
        text_file.write(str(locals().items())) # Amazing function to store all the local variables in a local scope.
        text_file.close()
        # Append a text (date) to a file in Python
        file_object = open('{}/{}_variables.txt'.format(dir_name, output_name), 'a') # Open a file with access mode 'a'
        file_object.write(str(datetime.datetime.now())) # Append 'datetime of now' at the end of file
        file_object.close() # Close the file
   
    return df_combined

def batch_get_propagation_distance(base_name, img_num_list, zeropad=3,\
                                   suffix_1='_line_1', suffix_2='_line_2',\
                                   line_color='r', data_bar_top=690,\
                                   show_overlay=False, pix_size_given=True,\
                                   y_min=713, y_max=750, x_min=5, x_max=190,\
                                   y_min_bar=730, y_max_bar=760, x_min_bar=5,\
                                   x_max_bar=190, e_w=1, s_w=1, threshold=25,\
                                   minLineLength=25, maxLineGap=10, show_img=False,\
                                   show_edge=False, show_scale_bar=False,\
                                   save_df=False, h=120, alph=33, t=6, l='2_p1',\
                                   dir_name='propagation_distance'):
    # 1. create file name list
    filename_list = create_filename_list(base_name, img_num_list, zeropad)
    # 2. batch process of calculating finger proagation distance
    concat_df = [] # Create an empty list to store concatenated data frames
    for name in filename_list:
        edges = fa.fingers.get_edges_in_img(name, data_bar_top)
        line_1 = fa.fingers.get_line_drawn_in_img(name, suffix_1,\
                                                     line_color, data_bar_top)
        line_2 = fa.fingers.get_line_drawn_in_img(name, suffix_2,\
                                                     line_color, data_bar_top)
        sorted_df_init = fa.fingers.get_coords_intersections(line_1, line_2,\
                                                                show_overlay)
        sorted_df_tips = fa.fingers.get_coords_intersections(edges, line_2,\
                                                                show_overlay)
        if pix_size_given == True:
            pix_size = fa.scale.get_pixel_size(name)[0]/1000
        if pix_size_given == False:
            pix_size = fa.scale.extract_pixel_size(name, y_min, y_max, x_min,\
                                                      x_max, y_min_bar, y_max_bar,\
                                                      x_min_bar, x_max_bar, e_w, s_w,\
                                                      threshold, minLineLength, maxLineGap,\
                                                      show_img, show_edge, show_scale_bar)[2]
        distance = fa.fingers.propagation_distance_of_fingers(sorted_df_init,\
                                                                  sorted_df_tips, pix_size)
        concat_df.append(distance)
    df_combined = pd.concat(concat_df, keys=filename_list)
    # 3. save the result
    if save_df==True:
        # saving the dataframe to an excel file
        output_name = '{}nm_{}deg_{}h_{}'.format(h,alph,t,l)
        suffix_name = ''
        fa.output.make_dir_and_output_df_to_excel(dir_name, df_combined,\
                                                  output_name, suffix_name)
        # saving the local variables to a text file
        text_file = open("{}/{}_variables.txt".format(dir_name, output_name), "w")
        text_file.write(str(locals().items())) # Amazing function to store all the local variables in a local scope.
        text_file.close()
        # Append a text (date) to a file in Python
        file_object = open('{}/{}_variables.txt'.format(dir_name, output_name), 'a') # Open a file with access mode 'a'
        file_object.write(str(datetime.datetime.now())) # Append 'datetime of now' at the end of file
        file_object.close() # Close the file
    
    return df_combined  

def batch_get_a_b_p(base_name, img_num_list, zeropad=3,\
                    threshold_alpha=100, minLineLength_alpha=100, maxLineGap_alpha=5,\
                    threshold_beta=100, minLineLength_beta=100, maxLineGap_beta=10,\
                    data_bar_top=690, show_image=False, ori_l=0, ori_u=50,\
                    suffix='_line1', line_color='r', show_overlay=False,\
                    pix_size_given=True, y_min=713, y_max=750, x_min=5,\
                    x_max=190, y_min_bar=730, y_max_bar=760, x_min_bar=5,\
                    x_max_bar=190, e_w=1, s_w=1, threshold_s=25,\
                    minLineLength_s=25, maxLineGap_s=10, show_img=False,\
                    show_edge=False, show_scale_bar=False, save_df=False,\
                    h=120, alph=33, t=6, l='2_p1', dir_name_a='wire_width',\
                    dir_name_b='finger_width', dir_name_p='finger_period'):
    
    # 1. create file name list
    filename_list = create_filename_list(base_name, img_num_list, zeropad)
    
    # 2. batch process of calculating a, b, and p, where a is wire width,
    # b is finger width, and p is finger period measured along the red or black line.
    # Then, compute average finger propagation direction, beta,
    # and compute a_p, b_p, and p_p, where a_p and b_p are the actual widths 
    # of wires and fingers (i.e. perpendicular to the wires) and p_p is the finger period
    # perpendicular to the wire arrays
    concat_df_a = []
    concat_df_b = []
    concat_df_p = []
    for name in filename_list:
        # 1.calculate alpha (i.e. initial edge orientation w.r.t. x-axis)
        if line_color=='r':
            df_alpha = fa.line_orientation.get_red_line_orientation(name, suffix,\
                                                                    threshold_alpha,\
                                                                    minLineLength_alpha,\
                                                                    maxLineGap_alpha,\
                                                                    data_bar_top, show_image)
            alpha = df_alpha['Red line orientation (deg)'].mean() 
        if line_color=='k':
            df_alpha = fa.line_orientation.get_black_line_orientation(name, suffix,\
                                                                      threshold_alpha,\
                                                                      minLineLength_alpha,\
                                                                      maxLineGap_alpha,\
                                                                      data_bar_top, show_image)
            alpha = df_alpha['Black line orientation (deg)'].mean()
        # 2. calculate beta (i.e. wire orientation w.r.t. x-axis)
        df_beta = fa.line_orientation.get_finger_orientation(name, threshold_beta,\
                                                             minLineLength_beta,\
                                                             maxLineGap_beta,\
                                                             data_bar_top,\
                                                             show_image)
        df_beta = df_beta[df_beta>ori_l].dropna()
        df_beta = df_beta[df_beta<ori_u].dropna()
        beta = df_beta['Finger orientation (deg)'].mean()
        # 3. calculate wire width, finger width, and finger period along alpha
        edges = fa.fingers.get_edges_in_img(name, data_bar_top)
        line = fa.fingers.get_line_drawn_in_img(name, suffix, line_color, data_bar_top)
        sorted_df = fa.fingers.get_coords_intersections(edges, line, show_overlay)
        if pix_size_given == True:
            pix_size = fa.scale.get_pixel_size(name)[0]/1000
        if pix_size_given == False:
            pix_size = fa.scale.extract_pixel_size(name, y_min, y_max, x_min,\
                                                      x_max, y_min_bar, y_max_bar,\
                                                      x_min_bar, x_max_bar, e_w, s_w,\
                                                      threshold_s, minLineLength_s, maxLineGap_s,\
                                                      show_img, show_edge, show_scale_bar)[2]
        a = fa.fingers.get_wire_widths_along_line(sorted_df, pix_size)
        a.rename(columns={'Wire width (\u03BCm)':'Wire width along alpha (\u03BCm)'}, inplace=True)
        b = fa.fingers.get_finger_widths_along_line(sorted_df, pix_size)
        b.rename(columns={'Finger width (\u03BCm)':'Finger width along alpha (\u03BCm)'}, inplace=True)
        p = fa.fingers.get_finger_periods_along_line(sorted_df, pix_size)
        p.rename(columns={'Finger period (\u03BCm)':'Finger period along alpha (\u03BCm)'}, inplace=True)
        # 4. calculate wire width, finger width, and finger period perpendicular to wires
        m = np.sin(np.deg2rad(abs(alpha)+abs(beta)))
        a_p = a*m
        a_p.rename(columns={'Wire width along alpha (\u03BCm)':'Wire width (\u03BCm)'}, inplace=True)
        b_p = b*m
        b_p.rename(columns={'Finger width along alpha (\u03BCm)':'Finger width (\u03BCm)'}, inplace=True)
        p_p = p*m
        p_p.rename(columns={'Finger period along alpha (\u03BCm)':'Finger period (\u03BCm)'}, inplace=True)
        # 5. concatenate the dataframes side by side
        df_a = pd.concat([a, a_p], axis=1)
        df_b = pd.concat([b, b_p], axis=1)
        df_p = pd.concat([p, p_p], axis=1)
        # 6. append the concatenated dataframe to the empty list of 'concat_df'
        concat_df_a.append(df_a)
        concat_df_b.append(df_b)
        concat_df_p.append(df_p)
    df_combined_a = pd.concat(concat_df_a, keys=filename_list)
    df_combined_b = pd.concat(concat_df_b, keys=filename_list)
    df_combined_p = pd.concat(concat_df_p, keys=filename_list)
    
    # 3. save the result
    if save_df==True:
        # saving the dataframe to an excel file
        output_name = '{}nm_{}deg_{}h_{}{}'.format(h,alph,t,l,suffix)
        fa.output.make_dir_and_output_df_to_excel(dir_name_a, df_combined_a,\
                                                  output_name, '')
        fa.output.make_dir_and_output_df_to_excel(dir_name_b, df_combined_b,\
                                                  output_name, '')
        fa.output.make_dir_and_output_df_to_excel(dir_name_p, df_combined_p,\
                                                  output_name, '')
        # saving the local variables to a text file
        text_file = open("{}/{}_variables.txt".format(dir_name_a, output_name), "w")
        text_file.write(str(locals().items())) # Amazing function to store all the local variables in a local scope.
        text_file.close()
        
        text_file_2 = open("{}/{}_variables.txt".format(dir_name_b, output_name), "w")
        text_file_2.write(str(locals().items())) # Amazing function to store all the local variables in a local scope.
        text_file_2.close()
        
        text_file_3 = open("{}/{}_variables.txt".format(dir_name_p, output_name), "w")
        text_file_3.write(str(locals().items())) # Amazing function to store all the local variables in a local scope.
        text_file_3.close()
    
    return (df_combined_a, df_combined_b, df_combined_p)

def check_get_pixel_size(filename_list, data_bar_min=712, data_bar_max=760):
    """
    check if the function 'get_pix_size' correctly reads the pixel size and 
    the prefix for the given list of file names. 

    Parameters
    ----------
    filename_list : list
        list of file names. The file names should not have extension. 
    data_bar_min : int, optional
        The y coordinate of the top of data zone.
        The default is 712 for MIT CMSE SEM image, while 690 for Harvard CNS. 
    data_bar_max : int, optional
        The y coordinate of the bottom of data zone. The default is 760.

    Returns
    -------
    pandas data frame with columns consisting of 'filename', 'pixel size',
    and 'prefix'. 

    """
    df_name = []
    df_pix_size = []
    df_prefix = []
    for name in filename_list:
        df_name.append(name)
        pix_size, prefix = fa.scale.get_pixel_size(name, data_bar_min, data_bar_max)
        df_pix_size.append(pix_size)
        df_prefix.append(prefix)
    df_name = pd.DataFrame(df_name, columns=['file name'])
    df_pix_size = pd.DataFrame(df_pix_size, columns=['pixel size'])
    df_prefix = pd.DataFrame(df_prefix, columns=['prefix'])
    df_result = pd.concat([df_name, df_pix_size, df_prefix], axis=1)
    
    return df_result

def check_extract_pixel_size(filename_list, y_min=713, y_max=750, x_min=5, x_max=190,\
                            y_min_bar=730, y_max_bar=760, x_min_bar=5, x_max_bar=190,\
                            e_w=1, s_w=1, threshold=25, minLineLength=25, maxLineGap=10,\
                            show_img=False, show_edge=False, show_scale_bar=False):
    df_name = []
    df_text = []
    df_num_pix = []
    df_pix_size = []
    for name in filename_list:
        df_name.append(name)
        text, num_pix, pix_size = fa.scale.extract_pixel_size(name, y_min, y_max, x_min, x_max,\
                                      y_min_bar, y_max_bar, x_min_bar, x_max_bar,\
                                      e_w, s_w, threshold, minLineLength, maxLineGap,\
                                      show_img, show_edge, show_scale_bar)
        df_text.append(text)
        df_num_pix.append(num_pix)
        df_pix_size.append(pix_size)
    df_name = pd.DataFrame(df_name, columns=['File name'])
    df_text = pd.DataFrame(df_text, columns=['Number above the scale bar'])
    df_num_pix = pd.DataFrame(df_num_pix, columns=['Number of pixels in the scale bar'])
    df_pix_size = pd.DataFrame(df_pix_size, columns=['Pixel size'])
    df_result = pd.concat([df_name, df_text, df_num_pix, df_pix_size], axis=1)
    
    return df_result

def batch_new_method_propagation_distance(base_name, img_num_list, l='2_p1', zeropad=3,\
                                          suffix_1='new_line_1', suffix_2='new_line_2',\
                                          threshold_alpha=100, minLineLength_alpha=100,\
                                          maxLineGap_alpha=5, threshold_beta=100,\
                                          minLineLength_beta=100, maxLineGap_beta=10,\
                                          data_bar_top=690, show_image=False, ori_l=0,\
                                          ori_u=50, line_color='r', show_overlay=False,\
                                          pix_size_given=True, y_min=713, y_max=750,\
                                          x_min=5, x_max=190, y_min_bar=730, y_max_bar=760,\
                                          x_min_bar=5, x_max_bar=190, e_w=1, s_w=1,\
                                          threshold_s=25, minLineLength_s=25, maxLineGap_s=10,\
                                          show_img=False, show_edge=False, show_scale_bar=False,\
                                          save_df=False, h=120, alph=33, t=6, reverse_sort=False,\
                                          dir_name='propagation_distance'):
    #1. Create file name list
    filename_list = create_filename_list(base_name, img_num_list, zeropad)
    #2. Calculate propagation distance of fingers using the new method
    concat_df = []
    for name in filename_list:
        # 1.calculate alpha (i.e. initial edge orientation w.r.t. x-axis)
        if line_color=='r':
            df_alpha = fa.line_orientation.get_red_line_orientation(name, suffix_1,\
                                                                    threshold_alpha,\
                                                                    minLineLength_alpha,\
                                                                    maxLineGap_alpha,\
                                                                    data_bar_top, show_image)
            alpha = df_alpha['Red line orientation (deg)'].mean() 
        if line_color=='k':
            df_alpha = fa.line_orientation.get_black_line_orientation(name, suffix_1,\
                                                                      threshold_alpha,\
                                                                      minLineLength_alpha,\
                                                                      maxLineGap_alpha,\
                                                                      data_bar_top, show_image)
            alpha = df_alpha['Black line orientation (deg)'].mean()
        # 2. calculate beta (i.e. wire orientation w.r.t. x-axis)
        df_beta = fa.line_orientation.get_finger_orientation(name, threshold_beta,\
                                                             minLineLength_beta,\
                                                             maxLineGap_beta,\
                                                             data_bar_top,\
                                                             show_image)
        df_beta = df_beta[df_beta>ori_l].dropna()
        df_beta = df_beta[df_beta<ori_u].dropna()
        beta = df_beta['Finger orientation (deg)'].mean()
        # 3. get pixel size
        if pix_size_given == True:
            pix_size = fa.scale.get_pixel_size(name)[0]/1000
        if pix_size_given == False:
            pix_size = fa.scale.extract_pixel_size(name, y_min, y_max, x_min,\
                                                      x_max, y_min_bar, y_max_bar,\
                                                      x_min_bar, x_max_bar, e_w, s_w,\
                                                      threshold_s, minLineLength_s, maxLineGap_s,\
                                                      show_img, show_edge, show_scale_bar)[2]
        # 4. calculate propgation distance of fingers
        line_1 = fa.fingers.get_line_drawn_in_img(name, suffix_1, line_color, data_bar_top)
        line_2 = fa.fingers.get_line_drawn_in_img(name, suffix_2, line_color, data_bar_top)
        sorted_df = fa.fingers.get_coords_intersections(line_1, line_2, show_overlay)
        if reverse_sort == True:
            sorted_df = sorted_df.iloc[::-1].reset_index(drop=True)
        n = len(sorted_df)
        length = np.zeros(n)
        for i in range(1,n):
            x1 = sorted_df.x.iloc[0]
            y1 = sorted_df.y.iloc[0]
            x2 = sorted_df.x.iloc[i]
            y2 = sorted_df.y.iloc[i]
            length[i] = np.sqrt((x2-x1)**2 + (y2-y1)**2)*pix_size
        length = pd.DataFrame(length, columns=['distance'])
        length.distance.iloc[0] = float('nan')
        m = np.sin(np.deg2rad(abs(alpha)+abs(beta)))
        d=length/m
        concat_df.append(d)
    df = pd.concat(concat_df, keys=filename_list)
    
   # 3. save the result
    if save_df==True:
        # saving the dataframe to an excel file
        output_name = '{}nm_{}deg_{}h_{}'.format(h,alph,t,l)
        suffix_name = ''
        fa.output.make_dir_and_output_df_to_excel(dir_name, df,\
                                                  output_name, suffix_name)
        # saving the local variables to a text file
        text_file = open("{}/{}_variables.txt".format(dir_name, output_name), "w")
        text_file.write(str(locals().items())) # Amazing function to store all the local variables in a local scope.
        text_file.close()
        # Append a text (date) to a file in Python
        file_object = open('{}/{}_variables.txt'.format(dir_name, output_name), 'a') # Open a file with access mode 'a'
        file_object.write(str(datetime.datetime.now())) # Append 'datetime of now' at the end of file
        file_object.close() # Close the file
    
    return df          