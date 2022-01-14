#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 23:27:08 2019

@author: yoonahshin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import finger_analysis as fa

def save_plt_fig(DirectoryName, SaveName, ext='eps'):
    if not os.path.isdir(DirectoryName):
        os.makedirs(DirectoryName)
    path = os.path.join(DirectoryName, SaveName)
    plt.savefig(path, format=ext)
    return None

def draw_plot(leg, x, y, xlabel, ylabel):
    """
    leg: legend (e.g. 'Across the corner')
    x : numpy array of data of x axis
    y: numpy array of data of y axis
    xlabel: x label for x-axis
    ylabel: y label for y-axis
    """
    # Plot settings
    plt.rcParams.update({'font.size': 22}) # change the font size of values on the axes and that of legend
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.it'] = 'Arial:italic'
    plt.rcParams['mathtext.rm'] = 'Arial'
    plt.rcParams["font.family"] = 'Arial' 
    # Plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    ax.plot(x, y,'b-', lw=2, label=leg)
    # Using the specialized math font elsewhere, plus a different font
    ax.set_xlabel('{}'.format(xlabel),fontsize=32)
    ax.set_ylabel('{}'.format(ylabel),fontsize=32)
    # legend 
    plt.legend(frameon=False)
    ax.tick_params(direction='in', length=6, width=1, colors='k', \
                   top=True, bottom=True, left=True, right=True)
    plt.tight_layout()
    return fig

def average_height_read(filename_avg):
    """
    Read a .txt file that contains info about statistical measurements
    of given area (e.g. Average height value, RMS roughness, Min value,
    Max value, etc), pick up only the average height value and return it.
    The .txt file is obtained by the following steps:
    1. Open a raw AFM data file on Gwyddion 2.7.11
    2. Change the color mode to 'code-V' for visibility (up to your preference)
    3. Level the data by fitting a plane through three points (there is an icon for this)
    (Choose three points on the continuous film region)
    4. Shift minimum data value to zero (icon)
    5. Hit the icon called 'statistical quantities'
    6. Change the masking mode to 'Include only masked region' and click on 'Instant updates'
    and draw a box for region of interest (i.e. masked region), which is a finger region.
    7. Then, save the result to a .txt file using the following naming scheme:
    'flat_ProfileNumber_stat.txt' (e.g. 'flat_1_stat.txt')
        Input argument: .txt file; the file name must follow this naming
                scheme: 'flat_ProfileNumber_stat.txt' (e.g. 'flat_1_stat.txt')
        Return: average height value in the finger region (i.e. masked region)
    """
    # Picking up only the average height value from the .txt file 
    flat = pd.read_csv(filename_avg, sep='\tab', delimiter=' ', \
                       header=None, skiprows= 6, nrows=1, engine='python')[13]
    # Convert the pandas data object to numpy array for subsequent calculations 
    flat = flat.to_numpy()
    return flat
    
def corner(filename_corner, filename_avg, show_fig=False, save_fig=False):
    """
    Read a .txt file of profile across the corner, which was extracted 
    along the retraction direction of a finger and shift the zero of height 
    to the exposed substrate area (which is called a finger) by subtracting 
    the average height value of this finger. Return rim height at the corner.
    The file for 'filename_corner' is obtained by following procedure:
    1. Same steps (1-4) as that described in the function 'average_height_read'
    2. Hit the icon called 'Extract profiles' and draw a profile along the finger
    retraction direction, across the corner.
    3. Set the profile thickness to 1, extract the profile by hitting 'Apply'
    4. Then, right click on the extracted profile and save the result as a .txt file
    using the following naming scheme:
    'corner_ProfileNumber.txt' (e.g. 'corner_1.txt) 
        Input arguments: .txt file (as described above)
                         .txt file (i.e. file mentioned in the function average_height_read) 
        Return: rim height at the corner (the unit is 'nm')
    """
    data = pd.read_csv(filename_corner, sep='  ', header=None, \
                       skiprows=3, engine='python')
    x = data[0].to_numpy()*1e6 # distance along finger retraction direction (μm)
    y = data[1].to_numpy()*1e9 # height profile across the corner (nm)
    y = y - average_height_read(filename_avg) # shifting the zero of height to the exposed substrate
    rc = y.max() # rim height at the corner

    # Plot
    fig = draw_plot(leg='Across the corner', x=x, y=y,\
                    xlabel="$\mathrm{x}$ ($\mu$m)",\
                    ylabel=r"$\mathrm{Height}$ (nm)")
    
    if save_fig == True:
        directory_name = 'output'
        save_name = '{}.eps'.format(filename_corner[:-4])
        save_plt_fig(directory_name, save_name)
        
    if show_fig == False:
        plt.close(fig)
        
    return rc


def root(filename_root, filename_avg, show_fig=False, save_fig=False):
    """
    Read a .txt file of profile across the root, which was extracted along 
    in-plane normal of the side at the root. Likewise the function 'corner', this
    function also shifts the zero of height value to the exposed substrate area 
    due to dewetting. Then return rim height at the root. 
        Input arguments: filename_root, filename_avg
        Return: rim height at the root (nm)
    """
    data_root = pd.read_csv(filename_root, sep='  ', header=None, \
                            skiprows=3, engine='python')
    x_root = data_root[0].to_numpy()*1e6 # direction in-plane normal to side (μm)
    y_root = data_root[1].to_numpy()*1e9 # height profile across the root (nm)
    y_root = y_root - average_height_read(filename_avg)
    rm = y_root.max() # rim height at the root 
    
    # Plot
    fig = draw_plot(leg='Across the {}'.format(filename_root[:-6]),\
                    x=x_root, y=y_root, xlabel="$\mathrm{x}$ ($\mu$m)",\
                    ylabel=r"$\mathrm{Height}$ (nm)")
    
    if save_fig == True:
        directory_name = 'output'
        save_name = '{}.eps'.format(filename_root[:-4])
        save_plt_fig(directory_name, save_name)
    
    if show_fig == False:
        plt.close(fig)

    return rm


def side(filename_side, filename_avg, filename_corner, filename_root,\
         side_i, y_min, y_max, show_fig = False, save_fig = False):
    """
    Read a .txt file of profile along the side at the rim of the side. 
    Shift the zero value of height to the exposed substrate area (where the 
    region called 'finger'). From the original profile, identify the position 
    of corner and root and return plot that starts from the corner and ends 
    at the root. With this actual profile, the function also plots simplified
    linear rim height profile on the same plot (and save the plot). Also, 
    returns the results in a .xlsx file.  
        Input: 1) filename_side (str); the file name must follow this naming
                scheme: '{type_of_profile}_{number}.txt' (e.g. 'side1_1.txt') 
                2) filename_corner (str); (e.g. 'corner_1.txt')
                3) filename_root (str); (e.g. 'root1_1.txt')
                4) filename_avg (str); (e.g. 'flat_1_stat.txt')
                5) temp_period (str); template wavelength of the patch (e.g. '3.58')
                6) alpha (str); initial macroscopic edge orientation (e.g. '33deg')
                7) h (str); initial film thickness (e.g. '120nm')
                8) xlabel (str); label for x axis
                9) ylabel (str); label for y axis
                10) save_fig (bool); True or False (save or not)
                11) side_i (str); '1' or '2'; index of sides
                12) y_min (int); minimum of y-axis value
                13) y_max (int); maximum of y-axis value
        Return: Four values in a dataframe:
                1) rim height at the corner (nm)
                2) rim height at the root (nm)
                3) length of the side (μm)
                4) slope of the simplified linear profile along the side (rad) 
    """
    # read in the side profile
    data_side = pd.read_csv(filename_side, sep='  ', header=None, \
                            skiprows=3, engine='python')
    x_side = data_side[0].to_numpy()*1e6 # direction along the side, from corner to root (μm)
    y_side = data_side[1].to_numpy()*1e9 # height profile at the rim of side (nm)
    y_side = y_side - average_height_read(filename_avg) # subtract the average height in the finger region

    # rim height at the corner and root 
    rc = corner(filename_corner, filename_avg)
    rm = root(filename_root, filename_avg)
    
    # crop the profile so that the cropped profile starts from the corner and ends at the root.  
    idx1 = np.argmin(abs(y_side - rc))
    idx2 = np.argmin(abs(y_side - rm)) + 1 # +1 because when slicing array, the end index is not included.
    x_crop = x_side[idx1:idx2] - x_side[idx1] # crop the side profile from corner to root and shift the origin of x to the corner. 
    y_crop = y_side[idx1:idx2] # height is already shifted such that the finger region is zero. 
    side_length = x_crop[-1] # last element of x in the cropped profile is the length of side (μm)
    
    # Simplified linear profile 
    slope = (y_crop[-1] - y_crop[0])/(x_crop[-1])
    simp_profile = slope*(x_crop-x_crop[0]) + y_crop[0]

    # Plot settings
    plt.rcParams.update({'font.size': 22}) # change the font size of values on the axes and that of legend
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.it'] = 'Arial:italic'
    plt.rcParams['mathtext.rm'] = 'Arial'
    plt.rcParams["font.family"] = 'Arial'
    plt.rcParams["legend.loc"] = "upper left" # added this line on 2020/06/06
    #Plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    ax.plot(x_crop, y_crop,'b-', lw=6, label='Actual profile along the side {}'.format(side_i))
    ax.plot(x_crop, simp_profile,'k-.', label='Simplified profile')
    ax.set_ylim([y_min, y_max]) # added this line on 2020/06/06
    # Using the specialized math font elsewhere, plus a different font
    if filename_side[4] == '1':
        ax.set_xlabel('$\mathrm{l_1}$ ($\mu$m)', fontsize=32)
    else:
        ax.set_xlabel('$\mathrm{l_2}$ ($\mu$m)', fontsize=32)
    ax.set_ylabel('Height (nm)',fontsize=32)
    # legend 
    plt.legend(frameon=False)
    ax.tick_params(direction='in', length=6, width=1, colors='k', \
                   top=True, bottom=True, left=True, right=True)
    plt.tight_layout()
    
    if save_fig == True:
        directory_name = 'output'
        save_name = '{}.eps'.format(filename_side[:-4])
        save_plt_fig(directory_name, save_name)
        
    if show_fig == False:
        plt.close(fig)
    
    return (side_length, slope/1000)


def save_figures_and_results(n, y_min, y_max, save_fig=False, show_fig=False, save_df=False):
    """
    n: (int) total number of profiles 
    """   
    # Initialize dataframe for storing results 
    col_names = ['profile_number','rc(nm)', 'rm_1(nm)','rm_2(nm)',\
                 'm_1(μm)','m_2(μm)','q_1(rad)','q_2(rad)']
    df = np.zeros((n,len(col_names))) 
    # Generates array of numbers up to the provided number of profiles, n
    num_list = np.arange(1,n+1) 
    for num in num_list:
        # Get file names 
        num = str(num) # change data type from int to str
        filename_avg = 'flat_{}_stat.txt'.format(num)
        filename_c = 'corner_{}.txt'.format(num)
        filename_r1 = 'root1_{}.txt'.format(num)
        filename_s1 = 'side1_{}.txt'.format(num)
        filename_r2 = 'root2_{}.txt'.format(num)
        filename_s2 = 'side2_{}.txt'.format(num)
        # Rim height at the corner 
        rc = corner(filename_c, filename_avg, show_fig, save_fig)
        # Rim height at the root 1 
        r_m1 = root(filename_r1, filename_avg, show_fig, save_fig)
        # Rim height at the root 2
        r_m2 = root(filename_r2, filename_avg, show_fig, save_fig)
        # Length and slope of simplified profile of side1
        m1, q1 = side(filename_s1, filename_avg, filename_c, filename_r1,\
                      '1', y_min, y_max, show_fig, save_fig)
        # Length and slope of simplified profile of side2
        m2, q2 = side(filename_s2, filename_avg, filename_c, filename_r2,\
                      '2', y_min, y_max, show_fig, save_fig)
        # Fill in the dataframe with results
        df[int(num)-1]= int(num), rc, r_m1, r_m2, m1, m2, q1, q2 
    
    # Put the results into Pandas DataFrame
    df = pd.DataFrame(df, columns=col_names)
    # Save the results
    if save_df == True:
        fa.make_dir_and_output_df_to_excel('output',df,'Summary of results','')
            
    return df

    
    

    
    

