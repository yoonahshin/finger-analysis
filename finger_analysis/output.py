#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 07:54:33 2020

@author: yoonahshin
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
# Plot settings
#plt.rcParams.update({'font.size': 18}) 
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams["font.family"] = 'Arial'

def save_fig(directory_name, save_name):
    """
    If directory with name directory_name does not exist, create a directory
    with the name and save the figure in the directory. 
    """
    if not os.path.isdir(directory_name):
        os.makedirs(directory_name)    
    path = os.path.join(directory_name, save_name)
    plt.savefig(path, bbox_inches='tight')
    return None

def make_dir_and_output_df_to_excel(DirectoryName, df, filename, suffix):
    """
    If directory with name DirectoryName does not exist, create a directory 
    with the name and save the dataframe as an excel file with name,
    filename+suffix.xlsx in the directory. 
    """
    if not os.path.isdir(DirectoryName):
        os.makedirs(DirectoryName)
    path = os.path.join(DirectoryName, filename+suffix+'.xlsx')
    writer = pd.ExcelWriter(path)
    df.to_excel(writer)
    writer.save()
    return None

def plot_histogram(dir_name, h, alph, t, idx, suffix_name,\
                   x_label, y_label, x_min, x_max, tick_spacing,\
                   fontsize=22, save_format='png', plot_mean=False,\
                   save_hist=False, color_user='b'):
    """
    dir_name = 'finger_period'
    h = 120 # initial film thickness (nm)
    alph = 20 # initial edge orientation (°)
    t = 3 # annealing time (hour)
    idx = 0 or 1
    x_label = 'Natural period (\u03BCm)'
    x_min = 0
    x_max = 6
    """
    filename = '{}nm_{}deg_{}h_{}'.format(h, alph, t, suffix_name)
    x = pd.read_excel('{}/{}.xlsx'.format(dir_name, filename))
    x = x.loc[:, ~x.columns.str.contains('^Unnamed')].iloc[:,idx]
    
    ax = sns.distplot(x, hist_kws={'edgecolor':'k', 'color':color_user}, kde=False)
    if plot_mean == True:
        plt.axvline(x.mean(), color=color_user, linewidth=2.5, linestyle='--')
    
    # Plot settings
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_xlim(xmin=x_min, xmax=x_max)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    save_name = '{}_idx_{}.{}'.format(filename, idx, save_format)
    if save_hist==True:
        save_fig(dir_name, save_name)
    
    return x.mean(), x.std(), len(x)
