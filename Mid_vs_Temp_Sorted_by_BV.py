############################### Midpoint vs. Temperature (Sorted by BV) #################################
# This script plots the temperature of the system on the x-axis and the midpoint of the alpha peak histogram
# on the y-axis.
# This version of the plot uses only one separation and bias voltage, however it will color different sets of
# data based on the date they were taken.
# The script will plot the raw data as error bars, and the best fit line for the data will be found and plotted.
# This script does not plot ratios.
#==========================================================================================================

import h5py
import csv
import pandas as pd
import numpy as np
from numpy import sqrt

import scipy
from scipy import odr

from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from Functions import *
from Midpoint_vs_Temperature import plot_data, set_text_str

#==========================================================================================================
### Variables ###

## Change these variables to specify the desired conditions for the plot
date_list = ['20181211']
separation = '27'

temperature_ints = np.array([166, 167, 168, 169, 170, 171, 172])
temperature_ints_shifted = temperature_ints-169
bias_voltages = np.array(['47V', '48V', '49V', '50V', '51V', '52V'])

alphas_filename = 'alphas.h5'

# Dictionaries that set the colors of the lines and error bars based on the bias voltage
colors = {
    '47V': '#FF0000',
    '48V': '#FF7D00',
    '49V': '#FFE100',
    '50V': '#00C800',
    '51V': '#0096FF',
    '52V': '#A000FF',
}

error_colors = {
    '47V': '#BF0000',
    '48V': '#CC6400',
    '49V': '#CCB400',
    '50V': '#008000',
    '51V': '#0050FF',
    '52V': '#7800BF',
}

#==========================================================================================================
### Executing Functions ###

# Create the figure and set the size of the figure
fig, ax = plt.subplots(1, 1, figsize=(9, 6))
ax.set_position((0.06, 0.1, 0.6, 0.8))

# Create the dataframe with every data point
dataframe = compile_data(alphas_filename, date_list)

# Add the two data points from 12/12/18 if we're working with data from 12/11/18
date = dataframe.iloc[0]['date']
if date == '20181211':
    two_point_df = compile_data(alphas_filename, ['20181212'])
    dataframe = dataframe.append(two_point_df)

# Loop through every bias voltage in the bias voltage list to plot the data
for voltage in bias_voltages:

    dataframe_bv = create_df(dataframe, separation, voltage)

    # Eliminates a data point that's been mislabeled as 50V in the data set from 12/11/18
    if date == '20181211' and voltage == '50V':
        dataframe_bv = dataframe_bv[dataframe_bv['midpoint'] > 0.8]
    
    # Plot the data
    fit_parameters, cov_matrix, txt_str = plot_data(ax, 'a', 'b', dataframe_bv, separation, voltage, colors[voltage], error_colors[voltage], label=voltage)

#==========================================================================================================
### Plot Settings ###

# Setting the axis labels
ax.set_xlabel('Temperature [K]')
ax.set_ylabel('Midpoint [V]')

# Setting the x and y limits
ax.set_xlim(-4, 4, 1)

# Setting the title
plt.suptitle('Midpoint vs. Temperature at {}mm Separation Taken on {}'.format(separation, date))

# Label the x-ticks with the actual temperature values
for ax in [ax]:
    locs = ax.get_xticks()
    adjusted_locs = [str(int(l+169)) for l in locs]
    ax.set_xticklabels(adjusted_locs)

plt.grid(True)
ax.legend(bbox_to_anchor=(1.3, 1.0))
plt.show()