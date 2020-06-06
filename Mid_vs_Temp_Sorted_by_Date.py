############################### Midpoint vs. Temperature (Sorted by Date) #################################
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
date_list = ['20190207', '20181211']
bias_voltage = '50V'
separation = '27'
y_range = (0.5, 1.4)

## Color variables
colors = {
    '20190207': '#FF0000',
    '20190124': '#FF7D00',
    '20181212': '#00C800',
    '20181211': '#0096FF',
    '20181210': '#A000FF',
}

error_colors = {
    '20190207': '#BF0000',
    '20190124': '#CC6400',
    '20181212': '#008000',
    '20181211': '#0050FF',
    '20181210': '#7800BF',
}

label = {
    '20190207': '02/07/2019',
    '20190124': '01/24/2019',
    '20181212': '12/12/2018',
    '20181211': '12/11/2018',
    '20181210': '12/10/2018',
}

## Unchanged variables
alphas_filename = 'alphas.h5'
temperature_ints = np.array([166, 167, 168, 169, 170, 171, 171])
temperature_ints_shifted = temperature_ints-169

#==========================================================================================================
### Executing Functions ###

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.set_position((0.1, 0.1, 0.6, 0.8))

dataframes = [compile_data(alphas_filename, [date]) for date in date_list]

txt_pos = 0.55
for df in dataframes:

    # Create the dataframe containing the data to be plotted
    # dataframe = compile_data(alphas_filename, [date])
    dataframe = create_df(df, separation, bias_voltage)
    date = dataframe.iloc[0]['date']

    if date == '20181211':
        # Adds the two data points from 12/12/2018 to the set of data taken on 12/11/2018
        two_point_df = compile_data(alphas_filename, ['20181212'])
        two_point_df = create_df(two_point_df, separation, bias_voltage)
        dataframe = dataframe.append(two_point_df)

        # Eliminates a data point that's been mislabeled as 50V
        if bias_voltage == '50V':
            dataframe = dataframe[dataframe['midpoint'] > 0.8]
    
    # Exclude the outlier found on 02/07/2019
    if date == '20190207':
        dataframe = dataframe[dataframe['midpoint'] > 1.0]

    # Plot the data
    fit_parameters, cov_matrix, txt_str = plot_data(ax, 'a', 'b', dataframe, separation, bias_voltage, colors[date], error_colors[date], label=label[date])

    # Create and place the text box
    plt.figtext(0.75, txt_pos, txt_str, color=colors[date], fontsize=10)
    txt_pos = txt_pos-0.2

### Plot Settings ###

# Setting the axis labels 
ax.set_xlabel('Temperature [K]')
ax.set_ylabel('Midpoint [V]')

# Setting the x and y limits
ax.set_xlim(-4, 4, 1)
ax.set_ylim(y_range)

# Setting the title
plt.suptitle('Midpoint vs. Temperature at {} Bias Voltage, {}mm Separation'.format(bias_voltage, separation))

# Label the x-ticks with the actual temperature values
for ax in [ax]:
    locs = ax.get_xticks()
    adjusted_locs = [str(int(l+169)) for l in locs]
    ax.set_xticklabels(adjusted_locs)

plt.grid(True)
ax.legend(bbox_to_anchor=(1.3, 1.0))
plt.show()