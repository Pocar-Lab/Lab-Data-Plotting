################################## Midpoint vs. Temperature Ratio Plot ####################################
# This script creates a plot of all of the ratios from the midpoint vs. temperature plots at all bias voltages
# for a set of data.  
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

#==========================================================================================================
### Variables ###

date_list = ['20190516', '20190424']
separations = ['27', '38']
bias_voltages = ['47V', '48V', '49V', '50V', '51V', '52V']

alphas_filename = 'alphas.h5'
temperature_ints = np.array([166, 167, 168, 169, 170, 171, 172])
temperature_ints_shifted = temperature_ints-169

# Dictionaries that set the color of the line and error bars based on the value of the bias voltage. 
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

#==================================================================================================
### Executing Functions

separations = sorted(separations)

# Variables for plotting
fig, ax_ratio = plt.subplots()
# ax_ratio.set_position((0.1, 0.1, 0.8, 0.8))
axes = [ax_ratio]

df_all_data = compile_data(alphas_filename, date_list)

for voltage in bias_voltages:

    # Get the separated dataframes
    df_small = create_df(df_all_data, separations[0], voltage)
    df_large = create_df(df_all_data, separations[1], voltage)

    # Define the x and y values
    temps_small, midpoints_small, temp_errors_small, midpoint_errors_small = define_xy_values(df_small, 'temperature_avg', 'midpoint', 'temperature_rms', 'midpt_error')
    temps_small_shifted = temps_small-169
    temps_large, midpoints_large, temp_errors_large, midpoint_errors_large = define_xy_values(df_large, 'temperature_avg', 'midpoint', 'temperature_rms', 'midpt_error')
    temps_large_shifted = temps_large-169

    # Get the fit parameters for both sets of data
    fit_parameters_small, cov_matrix_small = get_fit_parameters(temps_small_shifted, midpoints_small, temp_errors_small, midpoint_errors_small)
    fit_parameters_large, cov_matrix_large = get_fit_parameters(temps_large_shifted, midpoints_large, temp_errors_large, midpoint_errors_large)

    # Find the ratios and the errors on the ratios 
    ratio_y_vals, ratio_line, ratio_errors = get_ratio_errors(fit_parameters_small, fit_parameters_large, cov_matrix_small, cov_matrix_large, temperature_ints_shifted)

    # Plot the ratio line and the ratio errors
    ax_ratio.plot(temperature_ints_shifted, ratio_line, c=colors[voltage], label=voltage)
    ax_ratio.errorbar(temperature_ints_shifted, ratio_y_vals, ratio_errors, ls='none', color=error_colors[voltage], barsabove=True, zorder=3)

#==================================================================================================
### Plot Settings

# Setting the axis labels
ax_ratio.set_xlabel('Temperature', fontsize=14)
ax_ratio.set_ylabel('Ratio', fontsize=14)

# Setting the super title and the title 
plt.suptitle('Ratio of Best Fit Lines at {}mm and {}mm'.format(separations[0], separations[1]))
date_small = ', '.join(df_small.date.unique())
date_large = ', '.join(df_large.date.unique())
plt.title('{}mm: {}     {}mm: {}'.format(separations[0], date_small, separations[1], date_large))

# Label the x-ticks with the actual temperature values (166-172)
for ax in axes:
    locs = ax.get_xticks()
    adjusted_locs = [str(int(l+169)) for l in locs]
    ax.set_xticklabels(adjusted_locs)

# ax_ratio.legend(bbox_to_anchor=(1.1, 1.0))
ax_ratio.legend()
plt.grid(True)
plt.show()