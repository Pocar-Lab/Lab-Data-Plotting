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
from Midpoint_vs_Temperature import plot_residuals

#==========================================================================================================
### Variables ###

# date_list = ['20190516', '20190424']
date_before = ['20190207']
date_after = ['20190516']
separation = ['27']
# separations = ['27', '38']
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

# separations = sorted(separations)

# Variables for plotting
fig, ax_ratio = plt.subplots(figsize=(8, 6))
ax_ratio.set_position((0.1, 0.1, 0.7, 0.8))
axes = [ax_ratio]

# df_all_data = compile_data(alphas_filename, date_list)

for voltage in bias_voltages:

    # Get the separated dataframes
    # df_small = create_df(df_all_data, separations[0], voltage)
    # df_large = create_df(df_all_data, separations[1], voltage)
    df_before = compile_data(alphas_filename, date_before)
    df_before = create_df(df_before, separation[0], voltage)
    if voltage == '50V':
        df_before = df_before[df_before['midpoint'] > 1.0]
    
    df_after = compile_data(alphas_filename, date_after)
    df_after = create_df(df_after, separation[0], voltage)
    if voltage ==  '48V':
        df_after = df_after.iloc[1:]

    # Define the x and y values
    temps_before, midpoints_before, temp_errors_before, midpoint_errors_before = define_xy_values(df_before, 'temperature_avg', 'midpoint', 'temperature_rms', 'midpt_error')
    temps_before_shifted = temps_before-169
    temps_after, midpoints_after, temp_errors_after, midpoint_errors_after = define_xy_values(df_after, 'temperature_avg', 'midpoint', 'temperature_rms', 'midpt_error')
    temps_after_shifted = temps_after-169

    # Get the fit parameters for both sets of data
    fit_parameters_before, cov_matrix_before = get_fit_parameters(temps_before_shifted, midpoints_before, temp_errors_before, midpoint_errors_before)
    fit_parameters_after, cov_matrix_after = get_fit_parameters(temps_after_shifted, midpoints_after, temp_errors_after, midpoint_errors_after)

    # Get the residual standard deviation
    residuals_after, residual_std_after = plot_residuals(ax_ratio, temps_after_shifted, fit_parameters_after[0], midpoints_after, midpoint_errors_after, colors[voltage], error_colors[voltage], 'After baking')
    residuals_before, residual_std_before = plot_residuals(ax_ratio, temps_before_shifted, fit_parameters_before[0], midpoints_before, midpoint_errors_before, colors[voltage], error_colors[voltage], 'Before baking')

    # Find the ratios and the errors on the ratios 
    # ratio_y_vals, ratio_line, ratio_errors = get_ratio_errors(fit_parameters_before, fit_parameters_after, cov_matrix_before, cov_matrix_after, temperature_ints_shifted)
    ratio_line, ratio_y_vals, ratio_errors = get_ratios(fit_parameters_before, fit_parameters_after, residual_std_before, residual_std_after, temperature_ints_shifted)

    # Plot the ratio line and the ratio errors
    ax_ratio.plot(temperature_ints_shifted, ratio_line, c=colors[voltage], label=voltage)
    ax_ratio.errorbar(temperature_ints_shifted, ratio_y_vals, ratio_errors, ls='none', color=error_colors[voltage], barsabove=True, zorder=3)
    ax_ratio.fill_between(temperature_ints_shifted, ratio_y_vals-ratio_errors, ratio_y_vals+ratio_errors, color=colors[voltage], alpha=0.2)

#==================================================================================================
### Plot Settings

# Setting the axis labels
ax_ratio.set_xlabel('Temperature', fontsize=14)
ax_ratio.set_ylabel('Ratio', fontsize=14)

# Setting the super title and the title 
plt.suptitle('Ratio of Best Fit Lines After Baking/Before Baking vs. Temperature at {}mm Separation'.format(separation[0]))
date_1 = ', '.join(df_before.date.unique())
date_2 = ', '.join(df_after.date.unique())
plt.title('Before baking: {}    After baking: {}'.format(date_1, date_2))

# Label the x-ticks with the actual temperature values (166-172)
for ax in axes:
    locs = ax.get_xticks()
    adjusted_locs = [str(int(l+169)) for l in locs]
    ax.set_xticklabels(adjusted_locs)

ax_ratio.legend(bbox_to_anchor=(1.25, 1.0), frameon=False)
plt.grid(True)
plt.show()