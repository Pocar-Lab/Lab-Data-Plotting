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
from Midpoint_vs_Temperature import *

#==========================================================================================================
### Variables ###

date_list_before_27mm = ['20190207']
date_list_after_27mm = ['20190516']
date_list_before_38mm = ['20181204']
date_list_after_38mm = ['20190411']

separations = ['27', '38']
temperature = np.array([168])
temperature_shifted = temperature-169

bias_voltages = ['47V', '48V', '49V', '50V', '51V', '52V']
temperature_ints =np.array([166, 167, 168, 169, 170, 171, 172])
temperature_ints_shifted = temperature_ints-169
alphas_filename = 'alphas.h5'

# Plot Colors
line_color_27mm = '#FF5A8C'
error_color_27mm = '#B40F4C'

line_color_38mm = '#B073FF'
error_color_38mm = '#7800BF'
txt_color_38mm = '#E64676'
fill_color_38mm = '#E9D9FF'

#==========================================================================================================
### Executing Functions ###

fig, ax = plt.subplots(figsize=(7, 6))
ax.set_position((0.1, 0.1, 0.66, 0.8))

ratio_list_27mm = []
ratio_error_list_27mm = []

ratio_list_38mm = []
ratio_error_list_38mm = []

# Loop through the bias voltages, and find the ratio at the given temperature
for voltage in bias_voltages:

    # Get the dataframes for the data taken before and the data taken after
    df_before_27mm = compile_data(alphas_filename, date_list_before_27mm)
    df_before_27mm = create_df(df_before_27mm, separations[0], voltage)
    if voltage == '50V':
        df_before_27mm = df_before_27mm[df_before_27mm['midpoint'] > 1.0]
    
    df_after_27mm = compile_data(alphas_filename, date_list_after_27mm)
    df_after_27mm = create_df(df_after_27mm, separations[0], voltage)

    df_before_38mm = compile_data(alphas_filename, date_list_before_38mm)
    df_before_38mm = create_df(df_before_38mm, separations[1], voltage)
    
    df_after_38mm = compile_data(alphas_filename, date_list_after_38mm)
    df_after_38mm = create_df(df_after_38mm, separations[1], voltage)

    # Define the x and y values of the plot
    temps_before_27mm, midpoints_before_27mm, temp_errors_before_27mm, midpoint_errors_before_27mm = define_xy_values(df_before_27mm, 'temperature_avg', 'midpoint', 'temperature_rms', 'midpt_error')
    temps_after_27mm, midpoints_after_27mm, temp_errors_after_27mm, midpoint_errors_after_27mm = define_xy_values(df_after_27mm, 'temperature_avg', 'midpoint', 'temperature_rms', 'midpt_error')
    temps_before_shifted_27mm = temps_before_27mm-169
    temps_after_shifted_27mm = temps_after_27mm-169

    temps_before_38mm, midpoints_before_38mm, temp_errors_before_38mm, midpoint_errors_before_38mm = define_xy_values(df_before_38mm, 'temperature_avg', 'midpoint', 'temperature_rms', 'midpt_error')
    temps_after_38mm, midpoints_after_38mm, temp_errors_after_38mm, midpoint_errors_after_38mm = define_xy_values(df_after_38mm, 'temperature_avg', 'midpoint', 'temperature_rms', 'midpt_error')
    temps_before_shifted_38mm = temps_before_38mm-169
    temps_after_shifted_38mm = temps_after_38mm-169
    
    # Get the fit parameters for each set of data
    fit_parameters_before_27mm, cov_matrix_before_27mm = get_fit_parameters(temps_before_shifted_27mm, midpoints_before_27mm, temp_errors_before_27mm, midpoint_errors_before_27mm)
    fit_parameters_after_27mm, cov_matrix_after_27mm = get_fit_parameters(temps_after_shifted_27mm, midpoints_after_27mm, temp_errors_after_27mm, midpoint_errors_after_27mm)

    fit_parameters_before_38mm, cov_matrix_before_38mm = get_fit_parameters(temps_before_shifted_38mm, midpoints_before_38mm, temp_errors_before_38mm, midpoint_errors_before_38mm)
    fit_parameters_after_38mm, cov_matrix_after_38mm = get_fit_parameters(temps_after_shifted_38mm, midpoints_after_38mm, temp_errors_after_38mm, midpoint_errors_after_38mm)

    # Get the ratio point and the ratio error at the given temperature
    ratio_yvalue_27mm, ratio_point_27mm, ratio_error_27mm = get_ratio_errors(fit_parameters_before_27mm, fit_parameters_after_27mm, cov_matrix_before_27mm, cov_matrix_after_27mm, temperature_shifted)
    ratio_list_27mm.append(ratio_yvalue_27mm)
    ratio_error_list_27mm.append(ratio_error_27mm)

    ratio_yvalue_38mm, ratio_point_38mm, ratio_error_38mm = get_ratio_errors(fit_parameters_before_38mm, fit_parameters_after_38mm, cov_matrix_before_38mm, cov_matrix_after_38mm, temperature_shifted)
    ratio_list_38mm.append(ratio_yvalue_38mm)
    ratio_error_list_38mm.append(ratio_error_38mm)

# Convert all of the lists into numpy arrays
bias_voltages = np.array(bias_voltages)
ratio_list_27mm = np.array(ratio_list_27mm).T[0]      # Takes the transpose of the array - this makes it go from a single column to a single row.
ratio_error_list_27mm = np.array(ratio_error_list_27mm).T[0]

ratio_list_38mm = np.array(ratio_list_38mm).T[0]
ratio_error_list_38mm = np.array(ratio_error_list_38mm).T[0]

#==========================================================================================================
### Plotting the Data ###

ax.scatter(bias_voltages, ratio_list_27mm, color=line_color_27mm, label='27mm Separation'.format(temperature[0]))
ax.errorbar(bias_voltages, ratio_list_27mm, ratio_error_list_27mm, ls='none', color=error_color_27mm, barsabove=True, zorder=3)

ax.scatter(bias_voltages, ratio_list_38mm, color=line_color_38mm, label='38mm Separation'.format(temperature[0]))
ax.errorbar(bias_voltages, ratio_list_38mm, ratio_error_list_38mm, ls='none', color=error_color_38mm, barsabove=True, zorder=3)

#==========================================================================================================
### Plot Settings ###

# Setting the y range

# Setting the axis labels
ax.set_xlabel('Bias Voltage [V]', fontsize=14)
ax.set_ylabel('Ratio', fontsize=14)

# Setting the title
plot_suptitle = 'Ratio After Baking/Before Baking vs. Bias Voltage at {}K Temperature'.format(temperature[0])
date_before_27mm = ', '.join(df_before_27mm.date.unique())
date_after_27mm = ', '.join(df_after_27mm.date.unique())
date_before_38mm = ', '.join(df_before_38mm.date.unique())
date_after_38mm = ', '.join(df_after_38mm.date.unique())
plot_title_27mm = '27mm Separaton: Before baking: {}  After baking: {}'.format(date_before_27mm, date_after_27mm)
plot_title_38mm = '38mm Separation: Before baking: {}  After baking: {}'.format(date_before_38mm, date_after_38mm)
# plt.suptitle('\n'.join([plot_suptitle, ' ', plot_title_27mm, ' ', plot_title_38mm]), fontsize=12)
plt.suptitle('Ratio After Baking/Before Baking vs. Bias Voltage at {}K Temperature'.format(temperature[0]))

# Setting the text strings
txt_str_27mm = '\n'.join(['Dates taken at 27mm:', ' ', 'Before baking: {}'.format(date_before_27mm), 'After baking: {}'.format(date_after_27mm)])
plt.figtext(0.77, 0.65, txt_str_27mm, fontsize=10)

txt_str_38mm = '\n'.join(['Dates taken at 38mm:', ' ', 'Before baking: {}'.format(date_before_38mm), 'After baking: {}'.format(date_after_38mm)])
plt.figtext(0.77, 0.45, txt_str_38mm, fontsize=10)

plt.grid(True)
ax.legend(bbox_to_anchor=(1.35, 1.0), frameon=False)
plt.show()