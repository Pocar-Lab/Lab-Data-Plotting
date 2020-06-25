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

date_list_before = ['20190207']
date_list_after = ['20190516']
separation = ['27']
temperature = np.array([168])
temperature_shifted = temperature-169

bias_voltages = ['47V', '48V', '49V', '50V', '51V', '52V']
temperature_ints =np.array([166, 167, 168, 169, 170, 171, 172])
temperature_ints_shifted = temperature_ints-169
alphas_filename = 'alphas.h5'

# Plot Colors
line_color = '#B073FF'
error_color = '#7800BF'
txt_color = '#E64676'
fill_color = '#E9D9FF'

#==========================================================================================================
### Executing Functions ###

fig, ax = plt.subplots(figsize=(7, 6))
ax.set_position((0.1, 0.15, 0.7, 0.7))

ratio_list = []
ratio_error_list = []

# Loop through the bias voltages, and find the ratio at the given temperature
for voltage in bias_voltages:

    # Get the dataframes for the data taken before and the data taken after
    df_before = compile_data(alphas_filename, date_list_before)
    df_before = create_df(df_before, separation[0], voltage)
    if voltage == '50V':
        df_before = df_before[df_before['midpoint'] > 1.0]
    
    df_after = compile_data(alphas_filename, date_list_after)
    df_after = create_df(df_after, separation[0], voltage)

    # Define the x and y values of the plot
    temps_before, midpoints_before, temp_errors_before, midpoint_errors_before = define_xy_values(df_before, 'temperature_avg', 'midpoint', 'temperature_rms', 'midpt_error')
    temps_after, midpoints_after, temp_errors_after, midpoint_errors_after = define_xy_values(df_after, 'temperature_avg', 'midpoint', 'temperature_rms', 'midpt_error')
    temps_before_shifted = temps_before-169
    temps_after_shifted = temps_after-169
    
    # Get the fit parameters for each set of data
    fit_parameters_before, cov_matrix_before = get_fit_parameters(temps_before_shifted, midpoints_before, temp_errors_before, midpoint_errors_before)
    fit_parameters_after, cov_matrix_after = get_fit_parameters(temps_after_shifted, midpoints_after, temp_errors_after, midpoint_errors_after)

    # Get the ratio point and the ratio error at the given temperature
    ratio_yvalue, ratio_point, ratio_error = get_ratio_errors(fit_parameters_before, fit_parameters_after, cov_matrix_before, cov_matrix_after, temperature_shifted)
    ratio_list.append(ratio_yvalue)
    ratio_error_list.append(ratio_error)

# Convert all of the lists into numpy arrays
bias_voltages = np.array(bias_voltages)
ratio_list = np.array(ratio_list).T[0]      # Takes the transpose of the array - this makes it go from a single column to a single row.
ratio_error_list = np.array(ratio_error_list).T[0]

print(bias_voltages)
print(ratio_list)
print(ratio_error_list)


#==========================================================================================================
### Plotting the Data ###

ax.scatter(bias_voltages, ratio_list, color=line_color, label='Ratio at {}K'.format(temperature[0]))
ax.errorbar(bias_voltages, ratio_list, ratio_error_list, ls='none', color=error_color, barsabove=True, zorder=3)

#==========================================================================================================
### Plot Settings ###

# Setting the y range

# Setting the axis labels
ax.set_xlabel('Bias Voltage [V]', fontsize=14)
ax.set_ylabel('Ratio', fontsize=14)

# Setting the title
plot_suptitle = 'Ratio After Baking/Before Baking vs. Bias Voltage at {}K Temperature, {}mm Separation'.format(temperature[0], separation[0])
date_before = ', '.join(df_before.date.unique())
date_after = ', '.join(df_after.date.unique())
plot_title = 'Before baking: {}     After baking: {}'.format(date_before, date_after)
plt.suptitle('\n'.join([plot_suptitle, plot_title]), fontsize=12)

plt.grid(True)
ax.legend(bbox_to_anchor=(1.3, 1.0))
plt.show()