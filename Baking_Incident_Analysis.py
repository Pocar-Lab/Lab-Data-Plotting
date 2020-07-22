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
bias_voltage = '49V'
separation = ['27']

plot_ratios = True
y_range_data = (0.5, 1.1)
y_range_ratio = (1.0, 1.6)

# Color variables 
data_colors = {
    '20190516': '#FF5A8C',
    '20190207': '#B073FF',
}

data_error_colors = {
    '20190516': '#B40F4C',
    '20190207': '#7800BF',
}

ratio_color = '#00B400'
ratio_error_color = '#007800'

## Unchanged variables
alphas_filename = 'alphas.h5'
temperature_ints =np.array([166, 167, 168, 169, 170, 171, 172])
temperature_ints_shifted = temperature_ints-169

#==========================================================================================================
### Executing Functions ###

# Create the dataframes for the data taken before and data taken after the baking incident.
df_before = compile_data(alphas_filename, date_list_before)
df_before = create_df(df_before, separation[0], bias_voltage)

if bias_voltage == '50V':
    df_before = df_before[df_before['midpoint'] > 1.0]

df_after = compile_data(alphas_filename, date_list_after)
df_after = create_df(df_after, separation[0], bias_voltage)

if bias_voltage ==  '48V':
    df_after = df_after.iloc[1:]

# Preliminary variables for plotting
fig, (ax_data, ax_sub) = plt.subplots(2, 1, sharex=False, figsize=(9, 6))
ax_data.set_position((0.08, 0.4, 0.55, 0.5))
ax_sub.set_position((0.08, 0.1, 0.6, 0.2))
axes = [ax_data, ax_sub]

plot_suptitle = 'Midpoint vs. Temperature at {}mm Separation, {} Bias Voltage\n'.format(separation[0], bias_voltage)
date_before = ', '.join(df_before.date.unique())
date_after = ', '.join(df_after.date.unique())
plot_title = 'Before baking: {}     After baking: {}'.format(date_before, date_after)

# Call the function to plot the data, and return the fit parameters, cov matrix, and text string
fit_parameters_after, temps_shifted_after, midpoints_after, midpoint_errors_after, cov_matrix_after, txt_str_after = plot_data(ax_data, 'c', 'd', df_after, separation[0], bias_voltage, data_colors[date_list_after[0]], data_error_colors[date_list_after[0]], label='After baking')
fit_parameters_before, temps_shifted_before, midpoints_before, midpoint_errors_before, cov_matrix_before, txt_str_before = plot_data(ax_data, 'a', 'b', df_before, separation[0], bias_voltage, data_colors[date_list_before[0]], data_error_colors[date_list_before[0]], label='Before baking')

# Plot the residuals on a subplot below the main plot
residuals_after, residual_std_after = plot_residuals(ax_sub, temps_shifted_after, fit_parameters_after[0], midpoints_after, midpoint_errors_after, data_colors[date_list_after[0]], data_error_colors[date_list_after[0]], 'After baking')
residuals_before, residual_std_before = plot_residuals(ax_sub, temps_shifted_before, fit_parameters_before[0], midpoints_before, midpoint_errors_before, data_colors[date_list_before[0]], data_error_colors[date_list_before[0]], 'Before baking')
print(residual_std_after)
print(residual_std_before)
residual_percentages(ax_sub, temps_shifted_after, fit_parameters_after[0], midpoints_after, data_colors[date_list_after[0]], data_error_colors[date_list_after[0]], 'After baking')
residual_percentages(ax_sub, temps_shifted_before, fit_parameters_before[0], midpoints_before, data_colors[date_list_before[0]], data_error_colors[date_list_before[0]], 'Before baking')

# Setting the positions of the text on the figure
if not plot_ratios:
    plt.figtext(0.75, 0.55, txt_str_after, color=data_colors[date_list_after[0]], fontsize=10)
    plt.figtext(0.75, 0.35, txt_str_before, color=data_colors[date_list_before[0]], fontsize=10)

if plot_ratios:

    ax_ratio = ax_data.twinx()
    ax_ratio.set_position((0.08, 0.4, 0.6, 0.5))
    ax_ratio.tick_params(axis='y', colors='green')
    axes = [ax_data, ax_sub, ax_ratio]

    # Find and plot ratios and the ratio errors
    # ratio_yvalues, ratio_line, ratio_errors = get_ratio_errors(fit_parameters_before, fit_parameters_after, cov_matrix_before, cov_matrix_after, temperature_ints_shifted)
    ratio_line, ratio_yvalues, ratio_errors = get_ratios(fit_parameters_before, fit_parameters_after, residual_std_before, residual_std_after, temperature_ints_shifted)
    ax_ratio.plot(temperature_ints_shifted, ratio_line, c=ratio_color, label='After baking/Before baking', linewidth=0.8)
    ax_ratio.errorbar(temperature_ints_shifted, ratio_yvalues, ratio_errors, ls='none', color=ratio_error_color, barsabove=True, zorder=3)
    ax_ratio.fill_between(temperature_ints_shifted, ratio_yvalues-ratio_errors, ratio_yvalues+ratio_errors, color=ratio_color, alpha=0.2)

    ratio_yvalues = np.array(ratio_yvalues)
    average_ratio = np.mean(ratio_yvalues)
    # print(average_ratio)

    # Setting the positions of the text on the figure
    plt.figtext(0.78, 0.55, txt_str_after, color=data_colors[date_list_after[0]], fontsize=10)
    plt.figtext(0.78, 0.4, txt_str_before, color=data_colors[date_list_before[0]], fontsize=10)
    plt.figtext(0.78, 0.3, '\n'.join(['Average Ratio:', '{:.2f} +/- {:.2f}'.format(average_ratio, ratio_errors[0]),]), color=ratio_color, fontsize=10)

before = (fit_parameters_before[0][0])*(-0.5) + fit_parameters_before[0][1]
after = (fit_parameters_after[0][0])*(-0.5) + fit_parameters_after[0][1]
# print(after/before)

#==========================================================================================================
### Plot Settings ###

# Setting the y range
ax_data.set_ylim(*y_range_data)
ax_sub.set_ylim(-1.2, 1.2)

# Setting the axis labels
ax_data.set_xlabel('Temperature [K]', fontsize=14)
ax_data.set_ylabel('SiPM Output [V]', fontsize=14)
ax_sub.set_xlabel('Temperature [K]', fontsize=14)
ax_sub.set_ylabel('Residuals [%]', fontsize=14)

# Label the x-ticks with the actual temperature values (166-172)
for ax in axes:
    locs = ax.get_xticks()
    adjusted_locs = [str(int(l+169)) for l in locs]
    ax.set_xticklabels(adjusted_locs)

# Setting the super title and the title
plt.title('\n'.join([plot_suptitle, plot_title]), fontsize=12)

# Settings for the ratio plot
if plot_ratios:
    ax_ratio.set_ylim(y_range_ratio)
    ax_ratio.set_ylabel('Ratio', color='green', fontsize=14)
    ax_ratio.legend(bbox_to_anchor=(1.56, 0.75), frameon=False)
    ax_ratio.grid(False)

ax_data.legend(bbox_to_anchor=(1.4, 1.05), frameon=False)
# ax_sub.legend(bbox_to_anchor=(1.52, 0.7), frameon=False)
plt.show()