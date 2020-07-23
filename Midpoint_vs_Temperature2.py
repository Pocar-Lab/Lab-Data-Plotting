######################################## Midpoint vs. Temperature ##########################################
# This script plots the temperature of the system on the x-axis and the midpoint of the alpha peak histogram
# on the y-axis. 
# The raw data will be plotted as error bars, and the best fit line for the data will be found and plotted.
# This script can also find and plot the ratio of two best fit lines for separate sets of data, and the 
# errors of these ratios. 
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

## Change these variables to specify the desired conditions for the plot
dates = ['20190516']                # List all of the dates to be included
separations = ['27']                # List all of the separations to be included
bias_voltages = ['50V']             # List the bias voltages to be included
num_sets = 1                        # How many data sets should be plotted?

## These variables set the labels for the plot legends. Change these to indicate what should be written on the labels.
set_1 = '27mm'
set_2 = '38mm'
# set_3 = '31mm'

plot_ratios = False
y_range_data = (0.0, 1.8)
y_range_ratio = (0.1, 1.0)

## Color variables
data_colors = {
    'set_1': '#FF5A8C',
    'set_2': '#B073FF',
    'set_3': '#FF7D00',
}

data_ecolors = {
    'set_1': '#B40F4C',
    'set_2': '#7800BF',
    'set_3': '#CB4B00',
}

ratio_colors = {
    'ratio_1': '#00B400',
    'ratio_2': '#00E1BE',
    'ratio_3': '#0096FF',
}

ratio_error_colors = {
    'ratio_1': '#007800',
    'ratio_2': '#00A68C',
    'ratio_3': '#005F80',
}

## Unchanged variables
alphas_filename = 'alphas.h5'
temp_ints = np.array([166, 167, 168, 169, 170, 171, 172])
temp_ints_shifted = temp_ints-169

#==========================================================================================================
### Functions ###

# Set text strings to display on the figure
def set_text_str(slope_label, intercept_label, slope, intercept, slope_error, intercept_error, red_chi2_2d):
    txt_str = '\n'.join(['V = {}*(T-169) + {}'.format(slope_label, intercept_label),
                        'slope {} = {:.4f} +/- {:.4f}'.format(slope_label, slope, slope_error),
                        'intercept {} = {:.4f} +/- {:.4f}'.format(intercept_label, intercept, intercept_error),
                        r'$\chi^2_{2D}/dof$' + f' = {red_chi2_2d:.1f}'])
    return txt_str

# Plot the raw data
def plot_data(ax, slope_label, intercept_label, df, color, ecolor, label):

    temps, midpoints, temp_errors, midpoint_errors = define_xy_values(df, 'temperature_avg', 'midpoint', 'temperature_rms', 'midpt_error')
    temps_shifted = temps-169

    fit_pars, cov_matrix = get_fit_parameters(temps_shifted, midpoints, temp_errors, midpoint_errors)
    opt_pars = fit_pars[0]
    best_fit_line = linear_func(opt_pars, temp_ints_shifted)

    ax.errorbar(temps_shifted, midpoints, midpoint_errors, temp_errors, ls='none', color=ecolor, barsabove=True, zorder=3, label=label)
    ax.plot(temp_ints_shifted, best_fit_line, c=color, label=label, linewidth=0.8)

    # red_chisquare_1d = calc_red_chisquare_1d(optimized_parameters, temps_shifted, midpoints, midpoint_errors)
    red_chi2_2d = calc_red_chisquare_2d(opt_pars, temps_shifted, midpoints, temp_errors, midpoint_errors)

    (slope, intercept) = fit_pars[0]
    (slope_error, intercept_error) = fit_pars[1]
    txt_str = set_text_str(slope_label, intercept_label, slope, intercept, slope_error, intercept_error, red_chi2_2d)

    return fit_pars, temps_shifted, midpoints, midpoint_errors, cov_matrix, txt_str  # Fix the parameters going in when calling this function

#==========================================================================================================
### Executing Functions ###

if __name__ == '__main__':

    separations = sorted(separations)

    # Define preliminary variables for plotting
    fig, (ax_data, ax_sub) = plt.subplots(2, 1, sharex=False, figsize=(10, 7))
    ax_data.set_position((0.08, 0.4, 0.6, 0.5))
    ax_sub.set_position((0.08, 0.1, 0.6, 0.2))
    axes = [ax_data, ax_sub]

    # plot_suptitle = 'Midpoint vs. Temperature at {} Separation, {} Bias Voltage\n'.format(separations, bias_voltages)
    # plot_title = ''

    df_all_data = compile_data(alphas_filename)

    if num_sets == 1:

        # Create the data frame containing all of the data to be plotted
        dataframe = create_df(df_all_data, dates, separations[0], bias_voltages[0])

        # Exclude the outlier found on 02/07/2019
        if dates[0] == '20190207':
            dataframe = dataframe[dataframe['midpoint'] > 1.0]

        # Deletes the data point that's been mislabeled as 50V

        # Plot the raw data and the best fit line
        fit_pars, temps_shifted, midpoints, midpoint_errors, cov_matrix, txt_str = plot_data(ax_data, 'a', 'b', dataframe, data_colors['set_1'], data_ecolors['set_1'], label=set_1)
        
        # Plot the residuals on a subplot below the main plot
        get_residual_percentages(ax_sub, temps_shifted, temp_ints_shifted, fit_pars[0], midpoints, midpoint_errors, data_colors['set_1'], data_ecolors['set_1'])

        # Setting the super title and the title variables
        plot_suptitle = 'Midpoint vs. Temperature at {}mm Separation, {} Bias Voltage\n'.format(separations[0], bias_voltages[0])
        plot_title = 'Date(s) Taken: ' + ', '.join(dataframe.date.unique())

        # Creating the text box
        plt.figtext(0.75, 0.5, txt_str, color=data_colors['set_1'], fontsize=10)
    
    if num_sets == 2:

        # Creates two separate data frames each containing one set of data to be plotted
        df_1 = create_df(df_all_data, dates, separations[0], bias_voltages[0])
        df_2 = create_df(df_all_data, dates, separations[1], bias_voltages[0])

        # Setting the super title and the title variables
        plot_suptitle = 'Midpoint vs. Temperature at {} Bias Voltage\n'.format(bias_voltages[0])
        dates_1 = ', '.join(df_1.date.unique())
        dates_2 = ', '.join(df_2.date.unique())
        plot_title = 'Dates Taken: ' + dates_1 + ' ' + dates_2

        # Plot the raw data and the best fit line
        fit_pars_1, temps_shifted_1, midpt_1, midpt_err_1, cov_matrix_1, txt_str_1 = plot_data(ax_data, 'a', 'b', df_1, data_colors['set_1'], data_ecolors['set_1'], label=set_1)
        fit_pars_2, temps_shifted_2, midpt_2, midpt_err_2, cov_matrix_2, txt_str_2 = plot_data(ax_data, 'c', 'd', df_2, data_colors['set_2'], data_ecolors['set_2'], label=set_2)

        # Plot the residuals on a subplot below the main plot
        get_residual_percentages(ax_sub, temps_shifted_1, temp_ints_shifted, fit_pars_1[0], midpt_1, midpt_err_1, data_colors['set_1'], data_ecolors['set_1'])
        get_residual_percentages(ax_sub, temps_shifted_2, temp_ints_shifted, fit_pars_2[0], midpt_2, midpt_err_2, data_colors['set_2'], data_ecolors['set_2']) 

        # Creating the text boxes
        plt.figtext(0.75, 0.55, txt_str_1, color=data_colors['set_1'], fontsize=10)
        plt.figtext(0.75, 0.35, txt_str_2, color=data_colors['set_2'], fontsize=10)

    #==========================================================================================================

    ### Plot Settings ###

    # Setting the y range
    ax_data.set_ylim(*y_range_data)

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
    plt.suptitle('\n'.join([plot_suptitle, plot_title]), fontsize=12)

    # Settings for the ratio plot
    # if plot_ratios:
    #     ax_ratio.set_ylim(y_range_ratio)
    #     ax_ratio.set_ylabel('Ratio', fontsize=14)
    #     ax_ratio.legend(bbox_to_anchor=(1.55, 1.0))
    #     ax_ratio.grid(False)
    
    ax_data.legend(bbox_to_anchor=(1.3, 1.0), frameon=False)
    plt.show()
