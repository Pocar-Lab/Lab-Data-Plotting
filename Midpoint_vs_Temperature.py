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
date_list = ['20181204', '20190228', '20181119', '20190207', '20181211']    # List the dates for all of the data you wish to include in the plot
bias_voltage = '50V'                    # Enter the bias voltage you want to plot at
separations = ['27', '31.07', '38']              # List the separation(s) you wish to include in the plot. For more than one separation
plot_ratios = False                     # Indicate whether or not you want ratios included in on the plot 
y_range_data = (0.0, 1.8)               # Set the range for the left y-axis (this sets the scale for the data)
y_range_ratio = (0.1, 1.0)              # Set the range for the right y-axis (this sets the scale for the ratio)

## Color variables
data_colors = {
    '27': '#FF5A8C',
    '31.07': '#FF7D00',
    '38': '#B073FF',
}

data_error_colors = {
    '27': '#990F4C',
    '31.07': '#CC6400',
    '38': '#7800BF',
}

ratio_colors = {
    'ratio_1': '#00B400',
    'ratio_2': '#00E1BE',
    'ratio_3': '#00C8E1',
}

ratio_error_colors = {
    'ratio_1': '#007800',
    'ratio_2': '#00A68C',
    'ratio_3': '#005F80',
}

## Unchanged variables
alphas_filename = 'alphas.h5'
temperature_ints =np.array([166, 167, 168, 169, 170, 171, 172])
temperature_ints_shifted = temperature_ints-169

#==========================================================================================================
### Functions ###

# Set text strings to display on the figure
def set_text_str(slope_label, intercept_label, slope, intercept, slope_error, intercept_error, reduced_chisquare_1d, reduced_chisquare_2d):
    txt_str = '\n'.join(['Best fit function:',
                        'V = {}*(T-169) + {}'.format(slope_label, intercept_label),
                        'slope {} = {:.4f} +/- {:.4f}'.format(slope_label, slope, slope_error),
                        'intercept {} = {:.4f} +/- {:.4f}'.format(intercept_label, intercept, intercept_error),
                        r'$\chi^2{1D}$' + f' = {reduced_chisquare_1d:.6f}',
                        r'$\chi^2{2D}$' + f' = {reduced_chisquare_2d:.6f}'])
    return txt_str

# Plot the raw data
def plot_data(ax, slope_label, intercept_label, df, separation, voltage, color, ecolor, label):

    temps, midpoints, temp_errors, midpoint_errors = define_xy_values(df, 'temperature_avg', 'midpoint', 'temperature_rms', 'midpt_error')
    temps_shifted = temps-169

    fit_parameters, cov_matrix = get_fit_parameters(temps_shifted, midpoints, temp_errors, midpoint_errors)
    optimized_parameters = fit_parameters[0]
    best_fit_line = linear_func(optimized_parameters, temperature_ints_shifted)

    ax.errorbar(temps_shifted, midpoints, midpoint_errors, temp_errors, ls='none', color=ecolor, barsabove=True, zorder=3)
    ax.plot(temperature_ints_shifted, best_fit_line, c=color, label=label, linewidth=0.8)

    red_chisquare_1d = calc_red_chisquare_1d(optimized_parameters, temps_shifted, midpoints, midpoint_errors)
    red_chisquare_2d = calc_red_chisquare_2d(optimized_parameters, temps_shifted, midpoints, temp_errors, midpoint_errors)

    (slope, intercept) = fit_parameters[0]
    (slope_error, intercept_error) = fit_parameters[1]
    txt_str = set_text_str(slope_label, intercept_label, slope, intercept, slope_error, intercept_error, red_chisquare_1d, red_chisquare_2d)

    return fit_parameters, cov_matrix, txt_str

#==========================================================================================================
### Executing Functions ###

if __name__ == '__main__':

    separations = sorted(separations)

    # Define preliminary variables for plotting
    df_all_data = compile_data(alphas_filename, date_list)

    fig, ax_data = plt.subplots(1, 1, figsize=(9, 6))
    ax_data.set_position((0.1, 0.1, 0.6, 0.8))
    axes = [ax_data]

    plot_suptitle = 'Midpoint vs. Temperature at {} Bias Voltage\n'.format(bias_voltage)
    plot_title = ''

    if len(separations) == 1:

        dataframe = create_df(df_all_data, separations[0], bias_voltage)
        
        # Delete the data point that's been mislabeled as 50V
        # dataframe = dataframe[dataframe['midpoint'] > 1.0]

        fit_parameters, cov_matrix, txt_str = plot_data(ax_data, 'a', 'b', dataframe, separations[0], bias_voltage, data_colors[separations[0]], data_error_colors[separations[0]], label='{}mm'.format(separations[0]))

        # Setting the super title and the title variables
        plot_suptitle = ('Midpoint vs. Temperature at {} Bias Voltage, {}mm Separation'.format(bias_voltage, separations[0]))
        plot_title = ", ".join(dataframe.date.unique())
        
        # Creating the text box
        plt.figtext(0.78, 0.5, txt_str, color=data_colors[separations[0]], fontsize=8)


    elif len(separations) == 2:

        # Create the separate dataframes for each individual line
        df_small_sep = create_df(df_all_data, separations[0], bias_voltage)
        df_large_sep = create_df(df_all_data, separations[1], bias_voltage)

        # Delete the data point that's been mislabeled as 50V
        df_small_sep = df_small_sep[df_small_sep['midpoint'] > 1.0]

        # Call the function to plot the data, and return the fit parameters, cov matrix, and text string
        fit_parameters_small, cov_matrix_small, txt_str_small = plot_data(ax_data, 'a', 'b', df_small_sep, separations[0], bias_voltage, data_colors[separations[0]], data_error_colors[separations[0]], label='{}mm'.format(separations[0]))
        fit_parameters_large, cov_matrix_large, txt_str_large = plot_data(ax_data, 'c', 'd', df_large_sep, separations[1], bias_voltage, data_colors[separations[1]], data_error_colors[separations[1]], label='{}mm'.format(separations[1]))

        # Setting the plot title variable
        date_small = ", ".join(df_small_sep.date.unique())
        date_large = ", ".join(df_large_sep.date.unique())
        plot_title = '{}mm: {}     {}mm: {}'.format(separations[0], date_small, separations[1], date_large)

        # Setting the positions of the text on the figure
        plt.figtext(0.78, 0.5, txt_str_small, color=data_colors[separations[0]], fontsize=10)
        plt.figtext(0.78, 0.25, txt_str_large, color=data_colors[separations[1]], fontsize=10)

        if plot_ratios:

            ax_ratio = ax_data.twinx()
            ax_ratio.set_position((0.1, 0.1, 0.6, 0.8))
            axes = [ax_data, ax_ratio]

            # Find and plot ratios and the ratio errors
            ratio_yvalues, ratio_line, ratio_errors = get_ratio_errors(fit_parameters_small, fit_parameters_large, cov_matrix_small, cov_matrix_large, temperature_ints_shifted)
            ax_ratio.plot(temperature_ints_shifted, ratio_line, c=ratio_colors['ratio_1'], label='ratio {}mm/{}mm'.format(separations[1], separations[0]), linewidth=0.75)
            ax_ratio.errorbar(temperature_ints_shifted, ratio_yvalues, ratio_errors, ls='none', color=ratio_error_colors['ratio_1'], barsabove=True, zorder=3)


    elif len(separations) == 3:

        # Create the separate dataframes for each individual line, saving them to a list of all three
        # dataframes = [create_df(df_all_data, sep, bias_voltage) for sep in separations]

        dataframes = []
        for sep in separations:
            dataframe = create_df(df_all_data, sep, bias_voltage)
            if sep == '27':
                dataframe = dataframe[dataframe['midpoint'] > 1.0]
            dataframes.append(dataframe)

        # Call the function to plot the data, and return the fit parameters, cov matrix, and the text string
        parameter_labels = [['a', 'b'], ['c', 'd'], ['e', 'f']]     # List of the differnet parameter labels to use for the text string
        par_dict = dict()       # The dictionary the outputs in the plot_data function will be saved to 
        for i, sep in enumerate(separations):
            fit_parameters, cov_matrix, txt_str = plot_data(ax_data, parameter_labels[i][0], parameter_labels[i][1], dataframes[i], sep, bias_voltage, data_colors[sep], data_error_colors[sep], label='{}mm'.format(sep))
            par_dict[sep] = {'fit_parameters': fit_parameters, 'cov_matrix': cov_matrix, 'txt_str': txt_str}

        # Setting the plot title variable
        dates = []
        for i, sep in enumerate(separations):
            date_string = ', '.join(dataframes[i].date.unique())
            dates.append('{}mm: {}'.format(sep, date_string))
        plot_title = '     '.join(dates)

        # Setting the positions of the text on the figure
        plt.figtext(0.78, 0.5, par_dict[separations[0]]['txt_str'], color=data_colors[separations[0]], fontsize=10)
        plt.figtext(0.78, 0.3, par_dict[separations[1]]['txt_str'], color=data_colors[separations[1]], fontsize=10)
        plt.figtext(0.78, 0.1, par_dict[separations[2]]['txt_str'], color=data_colors[separations[2]], fontsize=10)

        # Plot the ratios if the plot_ratios variable is set to True
        if plot_ratios:

            ax_ratio = ax_data.twinx()
            ax_ratio.set_position((0.1, 0.1, 0.6, 0.8))

            # Find and plot the ratios and the ratio errors
            for a, b in combinations(separations, 2):       # Iterate through every combination of two separation values - a & b represent two possible separation values.
                ratio_yvalues_ab, ratio_line_ab, ratio_errors_ab = get_ratio_errors(par_dict[a]['fit_parameters'], par_dict[b]['fit_parameters'], par_dict[a]['cov_matrix'], par_dict[b]['cov_matrix'], temperature_ints_shifted)
                ax_ratio.plot(temperature_ints_shifted, ratio_line_ab, c=ratio_colors['ratio_1'], label='ratio {}mm/{}mm'.format(b, a), linewidth=0.75)
                ax_ratio.errorbar(temperature_ints_shifted, ratio_yvalues_ab, ratio_errors_ab, ls='none', color=ratio_error_colors['ratio_1'], barsabove=True, zorder=3)

        # =============================================================

        # # Create the separate dataframes for each individual line
        # df_27mm = create_df(df_all_data, separations[0], bias_voltage)
        # df_31mm = create_df(df_all_data, separations[1], bias_voltage)
        # df_38mm = create_df(df_all_data, separations[2], bias_voltage)

        # # Call the function to plot the data, and return the fit parameters, cov matrix, and text string
        # fit_parameters_27mm, cov_matrix_27mm, txt_str_27mm = plot_data(ax_data, 'a', 'b', df_27mm, separations[0], bias_voltage, data_colors[separations[0]], data_error_colors[separations[0]], label='{}mm'.format(separations[0]))
        # fit_parameters_31mm, cov_matrix_31mm, txt_str_31mm = plot_data(ax_data, 'c', 'd', df_31mm, separations[1], bias_voltage, data_colors[separations[1]], data_error_colors[separations[1]], label='{}mm'.format(separations[1]))
        # fit_parameters_38mm, cov_matrix_38mm, txt_str_38mm = plot_data(ax_data, 'e', 'f', df_38mm, separations[2], bias_voltage, data_colors[separations[2]], data_error_colors[separations[2]], label='{}mm'.format(separations[2]))

        # # Setting the plot title variable
        # date_27mm = ", ".join(df_27mm.date.unique())
        # date_31mm = ", ".join(df_31mm.date.unique())
        # date_38mm = ", ".join(df_38mm.date.unique())
        # plot_title = '27mm: {}     31mm: {}     38mm: {}'.format(date_27mm, date_31mm, date_38mm)

        # Setting the positions of the text on the figure
        # plt.figtext(0.78, 0.5, d[separations[0]]['txt_str'], color=data_colors[separations[0]], fontsize=10)
        # plt.figtext(0.78, 0.25, d[separations[1]]['txt_str'], color=data_colors[separations[1]], fontsize=10)
        # plt.figtext(0.78, 0.1, d[separations[2]]['txt_str'], color=data_colors[separations[2]], fontsize=10)

        # if plot_ratios:

        #     ax_ratio = ax_data.twinx()
        #     ax_ratio.set_position((0.1, 0.1, 0.6, 0.8))

            # # Find and plot the ratios and the ratio errors
            # ratio_yvalues_2731, ratio_line_2731, ratio_errors_2731 = get_ratio_errors(fit_parameters_27mm, fit_parameters_31mm, cov_matrix_27mm, cov_matrix_31mm, temperature_ints_shifted)
            # ratio_yvalues_2738, ratio_line_2738, ratio_errors_2738 = get_ratio_errors(fit_parameters_27mm, fit_parameters_38mm, cov_matrix_27mm, cov_matrix_38mm, temperature_ints_shifted)
            # ratio_yvalues_3138, ratio_line_3138, ratio_errors_3138 = get_ratio_errors(fit_parameters_31mm, fit_parameters_38mm, cov_matrix_31mm, cov_matrix_38mm, temperature_ints_shifted)

            # ax_ratio.plot(temperature_ints_shifted, ratio_line_2731, c=ratio_colors['ratio_1'], label='ratio {}mm/{}mm'.format(separations[1], separations[0]), linewidth=0.75)
            # ax_ratio.plot(temperature_ints_shifted, ratio_line_2738, c=ratio_colors['ratio_2'], label='ratio {}mm/{}mm'.format(separations[2], separations[0]), linewidth=0.75)
            # ax_ratio.plot(temperature_ints_shifted, ratio_line_3138, c=ratio_colors['ratio_3'], label='ratio {}mm/{}mm'.format(separations[2], separations[1]), linewidth=0.75)

            # ax_ratio.errorbar(temperature_ints_shifted, ratio_yvalues_2731, ratio_errors_2731, ls='none', color=ratio_error_colors['ratio_1'], barsabove=True, zorder=3)
            # ax_ratio.errorbar(temperature_ints_shifted, ratio_yvalues_2738, ratio_errors_2738, ls='none', color=ratio_error_colors['ratio_2'], barsabove=True, zorder=3)
            # ax_ratio.errorbar(temperature_ints_shifted, ratio_yvalues_3138, ratio_errors_3138, ls='none', color=ratio_error_colors['ratio_3'], barsabove=True, zorder=3)

    #==========================================================================================================

    ### Plot Settings ###

    # Setting the y range
    ax_data.set_ylim(*y_range_data)

    # Setting the axis labels
    ax_data.set_xlabel('Temperature [K]', fontsize=14)
    ax_data.set_ylabel('Midpoint [V]', fontsize=14)

    # Label the x-ticks with the actual temperature values (166-172)
    for ax in axes:
        locs = ax.get_xticks()
        adjusted_locs = [str(int(l+169)) for l in locs]
        ax.set_xticklabels(adjusted_locs)

    # Setting the super title and the title
    plt.suptitle(plot_suptitle)
    plt.title(plot_title)

    # Settings for the ratio plot
    if plot_ratios:
        ax_ratio.set_ylim(y_range_ratio)
        ax_ratio.set_ylabel('Ratio', fontsize=14)
        ax_ratio.legend(bbox_to_anchor=(1.436, 0.9))

    plt.grid(True)
    ax_data.legend(bbox_to_anchor=(1.3, 1.0))
    plt.show()