#==================================================================================================

import h5py
import csv
import pandas as pd

import numpy as np
from numpy import sqrt

import scipy
from scipy import odr

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from Midpoint_vs_Temp_Plotting import create_df, get_final_df, line_func, get_fit_parameters, get_ratio_line, get_ratio_errors

#==================================================================================================
### Variables

alphas_filename = 'alphas.h5'
date_list = ['20190516', '20190424']
separation_list = ['27', '38']
bias_voltages = ['47V', '48V', '49V', '50V', '51V', '52V']

temperature_values = np.array([166, 167, 168, 169, 170, 171, 172])
temperature_values_adjusted = temperature_values-169

# Dictionary that sets the color of the line based on the value of the bias voltage
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
### Functions

def analyze_data(df):
    x_range = np.array(range(166, 173))

    x = df[['temperature_avg']].values[:, 0]
    x_adjusted = x-169
    x_range_adjusted = x_range-169

    y = df[['midpoint']].values[:, 0]
    x_error = df[['temperature_rms']].values[:, 0]
    y_error = df[['midpt_error']].values[:, 0]

    optimized_parameters, parameter_errors, cov_matrix = get_fit_parameters(x_adjusted, y, x_error, y_error)

    return optimized_parameters, parameter_errors, cov_matrix

def plot_settings(separation_1, separation_2, df_1, df_2):
    # Set the axis labels
    ax.set_xlabel('Temperature [K]', fontsize=14)
    ax.set_ylabel('Ratio', fontsize=14)

    # Setting the super title and the title
    plt.suptitle('Ratio of Best Fit Lines at {}mm and {}mm'.format(separation_2, separation_1))

    date_1 = ", ".join(df_1.date.unique())
    date_2 = ", ".join(df_2.date.unique())

    plt.title('{}mm: {}   {}mm: {}'.format(separation_1, date_1, separation_2, date_2))

    # Label the x-ticks with the actual temperature values (166-172)
    locs = ax.get_xticks()
    adjusted_locs = [str(int(l+169)) for l in locs]
    ax.set_xticklabels(adjusted_locs)

    plt.grid(True)
    ax.legend()
    plt.show()

#==================================================================================================
### Execute Functions

if __name__ == '__main__':

    fig, ax = plt.subplots()

    # Create the dataframes
    df_dates = create_df(alphas_filename)

    for voltage in bias_voltages:

        # Get the separated dataframes
        df_27mm = get_final_df(df_dates, '27', voltage)
        df_38mm = get_final_df(df_dates, '38', voltage)

        # Deletes the first row in the data frame if the bias voltage is 48V
        if voltage == '48V':
            df_27mm = df_27mm.iloc[1:]

        # Analyze the data - get optimized parameters, parameter errors, and covariance matrix
        optimized_parameters_27mm, parameter_errors_27mm, cov_matrix_27mm = analyze_data(df_27mm)
        optimized_parameters_38mm, parameter_errors_38mm, cov_matrix_38mm = analyze_data(df_38mm)

        # Plot the ratio line
        ratio_line = get_ratio_line(optimized_parameters_27mm, optimized_parameters_38mm, temperature_values_adjusted)
        ax.plot(temperature_values_adjusted, ratio_line, c=colors[voltage], label=voltage)

        # Plot the ratio errors
        fit_parameters_27mm, fit_parameters_38mm, ratio_points, ratio_errors = get_ratio_errors(optimized_parameters_27mm, optimized_parameters_38mm, parameter_errors_27mm, parameter_errors_38mm, cov_matrix_27mm, cov_matrix_38mm, temperature_values_adjusted)
        ax.errorbar(temperature_values_adjusted, ratio_line, ratio_errors, ls='none', color=error_colors[voltage], barsabove=True, zorder=3)

    # Plot settings
    plot_settings('27', '38', df_27mm, df_38mm)