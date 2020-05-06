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
from Mid_vs_Temp_Ratios import analyze_data

#==================================================================================================
### Variables

alphas_filename = 'alphas.h5'
date_list = ['20190516', '20190424']
bias_voltages = ['47V', '48V', '49V', '50V', '51V', '52V']

temperature_values = np.array([169])
temperature_values_adjusted = temperature_values-169

# Plot Color
line_color = '#BE8CFF'
error_color = '#7800BF'
txt_color = '#E64676'

#==================================================================================================
### Functions

def plot_settings(separation_1, separation_2, df_1, df_2):
    # Set the axis labels
    ax.set_xlabel('Temperature [K]', fontsize=14)
    ax.set_ylabel('Bias Voltage [V]', fontsize=14)

    # Setting the super title and the title
    plt.suptitle('Ratio of Best Fit Lines at {}mm and {}mm vs. Bias Voltage'.format(separation_2, separation_1))

    date_1 = ", ".join(df_1.date.unique())
    date_2 = ", ".join(df_2.date.unique())

    plt.title('{}mm: {}   {}mm: {}'.format(separation_1, date_1, separation_2, date_2))

    plt.grid(True)
    # ax.legend()
    plt.show()

#==================================================================================================
### Execute Functions

if __name__ == '__main__':
    
    fig, ax = plt.subplots()

    # Create the dataframes
    df_dates = create_df(alphas_filename)

    ratios_bv = []
    ratio_errors_bv = []
    for voltage in bias_voltages:

        # Get the separated dataframes
        df_27mm = get_final_df(df_dates, '27', voltage)
        df_38mm = get_final_df(df_dates, '38', voltage)

        # Deletes the first row in the data frame if the bias voltage is 48V
        if voltage == '48V':
            df_27mm = df_27mm.iloc[1:]
        
        # Analyze the data - get optimized parameters, parameter errors, and the covariance matrix
        optimized_parameters_27mm, parameter_errors_27mm, cov_matrix_27mm = analyze_data(df_27mm)
        optimized_parameters_38mm, parameter_errors_38mm, cov_matrix_38mm = analyze_data(df_38mm)

        # Get the ratio points at 169K
        ratio_point = get_ratio_line(optimized_parameters_27mm, optimized_parameters_38mm, temperature_values_adjusted)
        ratios_bv.append(ratio_point)

        # Get the ratio errors at 169K
        fit_parameters_27mm, fit_parameters_38mm, ratio_points, ratio_errors = get_ratio_errors(optimized_parameters_27mm,optimized_parameters_38mm, parameter_errors_27mm, parameter_errors_38mm, cov_matrix_27mm, cov_matrix_38mm, temperature_values_adjusted)
        ratio_errors_bv.append(ratio_errors)

    ax.scatter(bias_voltages, ratios_bv, color=line_color)
    ax.errorbar(bias_voltages, ratios_bv, ratio_errors_bv, ls='none', color=error_color, barsabove=True, zorder=3)
    plot_settings('27', '38', df_27mm, df_38mm)