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

temperature_values = np.array([172])
temperature_values_adjusted = temperature_values-169

# Plot Color
line_color = '#B073FF'
error_color = '#7800BF'
txt_color = '#E64676'
fill_color = '#E9D9FF'

#==================================================================================================
### Functions

def const_func(p, x):
    # print(p, x)
    (a, b) = p
    return b*np.ones_like(x)

def get_constant_fit(x, y, y_err):
    global odr
    model_func = odr.Model(const_func)
    # print(x)
    # print(y)
    data = odr.RealData(x, y, sy=y_err)
    odr_object = odr.ODR(data, model_func, beta0=[0, 0], taufac=1E-5, partol=1E-5, maxit=10000000)

    out = odr_object.run()
    optimized_val = out.beta
    val_error = out.sd_beta

    return optimized_val, val_error

def plot_settings(separation_1, separation_2, df_1, df_2):
    # Set the axis labels
    ax.set_xlabel('Bias Voltage [V]', fontsize=14)
    ax.set_ylabel('Ratio', fontsize=14)

    # Setting the super title and the title
    plt.suptitle('Ratio of Best Fit Lines at {}mm and {}mm vs. Bias Voltage at {}K'.format(separation_2, separation_1, temperature_values[0]))

    date_1 = ", ".join(df_1.date.unique())
    date_2 = ", ".join(df_2.date.unique())

    plt.title('{}mm: {}   {}mm: {}'.format(separation_1, date_1, separation_2, date_2))

    plt.grid(True)
    ax.legend(bbox_to_anchor=(1.4, 1.0))
    # ax.legend()
    plt.show()

#==================================================================================================
### Execute Functions

if __name__ == '__main__':
    
    # Setting the width and height of the figure
    fig, ax = plt.subplots()
    fig.set_figwidth(9)
    fig.set_figheight(6)

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

    bias_voltage_nums = np.array([int(x[:-1]) for x in bias_voltages])
    ratios_bv_nums = np.array(ratios_bv).T[0]
    ratio_errors_bv_nums = np.array(ratio_errors_bv).T[0]
    # print(bias_voltage_nums, ratios_bv_nums, ratio_errors_bv_nums)

    opt_val, val_err = get_constant_fit(bias_voltage_nums, ratios_bv_nums, ratio_errors_bv_nums)
    print('opt val =', opt_val, 'val err =', val_err)

    # Get the average of all the points
    ratios_bv = np.array(ratios_bv)
    average_ratio = np.average(ratios_bv)
    average_list = np.empty(len(ratios_bv))
    average_list.fill(average_ratio)

    # Get the standard deviation of all the points 
    standard_dev = np.std(ratios_bv)
    print(standard_dev)
    upper_bound = average_list + standard_dev
    lower_bound = average_list - standard_dev

    ax.set_ylim(0.51, 0.545)

    ax.fill_between(bias_voltages, upper_bound, lower_bound, color=fill_color)
    ax.scatter(bias_voltages, ratios_bv, color=line_color, label='Ratio at {}K'.format(temperature_values[0]))
    ax.plot(bias_voltages, average_list, color=line_color, label='Average Ratio')
    ax.plot(bias_voltages, upper_bound, color=error_color, label='Standard Deviation', ls='--', linewidth=0.9)
    ax.plot(bias_voltages, lower_bound, color=error_color, ls='--', linewidth=0.9)
    ax.errorbar(bias_voltages, ratios_bv, ratio_errors_bv, ls='none', color=error_color, barsabove=True, zorder=3)

    txt_str = '\n'.join(['Average ratio: {:.4f}'.format(average_ratio), ' ', 'Standard Deviation: {:.4f}'.format(standard_dev)])
    
    # Setting the position of the text on the figure
    ax.set_position((0.1, 0.1, 0.6, 0.8))
    plt.figtext(0.75, 0.5, txt_str, fontsize=10)

    plot_settings('27', '38', df_27mm, df_38mm)