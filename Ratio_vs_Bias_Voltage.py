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

from Midpoint_vs_Temp_Plotting import create_df, get_final_df, line_func, get_fit_parameters, get_ratio_errors

#==================================================================================================
### Variables

alphas_filename = 'alphas.h5'
date_list = ['20190516', '20190424']
bias_voltages = ['47V', '48V', '49V', '50V', '51V', '52V']

temperature_value = np.array([172])
temperature_value_adjusted = temperature_value-169

# Plot Colors
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
    data = odr.RealData(x, y, sy=y_err)
    odr_object = odr.ODR(data, model_func, beta0=[0, 0], taufac=1E-5, partol=1E-5, maxit=10000000)

    out = odr_object.run()
    optimized_val = out.beta
    val_error = out.sd_beta

    return optimized_val, val_error

def analyze_data(df):
    x = df[['temperature_avg']].values[:, 0]
    x_adjusted = x-169

    y = df[['midpoint']].values[:, 0]
    x_error = df[['temperature_rms']].values[:, 0]
    y_error = df[['midpt_error']].values[:, 0]

    fit_parameters, cov_matrix = get_fit_parameters(x_adjusted, y, x_error, y_error)
    optimized_parameters = fit_parameters[0]
    best_fit_line = line_func(optimized_parameters, temperature_value_adjusted)

    return fit_parameters, best_fit_line, cov_matrix

#==================================================================================================
### Execute Functions

if __name__ == '__main__':

    # Setting the width and height of the figure
    fig, ax = plt.subplots()
    fig.set_figwidth(9)
    fig.set_figheight(6)

    # Create the dataframes
    df_dates = create_df(alphas_filename)

    ratio_list = []
    ratio_error_list = []

    for voltage in bias_voltages:
        
        # Get the separated dataframes
        df_27mm = get_final_df(df_dates, '27', voltage)
        df_38mm = get_final_df(df_dates, '38', voltage)

        # Deletes the first row in the data frame if the bias voltage is 48V and the separation is 27mm.
        if voltage == '48V':
            df_27mm = df_27mm.iloc[1:]
        
        # Analyze the data - get the fit parameters, the best fit line, and the covariance matrix
        fit_parameters_27mm, best_fit_line_27mm, cov_matrix_27mm = analyze_data(df_27mm)
        fit_parameters_38mm, best_fit_line_38mm, cov_matrix_38mm = analyze_data(df_38mm)

        # Get the ratio point and the ratio error at the given temperature value
        ratio_point, ratio_error = get_ratio_errors(fit_parameters_27mm, fit_parameters_38mm, best_fit_line_27mm, best_fit_line_38mm, cov_matrix_27mm, cov_matrix_38mm, temperature_value_adjusted)
        ratio_list.append(ratio_point)
        ratio_error_list.append(ratio_error)

    # Convert all of the lists into numpy arrays
    bias_voltages = np.array([int(x[:-1]) for x in bias_voltages])
    ratio_list = np.array(ratio_list).T[0]
    ratio_error_list = np.array(ratio_error_list).T[0]

    # Get the value and the error of the constant fit
    optimized_value, value_error = get_constant_fit(bias_voltages, ratio_list, ratio_error_list)
    print(optimized_value, value_error)

    # Get the average of all of the ratio vs. bias voltage points
    ratio_list = np.array(ratio_list)
    average_ratio = np.average(ratio_list)
    average_list = np.empty(len(ratio_list))
    average_list.fill(average_ratio)

    # Get the standard deviation of all of the ratio vs. bias voltage points
    standard_dev = np.std(ratio_list)
    print(standard_dev)
    upper_bound = average_list + standard_dev
    lower_bound = average_list - standard_dev

    #==============================================================================================
    ## Plotting the data

    # Set the limits of the y-axis
    ax.set_ylim(0.51, 0.545)

    # Plot the ratio vs. bias voltage points, error bars, the average line, standard deviation lines, and the filling within the standard deviation
    ax.fill_between(bias_voltages, upper_bound, lower_bound, color=fill_color)
    ax.scatter(bias_voltages, ratio_list, color=line_color, label='Ratio at {}K'.format(temperature_value[0]))
    ax.plot(bias_voltages, average_list, color=line_color, label='Average Ratio')
    ax.plot(bias_voltages, upper_bound, color=error_color, label='Standard Deviation', ls='--', linewidth=0.9)
    ax.plot(bias_voltages, lower_bound, color=error_color, ls='--', linewidth=0.9)
    ax.errorbar(bias_voltages, ratio_list, ratio_error_list, ls='none', color=error_color, barsabove=True, zorder=3)

    # Set the text strings to go to the right of the plot
    txt_str_avg_std = '\n'.join(['Average ratio: {:.4f}'.format(average_ratio), ' ', 'Standard Deviation: {:.4f}'.format(standard_dev)])
    txt_str_fit = '\n'.join(['Result of Fit:', ' ', 'y = {:.4f}'.format(optimized_value[1]), ' ', 'Error on y = {:.4f}'.format(value_error[1])])
    
    # Setting the position of the text on the figure
    ax.set_position((0.1, 0.1, 0.6, 0.8))
    plt.figtext(0.75, 0.6, txt_str_avg_std, fontsize=10)
    plt.figtext(0.75, 0.4, txt_str_fit, fontsize=10)

    #==============================================================================================
    ## Plot Settings

    # Set the axis labels 
    ax.set_xlabel('Bias Voltage [V]', fontsize=14)
    ax.set_ylabel('Ratio', fontsize=14)

    # Setting the super title and the title
    plt.suptitle('Ratio of the Best Fit Lines at 27mm and 38mm vs. Bias Voltage at {}K'.format(temperature_value[0]))

    date_27mm = ", ".join(df_27mm.date.unique())
    date_38mm = ", ".join(df_38mm.date.unique())

    plt.title('27mm: {}   38mm: {}'.format(date_27mm, date_38mm))

    plt.grid(True)
    ax.legend(bbox_to_anchor=(1.4, 1.0))

    filename = 'Ratio vs. Bias Voltage at {}K with fit parameters.png'.format(temperature_value[0])
    fig.savefig(filename, dpi=200)
    # ax.legend()
    # plt.show()