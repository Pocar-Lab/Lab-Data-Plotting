import h5py
import csv
import pandas as pd

import numpy as np
from numpy import sqrt, exp
import scipy
from scipy import odr
from scipy.stats import chisquare

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#==================================================================================================
### Variables: Set these to indicate the desired date, separation, and bias voltage for the plot

date_list = ['20190516', '20190424']
separation_list = ['27', '38']
bias_voltages = ['47V', '48V', '49V', '50V', '51V', '52V']
# bias_voltages = ['47V']

temperature_values = np.array([166, 167, 168, 169, 170, 171, 172])

# A dictionary that sets the color of the line based on the value of the bias voltage
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

fig, ax = plt.subplots()

#==================================================================================================
### Functions: These are all of the functions that will be needed in order to create a complete plot

# Extracting the bias voltage
def get_final_df(df, separation, voltage):
    sep = df['separation'] == separation
    bv = df['biasV'] == voltage
    df_final = df_dates.loc[sep & bv]
    return df_final

# Finding the best fit line
# Model function:
def line_func(p, x):
    (a, b) = p
    # assert type(a) == float or type(a) == np.float or type(a) == np.float64, f"{type(a)} {a}"
    # assert type(x) == np.array or type(x) == np.ndarray, f"{type(x)} {x}"
    return a*x + b

# Get the a and b parameters that best fit the data:
def get_fit_parameters(x, y, x_err, y_err,):
    global odr
    model_func = odr.Model(line_func)
    data = odr.RealData(x, y, sx=x_err, sy=y_err)
    odr_object = odr.ODR(data, model_func, beta0=[40, 0.05], taufac=1E-5, sstol=1E-5, partol=1E-5, maxit=10000000)
    out = odr_object.run()
    optimized_parameters = out.beta
    parameter_errors = out.sd_beta
    cov_matrix = out.cov_beta
    y_fit = line_func(optimized_parameters, x)

    return optimized_parameters, parameter_errors, cov_matrix
'''
# Calculate the reduced chi-squared value:
def calc_reduced_chisquare(opt_params, x_vals, y_vals, y_errors):
    y_exp = line_func(opt_params, np.array(x_vals))
    len_y_data = len(y_vals)
    num_parameters = 2
    chisquare = np.sum(((y_exp-y_vals)/y_errors)**2)
    reduced_chisquare = chisquare/(len_y_data-num_parameters)
    return reduced_chisquare
'''

# Defines the x values, y values, x errors, and y errors in the data. 
# Finds the optimized parameters for the best fit line of the data, and returns the optimized parameters, parameter error, and the covariance matrix. 
def analyze_data(df):
    x_range = np.array(range(166, 173))

    x = df[['temperature_avg']].values[:, 0]
    x_adjusted = x-169
    x_range_adjusted = x_range-169

    y = df[['midpoint']].values[:, 0]
    x_error = df[['temperature_rms']].values[:, 0]
    y_error = df[['midpt_error']].values[:, 0]
    
    optimized_parameters, parameter_errors, cov_matrix = get_fit_parameters(x_adjusted, y, x_error, y_error)
    # best_fit_line = line_func(optimized_parameters, x_range_adjusted)
    
    return optimized_parameters, parameter_errors, cov_matrix

#==================================================================================================

#==================================================================================================
### Extract all of the information needed for the whole plot

# Use pandas to create a dataframe using all of the data in alphas.h5
# The result is a big table of data that looks like the alpha frame list in the app.
alphas_filename = 'alphas.h5'

with pd.HDFStore(alphas_filename, 'r') as hdf:
    i = 0
    for key in hdf.keys():
        if i == 0:
            whole_df = hdf[key]
        else: 
            whole_df = whole_df.append(hdf[key])
        i += 1

# Extract all of the columns needed 
df_cols = whole_df[['date', 'separation', 'biasV', 'midpoint', 'midpt_error', 'temperature_avg', 'temperature_rms']]

# Extract all of the dates needed - the resulting dataframe will include every date in date_list
df_dates = df_cols[df_cols['date'].isin(date_list)]

#==================================================================================================
# The following is a loop that iterates through all six bias voltages, plotting the ratio line for each bias voltage.

for voltage in bias_voltages:
    # Get the final dataframe
    df_27mm = get_final_df(df_dates, '27', voltage)
    df_38mm = get_final_df(df_dates, '38', voltage)

    # Analyze the data - get the optimized parameters, parameter errors, and the covariance matrix
    optimized_parameters_27mm, parameter_errors_27mm, cov_matrix_27mm = analyze_data(df_27mm)
    optimized_parameters_38mm, parameter_errors_38mm, cov_matrix_38mm = analyze_data(df_38mm)
    
    # List of integer temperature values
    temperature_values = np.array([166, 167, 168, 169, 170, 171, 172])
    temperature_values_adjusted = temperature_values-169

    # Variables representing the slope, intercept, and errors on the slope and intercept for both separations
    (slope_27mm, intercept_27mm) = optimized_parameters_27mm
    (slope_38mm, intercept_38mm) = optimized_parameters_38mm
    (slope_error_27mm, intercept_error_27mm) = parameter_errors_27mm
    (slope_error_38mm, intercept_error_38mm) = parameter_errors_38mm

    # Find the best fit lines for each separation
    best_fit_line_27mm = line_func(optimized_parameters_27mm, temperature_values_adjusted)
    best_fit_line_38mm = line_func(optimized_parameters_38mm, temperature_values_adjusted)
    print('27', best_fit_line_27mm)
    print('38', best_fit_line_38mm)

    # Find the ratio line by taking the ratio of the best fit lines
    ratio_line = best_fit_line_38mm/best_fit_line_27mm

    # Find and append expected values to different lists based on separation. The loop calculates the expected midpoint value based on the integer temperature value.
    expected_voltages_27mm = []
    expected_voltages_38mm = []
    ratio_errors = []
    for temperature in temperature_values_adjusted:
        # Expected values at 27mm
        expected_voltage_27mm = slope_27mm*temperature + intercept_27mm
        expected_voltages_27mm.append(expected_voltage_27mm)

        # Expected value at 38mm
        expected_voltage_38mm = slope_38mm*temperature + intercept_38mm
        expected_voltages_38mm.append(expected_voltage_38mm)

        # Correlation parameters (rho)
        rho_27mm = cov_matrix_27mm[0][1] / ( np.sqrt(cov_matrix_27mm[0][0]) * np.sqrt(cov_matrix_27mm[1][1]) )
        rho_38mm = cov_matrix_38mm[0][1] / ( np.sqrt(cov_matrix_38mm[0][0]) * np.sqrt(cov_matrix_38mm[1][1]) )

        # Best fit at the specified temperature value
        best_fit_27mm = line_func(optimized_parameters_27mm, temperature)
        best_fit_38mm = line_func(optimized_parameters_38mm, temperature)

        # Errors on the best fit lines
        best_fit_error_27mm = np.sqrt( temperature**2 * slope_error_27mm**2 + intercept_error_27mm**2 + 2*rho_27mm*slope_error_27mm*temperature*intercept_error_27mm)
        best_fit_error_38mm = np.sqrt( temperature**2 * slope_error_38mm**2 + intercept_error_38mm**2 + 2*rho_38mm*slope_error_38mm*temperature*intercept_error_38mm)

        # Calculate the ratio errors
        ratio_error = (best_fit_38mm/best_fit_27mm) * np.sqrt( (best_fit_error_38mm/best_fit_38mm)**2 + (best_fit_error_27mm/best_fit_27mm)**2 )
        ratio_errors.append(ratio_error)
        # print(ratio_error)

    # ratio_points = np.divide(expected_voltages_38mm, expected_voltages_27mm)

    ax.plot(temperature_values_adjusted, ratio_line, c=colors[voltage], label=voltage)
    ax.errorbar(temperature_values_adjusted, ratio_line, ratio_errors, ls='none', color=error_colors[voltage], barsabove=True, zorder=3)
    # print(voltage, ratio_points, ratio_line, ratio_errors)
    # exit()
#==================================================================================================
### Plot Settings

# ax.set_ylim(0.50, 0.55)

ax.set_xlabel('Temperature [K]', fontsize=14)
ax.set_ylabel('Ratio', fontsize=14)
plt.suptitle('Ratio of Best Fit Lines at 38mm and 27mm')

date_27mm = ", ".join(df_27mm.date.unique())
date_38mm = ", ".join(df_38mm.date.unique())

plt.title('27mm: {}   38mm: {}'.format(date_27mm, date_38mm))

# Label the x ticks with the actual temperature values (166-172)
locs = ax.get_xticks()
adjusted_locs = [str(int(l+169)) for l in locs]
ax.set_xticklabels(adjusted_locs)

plt.grid(True)
ax.legend()
plt.show()