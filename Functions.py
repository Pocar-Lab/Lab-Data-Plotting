############################################### Functions #################################################
# This script only contains functions to be used by other plotting scripts. 
# This file should remain unchanged unless there is a bug felt throughout most or all other scripts.
# Note: not all scripts will use all of these functions.
#==========================================================================================================

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

#==========================================================================================================
### Functions ###

# Uses pandas to compile all of the data found in alphas.h5 into a data frame.
# This makes it easier to work with and identify parts we want to analyze.
def compile_data(filename, dates):
    with pd.HDFStore(filename, 'r') as hdf:
        i = 0
        for key in hdf.keys():
            if i == 0:
                whole_df = hdf[key]
            else:
                whole_df = whole_df.append(hdf[key])
            i += 1
    
    df_cols = whole_df[['date', 'separation', 'biasV', 'midpoint', 'midpt_error', 'sigma', 'sigma_error', 
    'temperature_avg', 'temperature_rms']]
    df_all_dates = df_cols[df_cols['date'].isin(dates)]

    return df_all_dates

# Creates the data frame only containing the data with the specified conditions. 
# These conditions include date taken, separation, and bias voltage.
def create_df(df, separation, voltage):
    sep = df['separation'] == separation
    bv = df['biasV'] == voltage
    df = df.loc[sep & bv]
    return df

# Linear function (this is the model function for the best fit line).
def linear_func(p, x):
    (a, b) = p
    return a*x + b

# Get the slope (a) and the intercept (b) (aka optimized parameters) that best fit the data.
# Also returns the errors on the best fit parameters, and the covariance matrix.
def get_fit_parameters(x, y, x_error, y_error):
    global odr
    model_func = odr.Model(linear_func)
    data = odr.RealData(x, y, sx=x_error, sy=y_error)
    odr_object = odr.ODR(data, model_func, beta0=[40, 0.05], taufac=1E-5, partol=1E-5, maxit=10000000)

    out = odr_object.run()
    optimized_parameters = out.beta
    parameter_errors = out.sd_beta
    cov_matrix = out.cov_beta

    return (optimized_parameters, parameter_errors), cov_matrix

# Calculate the reduced chi-squared value in only one dimension (uses y-errors only)
def calc_red_chisquare_1d(opt_parameters, x_vals, y_vals, y_errors):
    y_expected = linear_func(opt_parameters, x_vals)

    num_parameters = 2
    degrees_of_freedom = len(y_vals)-num_parameters

    chisquare = np.sum(((y_expected-y_vals)/y_errors)**2)
    red_chisquare_1d = chisquare/degrees_of_freedom

    return red_chisquare_1d

# Calculate the reduced chi-squared value in two dimensions (uses x and y-errors)
def calc_red_chisquare_2d(opt_parameters, x_vals, y_vals, x_errors, y_errors):
    y_expected = linear_func(opt_parameters, x_vals)

    num_parameters = 2
    degrees_of_freedom = len(y_vals)-num_parameters
    a = opt_parameters[0]
    b = opt_parameters[1]

    chisquare = np.sum((y_vals-a*x_vals-b)**2/((a**2)*(x_errors**2)+(y_errors**2)))
    red_chisquare_2d = chisquare/degrees_of_freedom

    return red_chisquare_2d

# Define the x and y values of the plot using selected columns in the dataframe
def define_xy_values(df, x_column, y_column, x_error_column, y_error_column):
    x = df[[x_column]].values[:, 0]
    y = df[[y_column]].values[:, 0]
    x_error = df[[x_error_column]].values[:, 0]
    y_error = df[[y_error_column]].values[:, 0]

    return x, y, x_error, y_error

# Find the errors on the ratio by propagating the errors on the best fit lines
def get_ratio_errors(fit_parameters_1, fit_parameters_2, cov_matrix_1, cov_matrix_2, temperatures):

    # Variables representing the slope, intercept, and the errors on the slope and intercept
    optimized_parameters_1 = fit_parameters_1[0]
    (slope_error_1, intercept_error_1) = fit_parameters_1[1]
    optimized_parameters_2 = fit_parameters_2[0]
    (slope_error_2, intercept_error_2) = fit_parameters_2[1]

    # Find the ratio line
    best_fit_line_1 = linear_func(optimized_parameters_1, temperatures)
    best_fit_line_2 = linear_func(optimized_parameters_2, temperatures)
    ratio_line = best_fit_line_2/best_fit_line_1

    # Initializing lists for the expected voltages and ratio errors
    expected_voltages_1 = []
    expected_voltages_2 = []
    ratio_errors = []

    # Loop through each integer temperature value, calculating the ratio error at that point
    for temperature in temperatures:

        # Expected values at separation 1
        expected_voltage_1 = linear_func(optimized_parameters_1, temperature)
        expected_voltages_1.append(expected_voltage_1)

        # Expected values at separation 2
        expected_voltage_2 = linear_func(optimized_parameters_2, temperature)
        expected_voltages_2.append(expected_voltage_2)

        # Correlation parmeters (rho)
        rho_1 = cov_matrix_1[0][1] / ( np.sqrt(cov_matrix_1[0][0]) * np.sqrt(cov_matrix_1[1][1]) )
        rho_2 = cov_matrix_2[0][1] / ( np.sqrt(cov_matrix_2[0][0]) * np.sqrt(cov_matrix_2[1][1]) )

        # Errors on the best fit line at the current temperature value 
        best_fit_error_1 = np.sqrt( temperature**2 * slope_error_1**2 + intercept_error_1**2 + (2*rho_1 * slope_error_1 * temperature * intercept_error_1) )
        best_fit_error_2 = np.sqrt( temperature**2 * slope_error_2**2 + intercept_error_2**2 + (2*rho_2 * slope_error_2 * temperature * intercept_error_2) )

        # Calculate the ratio errors
        ratio_error = (expected_voltage_2 / expected_voltage_1) * np.sqrt( (best_fit_error_2/expected_voltage_2)**2 + (best_fit_error_1/expected_voltage_1)**2 )
        ratio_errors.append(ratio_error)

    # Get the y-values for the ratio erros
    ratio_yvals = np.divide(expected_voltages_2, expected_voltages_1)

    return ratio_yvals, ratio_line, ratio_errors

#==========================================================================================================