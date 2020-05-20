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

#==================================================================================================
### Variables

alphas_filename = 'alphas.h5'
date_list = ['20190228', '20190212', '20190207', '20190124', '20181212', '20181211', '20181210', '20181204', '20181119']

temperature_values = np.array([166, 167, 168, 169, 170, 171, 172])
temperature_values_adjusted = temperature_values-169

# Color Variables
color_27mm = '#FF5A8C'
ecolor_27mm = '#990F4C'
txt_color_27mm = '#E64676'

color_38mm = '#00C8E1'
ecolor_38mm = '#005F80'
txt_color_38mm = '#0086B3'

color_31mm = '#B073FF'
ecolor_31mm = '#7800BF'
txt_color_31mm = '#E64676'

ratio_color = '#00B400'
ratio_ecolor = '#007800'

#==================================================================================================
### Functions

# Uses pandas to create a dataframe using all of the data in a given file. 
# The result is a big table of data that looks like the alpha frame list in the app.
def create_df(filename, dates):
    with pd.HDFStore(filename, 'r') as hdf:
        i = 0
        for key in hdf.keys():
            if i == 0:
                whole_df = hdf[key]
            else:
                whole_df = whole_df.append(hdf[key])
            i += 1
    
    df_cols = whole_df[['date', 'separation', 'biasV', 'midpoint', 'midpt_error', 'temperature_avg', 'temperature_rms']]
    df_dates = df_cols[df_cols['date'].isin(dates)]

    return df_dates

# Creating the final data frame by extracing the correct separation and bias voltage.
def get_final_df(df, separation, voltage):
    sep = df['separation'] == separation
    bv = df['biasV'] == voltage
    df_final = df.loc[sep & bv]
    return df_final

# Linear function (model for the best fit function).
def line_func(p, x):
    (a, b) = p
    return a*x + b

# Get the slope (a) and intercept (b) parameters that best fit the data.
# Also returns the errors on the best fit parameters, and the covariance matrix.
# The optimized parameters and the parameter errors form one large tuple.
def get_fit_parameters(x, y, x_err, y_err):
    global odr
    model_func = odr.Model(line_func)
    data = odr.RealData(x, y, sx=x_err, sy=y_err)
    odr_object = odr.ODR(data, model_func, beta0=[40, 0.05], taufac=1E-5, partol=1E-5, maxit=10000000)

    out = odr_object.run()
    optimized_parameters = out.beta
    parameter_errors = out.sd_beta
    cov_matrix = out.cov_beta

    return (optimized_parameters, parameter_errors), cov_matrix

# Calculate the reduced chi-squared value in one dimension (uses y-errors only).
def calc_reduced_chisquare_1d(opt_parameters, x_vals, y_vals, y_errors):
    y_expected = line_func(opt_parameters, np.array(x_vals))
    len_y_data = len(y_vals)
    num_parameters = 2
    chisquare = np.sum(((y_expected-y_vals)/y_errors)**2)
    reduced_chisquare_1d = chisquare/(len_y_data-num_parameters)
    return reduced_chisquare_1d

# Calculate the reduced chi-squared value in two dimensions (used both x and y errors).
def calc_reduced_chisquare_2d(opt_parameters, x_vals, y_vals, x_errors, y_errors):
    y_expected = line_func(opt_parameters, np.array(x_vals))
    len_y_data = len(y_vals)
    num_parameters = 2
    a = opt_parameters[0]
    b = opt_parameters[1]
    chisquare = np.sum((y_vals-a*x_vals-b)**2/((a**2)*(x_errors**2)+(y_errors**2)))
    reduced_chisquare_2d = chisquare/(len_y_data-num_parameters)
    return reduced_chisquare_2d

# Plot the raw data points and the error bars, find and plot the best fit line.
# Returns the optimized parameters, parameter errors, and covariance matrix for the best fit line, and both reduced chisquare values.
def plot_data(df, color, ecolor, label):
    x = df[['temperature_avg']].values[:, 0]
    x_adjusted = x-169

    y = df[['midpoint']].values[:, 0]
    x_error = df[['temperature_rms']].values[:, 0]
    y_error = df[['midpt_error']].values[:, 0]

    ax_data.errorbar(x_adjusted, y, y_error, x_error, ls='none', color=ecolor, barsabove=True, zorder=3)

    fit_parameters, cov_matrix = get_fit_parameters(x_adjusted, y, x_error, y_error)
    optimized_parameters = fit_parameters[0]
    best_fit_line = line_func(optimized_parameters, temperature_values_adjusted)

    ax_data.plot(temperature_values_adjusted, best_fit_line, c=color, label=label, linewidth=0.8)

    reduced_chisquare_1d = calc_reduced_chisquare_1d(optimized_parameters, x_adjusted, y, y_error)
    reduced_chisquare_2d = calc_reduced_chisquare_2d(optimized_parameters, x_adjusted, y, x_error, y_error)

    return fit_parameters, best_fit_line, cov_matrix, reduced_chisquare_1d, reduced_chisquare_2d

# Find the errors on the ratio by propagating errors on the best fit lines.
def get_ratio_errors(fit_parameters_1, fit_parameters_2, best_fit_line_1, best_fit_line_2, cov_matrix_1, cov_matrix_2, temperatures):

    # Variables representing the slope, intercept, and errors on the slope and intercept
    optimized_parameters_1 = fit_parameters_1[0]
    (slope_error_1, intercept_error_1) = fit_parameters_1[1]
    optimized_parameters_2 = fit_parameters_2[0]
    (slope_error_2, intercept_error_2) = fit_parameters_2[1]

    # Initializing lists for the expected voltages and ratio errors
    expected_voltages_1 = []
    expected_voltages_2 = []
    ratio_errors = []

    # Loops through each integer temperature value, calculating the error on the ratio
    for temperature in temperatures:

        # Expected values at separation 1
        expected_voltage_1 = line_func(optimized_parameters_1, temperature)
        expected_voltages_1.append(expected_voltage_1)

        # Expected values at separation 2
        expected_voltage_2 = line_func(optimized_parameters_2, temperature)
        expected_voltages_2.append(expected_voltage_2)

        # Correlation parameters (rho)
        rho_1 = cov_matrix_1[0][1] / ( np.sqrt(cov_matrix_1[0][0]) * np.sqrt(cov_matrix_1[1][1]) )
        rho_2 = cov_matrix_2[0][1] / ( np.sqrt(cov_matrix_2[0][0]) * np.sqrt(cov_matrix_2[1][1]) )

        # Errors on the best fit lins at the current temperature value
        best_fit_error_1 = np.sqrt( temperature**2 * slope_error_1**2 + intercept_error_1**2 + (2*rho_1 * slope_error_1 * temperature * intercept_error_1) )
        best_fit_error_2 = np.sqrt( temperature**2 * slope_error_2**2 + intercept_error_2**2 + (2*rho_2 * slope_error_2 * temperature * intercept_error_2) )

        # Calculate the ratio errors
        ratio_error = (expected_voltage_2 / expected_voltage_1) * np.sqrt( (best_fit_error_2/expected_voltage_2)**2 + (best_fit_error_1/expected_voltage_1)**2 )
        ratio_errors.append(ratio_error)

    # Get the y-values for the ratio errors
    ratio_points = np.divide(expected_voltages_2, expected_voltages_1)

    return ratio_points, ratio_errors

# Set the text strings for the plot
def set_text_str(slope_label, intercept_label, slope, intercept, slope_error, intercept_error, reduced_chisquare_1d, reduced_chisquare_2d):
    txt_str = '\n'.join(['Best fit function:',
                        'V = {}*(T-169) + {}'.format(slope_label, intercept_label),
                        'slope {} = {:.4f} +/- {:.4f}'.format(slope_label, slope, slope_error),
                        'intercept {} = {:.4f} +/- {:.4f}'.format(intercept_label, intercept, intercept_error),
                        r'$\chi^2{1D}$' + f' = {reduced_chisquare_1d:.6f}',
                        r'$\chi^2{2D}$' + f' = {reduced_chisquare_2d:.6f}'])
    return txt_str

#==================================================================================================
### Executing the functions 

# This if statement allows the code to run when the script is not being imported to another script.
if __name__ == '__main__':

    # Variables
    bias_voltage = '50V'
    y_range_data = (0.1, 0.6)
    y_range_ratio = (0.1, 0.6)

    # Defining the variables for the figure and both axes, setting the width of the figure.
    fig, ax_data = plt.subplots()
    fig.set_figwidth(9)
    fig.set_figheight(6)
    ax_ratio = ax_data.twinx()

    # Create the dataframes
    df_dates = create_df(alphas_filename, date_list)

    df_27mm = get_final_df(df_dates, '27', bias_voltage)
    df_38mm = get_final_df(df_dates, '38', bias_voltage)
    df_31mm = get_final_df(df_dates, '31.07', bias_voltage)

    # Deletes the first row of the dataframe only if the bias voltage is 48V and the separation is 27mm.
    # if bias_voltage ==  '48V':
        # df_27mm = df_27mm.iloc[1]

    # Plot the data and set the variables for the optimized parameters, parameter errors, covariance matrix, and reduced chi-square values.
    fit_parameters_27mm, best_fit_line_27mm, cov_matrix_27mm, reduced_chisquare_1d_27mm, reduced_chisquare_2d_27mm = plot_data(df_27mm, color_27mm, ecolor_27mm, '27mm')
    fit_parameters_38mm, best_fit_line_38mm, cov_matrix_38mm, reduced_chisquare_1d_38mm, reduced_chisquare_2d_38mm = plot_data(df_38mm, color_38mm, ecolor_38mm, '38mm')
    fit_parameters_31mm, best_fit_line_31mm, cov_matrix_31mm, reduced_chisquare_1d_31mm, reduced_chisquare_2d_31mm = plot_data(df_31mm, color_31mm, ecolor_31mm, '31mm')

    # Find and plot the ratio line
    # ratio_line = best_fit_line_38mm/best_fit_line_27mm
    # ax_ratio.plot(temperature_values_adjusted, ratio_line, c=ratio_color, label='ratio', linewidth=0.75)

    # Find and plot the ratio errors
    # ratio_points, ratio_errors = get_ratio_errors(fit_parameters_27mm, fit_parameters_38mm, best_fit_line_27mm, best_fit_line_38mm, cov_matrix_27mm, cov_matrix_38mm, temperature_values_adjusted)
    # ax_ratio.errorbar(temperature_values_adjusted, ratio_points, ratio_errors, ls='none', color=ratio_ecolor, barsabove=True, zorder=3)

    #==============================================================================================
    ## Plot settings
    
    # Setting the y range for both axes
    # ax_data.set_ylim(*y_range_data)
    # ax_ratio.set_ylim(*y_range_ratio)

    # Setting the axis labels
    ax_data.set_xlabel('Temperature [K]', fontsize=14)
    ax_data.set_ylabel('Midpoint [V]', fontsize=14)
    ax_ratio.set_ylabel('Ratio', fontsize=14)

    # Setting the super title and the title (the super title goes above the title and uses bigger text)
    plt.suptitle('Midpoint vs. Temperature at {} Bias Voltage\n'.format(bias_voltage), fontsize=14)
    
    date_27mm = ", ".join(df_27mm.date.unique())
    date_38mm = ", ".join(df_38mm.date.unique())
    date_31mm = ". ".join(df_31mm.date.unique())

    plt.title('27mm: {}   38mm: {}'.format(date_27mm, date_38mm))

    # Creating the text boxes
    (slope_27mm, intercept_27mm), (slope_error_27mm, intercept_error_27mm) = fit_parameters_27mm
    (slope_38mm, intercept_38mm), (slope_error_38mm, intercept_error_38mm) = fit_parameters_38mm
    (slope_31mm, intercept_31mm), (slope_error_31mm, intercept_error_31mm) = fit_parameters_31mm
    txtstr_27mm = set_text_str('a', 'b', slope_27mm, intercept_27mm, slope_error_27mm, intercept_error_27mm, reduced_chisquare_1d_27mm, reduced_chisquare_2d_27mm)
    txtstr_38mm = set_text_str('c', 'd', slope_38mm, intercept_38mm, slope_error_38mm, intercept_error_38mm, reduced_chisquare_1d_38mm, reduced_chisquare_2d_38mm)
    txtstr_31mm = set_text_str('e', 'f', slope_31mm, intercept_31mm, slope_error_31mm, intercept_error_31mm, reduced_chisquare_1d_31mm, reduced_chisquare_2d_31mm)

    # Setting the position of the text on the figure
    ax_data.set_position((0.1, 0.1, 0.6, 0.8))
    ax_ratio.set_position((0.1, 0.1, 0.6, 0.8))
    plt.figtext(0.78, 0.5, txtstr_27mm, color=txt_color_27mm, fontsize=10)
    plt.figtext(0.78, 0.25, txtstr_38mm, color=txt_color_38mm, fontsize=10)
    plt.figtext(0.78, 0.1, txtstr_31mm, color=txt_color_31mm, fontsize=10)

    # Label the x-ticks with the actual temperature values (166-172)
    for ax in [ax_data, ax_ratio]:
        locs = ax.get_xticks()
        adjusted_locs = [str(int(l+169)) for l in locs]
        ax.set_xticklabels(adjusted_locs)
    
    plt.grid(False)
    ax_data.legend(bbox_to_anchor=(1.4, 1.0))
    ax_ratio.legend(bbox_to_anchor=(1.38, 0.9))
    plt.show()