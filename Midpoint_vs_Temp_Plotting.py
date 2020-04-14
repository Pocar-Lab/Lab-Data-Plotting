# Creates a plot of data taken from the lab at different separations. 
# Each set of data on the plot uses a different separation, and was taken at a different date. Both sets share the same bias voltage.
# The information being plotted was taken from the file "alphas.h5" which contains a summary of the information extracted from the peak height histograms.
#==================================================================================================

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
bias_voltage = '50V'

# Color Variables
color_27mm = '#FF5A8C'
ecolor_27mm = '#990F4C'
txt_color_27mm = '#E64576'

color_38mm = '#00C8E1'
ecolor_38mm = '#005F80'
txt_color_38mm = '#0086B3'

ratio_color = '#00B400'
ratio_ecolor = '#007800'

# Defining the variables for the figure and both axes, and setting the width of the figure
fig, ax_data = plt.subplots()
fig.set_figwidth(9)
fig.set_figheight(6)
ax_ratio = ax_data.twinx()

#==================================================================================================
### Functions: These are all of the functions that will be needed in order to create a plot

# Creating the final data frame by extracting the correct separation and bias voltage. 
# This data frame will be the input into the plot data function.
def get_final_df(df, separation, voltage):
    sep = df['separation'] == separation
    bv = df['biasV'] == voltage
    df_final = df_dates.loc[sep & bv]
    return df_final

# Linear model function:
def line_func(p, x):
    (a, b) = p
    return a*x + b

# Get the slope (a) and intercept (b) parameters that best fit the data:
def get_fit_parameters(x, y, x_err, y_err):
    global odr
    model_func = odr.Model(line_func)
    data = odr.RealData(x, y, sx=x_err, sy=y_err)
    odr_object = odr.ODR(data, model_func, beta0=[40, 0.05], taufac=1E-5, sstol=1E-5, partol=1E-5, maxit=10000000)
    out = odr_object.run()
    p_opt = out.beta
    p_err = out.sd_beta
    # p_matrix = out.cov_beta
    y_fit = line_func(p_opt, x)
    return p_opt, p_err # Returns a two-dimensional array where the first row is p_opt, and the second row is p_err

# Calculate the reduced chi-squared value:
def calc_reduced_chisquare_1d(opt_parameters, x_vals, y_vals, y_errors):
    y_expected = line_func(opt_parameters, np.array(x_vals))
    len_y_data = len(y_vals)
    num_parameters = 2
    chisquare = np.sum(((y_expected-y_vals)/y_errors)**2)
    reduced_chisquare_1d = chisquare/(len_y_data-num_parameters)
    return reduced_chisquare_1d

def calc_reduced_chisquare_2d(opt_parameters, x_vals, y_vals, x_errors, y_errors):
    y_expected = line_func(opt_parameters, np.array(x_vals))
    len_y_data = len(y_vals)
    num_parameters = 2
    a = opt_parameters[0]
    b = opt_parameters[1]
    chisquare = np.sum((y_vals-a*x_vals-b)**2/((a**2)*(x_errors**2)+(y_errors**2)))
    reduced_chisquare_2d = chisquare/(len_y_data-num_parameters)
    return reduced_chisquare_2d

# Plot the raw data points and the error bars, and finding and plotting the best fit line. 
# Returns the optimized parameters for the best fit line, and the reduced chisquared value.
def plot_data(df, color, ecolor, label):
    x_range = np.array(range(166, 173))

    x = df[['temperature_avg']].values[:, 0]
    x_adjusted = x-169
    x_range_adjusted = x_range-169
    
    y = df[['midpoint']].values[:, 0]
    x_error = df[['temperature_rms']].values[:, 0]
    y_error = df[['midpt_error']].values[:, 0]

    # ax_data.scatter(x, y, c=color)
    ax_data.errorbar(x, y, y_error, x_error, ls='none', color=ecolor, barsabove=True, zorder=3)

    optimized_parameters = get_fit_parameters(x, y, x_error, y_error)
    best_fit_line = line_func(optimized_parameters[0], x_range)
    ax_data.plot(x_range, best_fit_line, c=color, label=label, linewidth=0.8)

    reduced_chisquare_1d = calc_reduced_chisquare_1d(optimized_parameters[0], x, y, y_error)
    reduced_chisquare_2d = calc_reduced_chisquare_2d(optimized_parameters[0], x, y, x_error, y_error)
    
    return optimized_parameters, reduced_chisquare_1d, reduced_chisquare_2d

#==================================================================================================

#==================================================================================================
### Extract all of the information needed for the whole plot

# Use pandas to create a dataframe using all of the data in alphas.h5. 
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

#==================================================================================================
### Getting the two separate dataframes and plotting the Data

# Get the final data frames
df_27mm = get_final_df(df_dates, '27', bias_voltage)
df_38mm = get_final_df(df_dates, '38', bias_voltage)

# Plot the data - the plot data function is called and set as variables for optimized parameters and reduced chisquared values for the data.
optimized_parameters_27mm, reduced_chisquare_1d_27mm, reduced_chisquare_2d_27mm = plot_data(df_27mm, color_27mm, ecolor_27mm, '27mm')
optimized_parameters_38mm, reduced_chisquare_1d_38mm, reduced_chisquare_2d_38mm = plot_data(df_38mm, color_38mm, ecolor_38mm, '38mm')

#==================================================================================================

#==================================================================================================
### Finding Ratios

# List of integer temperature values
temperature_values = np.array([166, 167, 168, 169, 170, 171, 172])

# Variables representing the slope, intercept, and errors on the slope and intercept for both separations
(slope_27mm, intercept_27mm), (slope_error_27mm, intercept_error_27mm) = optimized_parameters_27mm
(slope_38mm, intercept_38mm), (slope_error_38mm, intercept_error_38mm) = optimized_parameters_38mm
(slope_27mm, intercept_27mm), (slope_error_27mm, intercept_error_27mm) = optimized_parameters_27mm
(slope_38mm, intercept_38mm), (slope_error_38mm, intercept_error_38mm) = optimized_parameters_38mm

# Find the ratio line
best_fit_line_27mm = line_func(optimized_parameters_27mm[0], temperature_values)
best_fit_line_38mm = line_func(optimized_parameters_38mm[0], temperature_values)
ratio_line = best_fit_line_38mm/best_fit_line_27mm
ax_ratio.plot(temperature_values, ratio_line, c=ratio_color, label='ratio', linewidth=0.75)

# Find and append expected values to different lists based on separation. The loop calculates the expected midpoint value based on the integer temperature value.
expected_voltages_27mm = []
expected_voltages_38mm = []
ratio_errors = []
for temperature in temperature_values:
    # Expected values at 27mm
    expected_voltage_27mm = slope_27mm*temperature + intercept_27mm
    expected_voltages_27mm.append(expected_voltage_27mm)

    # Expected values at 38mm
    expected_voltage_38mm = slope_38mm*temperature + intercept_38mm
    expected_voltages_38mm.append(expected_voltage_38mm)

    sqrt = np.sqrt
    (a, b), (da, db) = optimized_parameters_38mm
    (c, d), (dc, dd) = optimized_parameters_27mm
    
    T = temperature

    x = a*T + b
    y = c*T + d
    dx = sqrt(T**2 * da**2 + db**2)
    dy = sqrt(T**2 * dc**2 + dd**2)
    
    ratio_error = abs(x/y) * sqrt( (dx / x)**2 + (dy / y)**2 )

    ratio_errors.append(ratio_error)

print(ratio_errors)

# Find and plot the ratio points (optional - mainly used for testing purposes)
ratio_points = np.divide(expected_voltages_38mm, expected_voltages_27mm)

# ax_ratio.scatter(temperature_values, ratio_points, c=ratio_color)
ax_ratio.errorbar(temperature_values, ratio_points, ratio_errors, ls='none', color=ratio_ecolor, barsabove=True, zorder=3)

#==================================================================================================

#==================================================================================================
### Plot Settings
# Setting the y range for both axes
# ax_data.set_ylim(0.0, 0.9)
ax_ratio.set_ylim(0.0, 0.9)

# Setting the axis labels
ax_data.set_xlabel('Temperature [K]', fontsize=14)
ax_data.set_ylabel('Midpoint [V]', fontsize=14)
ax_ratio.set_ylabel('Ratio', fontsize=14)

# Setting the super title and the title (the super title goes above the title and uses bigger text)
plt.suptitle('Midpoint vs. Temperature at {} Bias Voltage\n'.format(bias_voltage), fontsize=14)

date_27mm = ", ".join(df_27mm.date.unique())
date_38mm = ", ".join(df_38mm.date.unique())

plt.title('27mm: {}   38mm: {}'.format(date_27mm, date_38mm))

# Creating Text boxes - list the slope, intercept, and errors for each separation
txtstr_27mm = '\n'.join(['slope = {:.4f} +/- {:.4f}'.format(optimized_parameters_27mm[0][0], optimized_parameters_27mm[1][0]),
                        'intercept = {:.4f} +/- {:.4f}'.format(optimized_parameters_27mm[0][1], optimized_parameters_27mm[1][1]),
                        r'$\chi^2_{1D}$' + f' = {reduced_chisquare_1d_27mm:.6f}',
                        r'$\chi^2_{2D}$' + f' = {reduced_chisquare_2d_27mm:.6f}'])

txtstr_38mm = '\n'.join(['slope = {:.4f} +/- {:.4f}'.format(optimized_parameters_38mm[0][0], optimized_parameters_38mm[1][0]),
                        'intercept = {:.4f} +/- {:.4f}'.format(optimized_parameters_38mm[0][1], optimized_parameters_38mm[1][1]),
                        r'$\chi^2_{1D}$' + f' = {reduced_chisquare_1d_38mm:.6f}',
                        r'$\chi^2_{2D}$' + f' = {reduced_chisquare_2d_38mm:.6f}'])

# Setting the position of the text on the figure
ax_data.set_position((0.1, 0.1, 0.6, 0.8))
ax_ratio.set_position((0.1, 0.1, 0.6, 0.8))
plt.figtext(0.8, 0.6, txtstr_27mm, color=txt_color_27mm, fontsize=10)
plt.figtext(0.8, 0.4, txtstr_38mm, color=txt_color_38mm, fontsize=10)

plt.grid(False)
ax_data.legend(bbox_to_anchor=(1.4, 1.0))
ax_ratio.legend(bbox_to_anchor=(1.38, 0.9))
plt.show()