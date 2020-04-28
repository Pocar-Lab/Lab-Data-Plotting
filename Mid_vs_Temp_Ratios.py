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

temperature_values = np.array([166, 167, 168, 169, 170, 171, 172])

# A dictionary that sets the color of the line based on the value of the bias voltage
colors = {
    '47V': '#FF0000',
    '48V': '#FF7D00',
    '49V': '#FFE100',
    '50V': '#00C800',
    '51V': '#0050FF',
    '52V': '#A000FF',
}

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
    assert type(a) == float or type(a) == np.float or type(a) == np.float64, f"{type(a)} {a}"
    assert type(x) == np.array or type(x) == np.ndarray, f"{type(x)} {x}"
    return a*x + b

# Get the a and b parameters that best fit the data:
def get_fit_parameters(x, y, x_err, y_err,):
    global odr
    model_func = odr.Model(line_func)
    data = odr.RealData(x, y, sx=x_err, sy=y_err)
    odr_object = odr.ODR(data, model_func, beta0=[40, 0.05], taufac=1E-5, sstol=1E-5, partol=1E-5, maxit=10000000)
    out = odr_object.run()
    p_opt = out.beta
    p_err = out.sd_beta
    y_fit = line_func(p_opt, x)

    return p_opt, p_err

# Calculate the reduced chi-squared value:
def calc_reduced_chisquare(opt_params, x_vals, y_vals, y_errors):
    y_exp = line_func(opt_params, np.array(x_vals))
    len_y_data = len(y_vals)
    num_parameters = 2
    chisquare = np.sum(((y_exp-y_vals)/y_errors)**2)
    reduced_chisquare = chisquare/(len_y_data-num_parameters)
    return reduced_chisquare

# Plot the raw data points, the error bars, finding and plotting the best fit line. 
# Returns the optimized parameters for the best fit line, and the reduced chisquare value.
def analyze_data(df):
    x_range = np.array(range(166, 173))

    x = df[['temperature_avg']].values[:, 0]
    x_adjusted = x-169
    x_range_adjusted = x_range-169
    
    y = df[['midpoint']].values[:, 0]
    x_error = df[['temperature_rms']].values[:, 0]
    y_error = df[['midpt_error']].values[:, 0]
    
    optimized_parameters = get_fit_parameters(x, y, x_error, y_error)
    best_fit_line = line_func(optimized_parameters[0], x_range)
    
    return optimized_parameters

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
    df_27mm = get_final_df(df_dates, '27', voltage)
    df_38mm = get_final_df(df_dates, '38', voltage)

    optimized_parameters_27mm = plot_data(df_27mm)
    optimized_parameters_38mm = plot_data(df_38mm)

    best_fit_line_27mm = line_func(optimized_parameters_27mm[0], temperature_values)
    best_fit_line_38mm = line_func(optimized_parameters_38mm[0], temperature_values)

    ratio_line = best_fit_line_38mm/best_fit_line_27mm

    plt.plot(temperature_values, ratio_line, c=colors[voltage], label=voltage)

#==================================================================================================
### Plot Settings

plt.ylim(0.50, 0.55)

plt.xlabel('Temperature [K]', fontsize=14)
plt.ylabel('Ratio', fontsize=14)
plt.suptitle('Ratio of Best Fit Lines at 38mm and 27mm')

date_27mm = ", ".join(df_27mm.date.unique())
date_38mm = ", ".join(df_38mm.date.unique())

plt.title('27mm: {}   38mm: {}'.format(date_27mm, date_38mm))

plt.grid(True)
plt.legend()
plt.show()