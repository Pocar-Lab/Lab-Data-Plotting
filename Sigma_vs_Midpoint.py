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

from Midpoint_vs_Temp_Plotting import line_func, get_fit_parameters, calc_reduced_chisquare_1d, calc_reduced_chisquare_2d

#==================================================================================================
### Variables

alphas_filename = 'alphas.h5'
date_list = ['20190516']

temperature_values = np.array([166, 167, 168, 169, 170, 171, 172])
temperature_values_adjusted = temperature_values-169

# Color Variables
data_color = '#0096FF'
error_color = '#0050FF'

#==================================================================================================
### Executing Functions

if __name__ == '__main__':

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_position((0.11, 0.1, 0.6, 0.8))

    bias_voltage = '48V'
    separation = '27'

    with pd.HDFStore(alphas_filename, 'r') as hdf:
        i = 0
        for key in hdf.keys():
            if i == 0:
                whole_df = hdf[key]
            else:
                whole_df = whole_df.append(hdf[key])
            i += 1
    
    df_cols = whole_df[['date', 'separation', 'biasV', 'midpoint', 'midpt_error', 'sigma', 'sigma_error']]
    df_dates = df_cols[df_cols['date'].isin(date_list)]
    sep = df_dates['separation'] == separation
    bv = df_dates['biasV'] == bias_voltage
    df_final = df_dates.loc[sep & bv]

    midpoints = df_final[['midpoint']].values[:, 0]
    sigmas = df_final[['sigma']].values[:, 0]
    midpoint_errors = df_final[['midpt_error']].values[:, 0]
    sigma_errors = df_final[['sigma_error']].values[:, 0]

    fit_parameters, cov_matrix = get_fit_parameters(midpoints, sigmas, midpoint_errors, sigma_errors)
    optimized_parameters = fit_parameters[0]
    (slope, intercept), (slope_error, intercept_error) = fit_parameters
    best_fit_line = line_func(optimized_parameters, midpoints)

    ax.errorbar(midpoints, sigmas, sigma_errors, midpoint_errors, ls='none', color=error_color, barsabove=True, zorder=3)
    ax.plot(midpoints, best_fit_line, c=data_color, linewidth=0.8)

    reduced_chisquare_1d = calc_reduced_chisquare_1d(optimized_parameters, midpoints, sigmas, sigma_errors)
    reduced_chisquare_2d = calc_reduced_chisquare_2d(optimized_parameters, midpoints, sigmas, midpoint_errors, sigma_errors)

    ax.set_xlabel('Midpoint [V]', fontsize=14)
    ax.set_ylabel('Sigma', fontsize=14)

    plt.suptitle('Sigma vs. Midpoint at {} Separation, {} Bias Voltage\n'.format(separation, bias_voltage))
    date = ", ".join(df_final.date.unique())
    plt.title(date)

    txt = '\n'.join(['slope = {:.4f} +/- {:.4f}'.format(slope, slope_error),
                    'intercept = {:.4f} +/- {:.4f}'.format(intercept, intercept_error),
                    r'$\chi^2{1D}$' + f' = {reduced_chisquare_1d:.6f}',
                    r'$\chi^2{2D}$' + f' = {reduced_chisquare_2d:.6f}'])

    plt.figtext(0.73, 0.6, txt, fontsize=10)

    plt.grid(True)
    plt.show()