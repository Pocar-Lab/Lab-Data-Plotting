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

from Midpoint_vs_Temp_Plotting import create_df, get_final_df, line_func, get_fit_parameters, calc_reduced_chisquare_1d, calc_reduced_chisquare_2d

#==================================================================================================
### Variables

alphas_filename = 'alphas.h5'
date_list = ['20190207', '20190124', '20181212', '20181211']
# date_list = ['20181212', '20181211', '20181210']
# date_list = ['20181212', '20181211']
# date_list = ['20181210']

temperature_values = np.array([166, 167, 168, 169, 170, 171, 172])
temperature_values_adjusted = temperature_values-169

# Dictionary that sets the color of the line based on the date
colors = {
    # '20190212': '#FFE100',
    '20190207': '#FF0000',
    '20190124': '#FF7D00',
    '20181212': '#00C800',
    '20181211': '#0096FF',
    '20181210': '#A000FF',
}

error_colors = {
    # '20190212': '#CCB400',
    '20190207': '#BF0000',
    '20190124': '#CC6400',
    '20181212': '#008000',
    '20181211': '#0050FF',
    '20181210': '#7800BF',
}

label = {
    # '20190212': '02/12/2019',
    '20190207': '02/07/2019',
    '20190124': '01/24/2019',
    '20181212': '12/12/2018',
    '20181211': '12/11/2018',
    '20181210': '12/10/2018',
}

#==================================================================================================
### Functions

def final_df(df, date, voltage):
    d = df['date'] == date
    bv = df['biasV'] == voltage
    df_final = df.loc[d & bv]
    return df_final

def plot_data(df, color, ecolor, axis, label):
    x = df[['temperature_avg']].values[:, 0]
    x_adjusted = x-169

    x_range = np.arange((min(x)-1), (max(x)+2))
    x_range_adjusted = x_range-169

    y = df[['midpoint']].values[:, 0]
    x_error = df[['temperature_rms']].values[:, 0]
    y_error = df[['midpt_error']].values[:, 0]

    axis.errorbar(x_adjusted, y, y_error, x_error, ls='none', color=ecolor, barsabove=True, zorder=3)

    fit_parameters, cov_matrix = get_fit_parameters(x_adjusted, y, x_error, y_error)
    optimized_parameters = fit_parameters[0]
    best_fit_line = line_func(optimized_parameters, x_range_adjusted)

    axis.plot(x_range_adjusted, best_fit_line, c=color, label=label, linewidth=0.8)
    reduced_chisquare_1d = calc_reduced_chisquare_1d(optimized_parameters, x_adjusted, y, y_error)
    reduced_chisquare_2d = calc_reduced_chisquare_2d(optimized_parameters, x_adjusted, y, x_error, y_error)

    return fit_parameters, best_fit_line, cov_matrix, reduced_chisquare_1d, reduced_chisquare_2d

def set_text_str(slope, intercept, slope_error, intercept_error, reduced_chisquare_1d, reduced_chisquare_2d):
    txt_str = '\n'.join(['V = a*(T-169) + b',
                        'slope a = {:.3f} +/- {:.3f}'.format(slope, slope_error),
                        'intercept b = {:.3f} +/- {:.3f}'.format(intercept, intercept_error),
                        r'$\chi^2{1D}$' + f' = {reduced_chisquare_1d:.6f}',
                        r'$\chi^2{2D}$' + f' = {reduced_chisquare_2d:.6f}'])
    return txt_str

#==================================================================================================
### Executing Functions

if __name__ == '__main__':

    fig, ax = plt.subplots()
    ax.set_position((0.1, 0.1, 0.6, 0.8))

    bias_voltage = '50V'

    # Create the large dataframe containing all of the data to be plotted
    df_dates = create_df(alphas_filename, date_list)

    for date in date_list:

        dataf = final_df(df_dates, date, bias_voltage)

        fit_parameters, best_fit_line, cov_matrix, reduced_chisquare_1d, reduced_chisquare_2d = plot_data(dataf, colors[date], error_colors[date], label[date], ax)
        (slope, intercept), (slope_error, intercept_error) = fit_parameters

        # txt = set_text_str(slope, intercept, slope_error, intercept_error, reduced_chisquare_1d, reduced_chisquare_2d)
        # plt.figtext(0.78, 0.55, txt, color=colors[date], fontsize=6)

    #==============================================================================================
    ## Plot settings

    # Setting the axis labels
    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel('Midpoint [V]')

    ax.set_xlim(-4, 4, 1)

    # Setting the super title and the title
    plt.suptitle('Midpoint vs. Temperature at {} Bias Voltage, 27mm Separation'.format(bias_voltage))
    
    # Label the x-ticks with the actual temperature values
    for ax in [ax]:
        locs = ax.get_xticks()
        adjusted_locs = [str(int(l+169)) for l in locs]
        ax.set_xticklabels(adjusted_locs)
    
    plt.grid(True)
    ax.legend(bbox_to_anchor=(1.4, 1.05))
    plt.show()