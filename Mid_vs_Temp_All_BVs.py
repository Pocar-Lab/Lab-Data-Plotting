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
from Mid_vs_Temp_Pre_Baking import final_df, plot_data

#==================================================================================================
### Variables

alphas_filename = 'alphas.h5'
dates_dec = ['20181211', '20181212']
dates_feb = ['20190207']

temperature_values = np.array([166, 167, 168, 169, 170, 171, 172])
temperature_values_adjusted = temperature_values-169

# Dictionary that sets the color based on the bias voltage
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

#==================================================================================================
### Functions

def set_text_str(bias_voltage, slope_val, slope_err):
    txt_str = '\n'.join([
                        '{}:'.format(bias_voltage),
                        'slope = {:.4f} +/- {:.4f}'.format(slope_val, slope_err)])
    return txt_str

def analyze_data(df, bvs, txt_y):

    slopes = []
    slope_errors = []
    txt_pos = 0.75
    for voltage in bvs:

        dataframe = get_final_df(df, '27', voltage)

        # fit_parameters, best_fit_line, cov_matrix, reduced_chisquare_1d, reduced_chisquare_2d = plot_data(dataframe, colors[voltage], error_colors[voltage], ax, voltage)
        
        x = dataframe[['temperature_avg']].values[:, 0]
        x_adjusted = x-169
        y_vals = dataframe[['midpoint']].values[:, 0]
        x_error = dataframe[['temperature_rms']].values[:, 0]
        y_error = dataframe[['midpt_error']].values[:, 0]
        
        fit_parameters, cov_matrix = get_fit_parameters(x_adjusted, y_vals, x_error, y_error)
        (slope, intercept), (slope_error, intercept_error) = fit_parameters
        slopes.append(slope)
        slope_errors.append(slope_error)

        txt = set_text_str(voltage, slope, slope_error)
        plt.figtext(txt_y, txt_pos, txt, fontsize=10)
        txt_pos = txt_pos - 0.1
    
    return slopes, slope_errors

#==================================================================================================
### Executing Functions

if __name__ == '__main__':

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_position((0.06, 0.1, 0.6, 0.8))

    bias_voltages = np.array(['47V', '48V', '49V', '50V', '51V', '52V'])

    df_dec = create_df(alphas_filename, dates_dec)
    df_feb = create_df(alphas_filename, dates_feb)

    slopes_dec, slope_errors_dec = analyze_data(df_dec, bias_voltages, 0.68)
    slopes_feb, slope_errors_feb = analyze_data(df_feb, bias_voltages, 0.85)

    ax.plot(bias_voltages, slopes_dec, c='#A000FF')
    ax.errorbar(bias_voltages, slopes_dec, slope_errors_dec, ls='none', color='#7800BF', barsabove=True, zorder=3)
    ax.plot(bias_voltages, slopes_feb, c='#0096FF')
    ax.errorbar(bias_voltages, slopes_feb, slope_errors_feb, ls='none', color='#0050FF', barsabove=True, zorder=3)

    # dec = '\n'.join(['20181211', '20181212'])
    dec = '20181211 - 20181212:'
    feb = '20190207:'

    plt.figtext(0.68, 0.85, dec, color='#A000FF', fontsize=10)
    plt.figtext(0.85, 0.85, feb, color='#0096FF', fontsize=10)

    #==============================================================================================
    ## Plot settings

    # Setting the axis labels 
    ax.set_xlabel('Bias Voltage [V]')
    ax.set_ylabel('Slope')

    # ax.set_xlim(-4, 4, 1)
    # ax.set_ylim(0.6, 1.2)

    # Setting the super title and the title
    # date = ", ".join(df_dates.date.unique())
    plt.suptitle('Linear Slope vs. Bias Voltage')

    # date = ", ".join(df_dates.date.unique())
    # plt.title('{}'.format(date))

    # Label the x-ticks with the actual temperature values
    # for ax in [ax]:
    #     locs = ax.get_xticks()
    #     adjusted_locs = [str(int(l+169)) for l in locs]
    #     ax.set_xticklabels(adjusted_locs)
    
    plt.grid(True)
    # ax.legend(bbox_to_anchor=(1.4, 1.05))
    plt.show()