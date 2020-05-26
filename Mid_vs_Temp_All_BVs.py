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
date_list = ['20190207']

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
### Executing Functions

if __name__ == '__main__':

    fig, ax = plt.subplots()
    ax.set_position((0.1, 0.1, 0.6, 0.8))

    bias_voltages = np.array(['47V', '48V', '49V', '50V', '51V', '52V'])

    df_dates = create_df(alphas_filename, date_list)

    for voltage in bias_voltages:

        dataframe = get_final_df(df_dates, '27', voltage)

        fit_parameters, best_fit_line, cov_matrix, reduced_chisquare_1d, reduced_chisquare_2d = plot_data(dataframe, colors[voltage], error_colors[voltage], ax, voltage)
        (slope, intercept), (slope_error, intercept_error) = fit_parameters

    #==============================================================================================
    ## Plot settings

    # Setting the axis labels 
    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel('Midpoint [V]')

    ax.set_xlim(-4, 4, 1)
    # ax.set_ylim(0.6, 1.2)

    # Setting the super title and the title
    plt.suptitle('Midpoint vs. Temperature at All Bias Voltages')

    date = ", ".join(df_dates.date.unique())
    plt.title(date)

    # Label the x-ticks with the actual temperature values
    for ax in [ax]:
        locs = ax.get_xticks()
        adjusted_locs = [str(int(l+169)) for l in locs]
        ax.set_xticklabels(adjusted_locs)
    
    plt.grid(True)
    ax.legend(bbox_to_anchor=(1.4, 1.05))
    plt.show()