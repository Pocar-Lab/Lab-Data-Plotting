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
date_list = ['20190212', '20190207', '20190124'] 
date_list2 = ['20181212', '20181211', '20181210']

current_date = ['20181210']


# Dictionary that sets the color of the line based on the date
colors = {
    '20190212': '#FF0000',
    '20190207': '#FF7D00',
    '20190124': '#FFE100',
    '20181212': '#00C800',
    '20181211': '#0096FF',
    '20181210': '#A000FF',
}

# colors = {
#     '20190212': '#FF7D00',
#     '20190207': '#00C800', 
#     '20190124': '#0096FF', 
#     '20181212': '#A000FF',
#     '20181211': '#A000FF',
#     '20181210': '#A000FF',
# }

error_colors = {
    '20190212': '#BF0000',
    '20190207': '#CC6400',
    '20190124': '#CCB400',
    '20181212': '#008000',
    '20181211': '#0050FF',
    '20181210': '#7800BF',
}

# error_colors = {
#     '20190212': '#CC6400', 
#     '20190207': '#008000',
#     '20190124': '#0050FF', 
#     '20181212': '#7800BF',
#     '20181211': '#7800BF',
#     '20181210': '#7800BF',
# }

label = {
    '20190212': '02/12/2019',
    '20190207': '02/07/2019',
    '20190124': '01/24/2019',
    '20181212': '12/12/2018',
    '20181211': '12/11/2018',
    '20181210': '12/10/2018',
}

temperature_values = np.array([166, 167, 168, 169, 170, 171, 172])
temperature_values_adjusted = temperature_values-169

#==================================================================================================
### Functions

def get_df(df, date, voltage):
    d = df['date'] == date
    bv = df['biasV'] == voltage
    df_final = df.loc[d & bv]
    return df_final

def plot_data(df, color, ecolor, label):
    x = df[['temperature_avg']].values[:, 0]
    x_adjusted = x-169

    # Set the range of x values to be the minimum x-value-1 and the maximum x-value+1
    x_range = np.arange((min(x)-1), (max(x)+2))
    x_range_adjusted = x_range-169

    y = df[['midpoint']].values[:, 0]
    x_error = df[['temperature_rms']].values[:, 0]
    y_error = df[['midpt_error']].values[:, 0]

    ax_data.errorbar(x_adjusted, y, y_error, x_error, ls='none', color=ecolor, barsabove=True, zorder=3)

    fit_parameters, cov_matrix = get_fit_parameters(x_adjusted, y, x_error, y_error)
    optimized_parameters = fit_parameters[0]
    best_fit_line = line_func(optimized_parameters, x_range_adjusted)

    ax_data.plot(x_range_adjusted, best_fit_line, c=color, label=label, linewidth=0.8)

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

    fig, ax_data = plt.subplots()
    ax_data.set_position((0.1, 0.1, 0.6, 0.8))

    bias_voltage = '50V'
    
    # Create the dataframes
    # df_dates = create_df(alphas_filename, date_list)

    # txt_pos = 0.55
    # for dates in date_list:

    #     df = get_df(df_dates, dates, bias_voltage)

    #     fit_parameters, best_fit_line, cov_matrix, reduced_chisquare_1d, reduced_chisquare_2d = plot_data(df, colors[dates], error_colors[dates], label[dates])

    #     (slope, intercept), (slope_error, intercept_error) = fit_parameters

    #     txt = set_text_str(slope, intercept, slope_error, intercept_error, reduced_chisquare_1d, reduced_chisquare_2d)

    #     plt.figtext(0.78, txt_pos, txt, color=colors[dates], fontsize=6)

    #     txt_pos = txt_pos-0.1

    df_dates = create_df(alphas_filename, current_date)
    dataframe = get_df(df_dates, current_date[0], bias_voltage)
    
    # dataframe = dataframe.drop(index=dataframe.index[[-3]])
    
    fit_parameters, best_fit_line, cov_matrix, reduced_chisquare_1d, reduced_chisquare_2d = plot_data(dataframe, colors[current_date[0]], error_colors[current_date[0]], label[current_date[0]])
    (slope, intercept), (slope_error, intercept_error) = fit_parameters
    txt = set_text_str(slope, intercept, slope_error, intercept_error, reduced_chisquare_1d, reduced_chisquare_2d)
    plt.figtext(0.72, 0.5, txt, color=colors[current_date[0]], fontsize=10)

    #==============================================================================================
    ## Plot settings

    # Setting the axis labels
    ax_data.set_xlabel('Temperature [K]')
    ax_data.set_ylabel('Midpoint [V]')

    ax_data.set_xlim(-4, 8, 1)
    
    # Setting the super title and the title
    plt.suptitle('Midpoint vs. Temperature at {} Bias Voltage, 27mm Separation'.format(bias_voltage))

    # Label the x-ticks with the actual temperature values (166-172)
    for ax in [ax_data]:
        locs = ax.get_xticks()
        adjusted_locs = [str(int(l+169)) for l in locs]
        ax.set_xticklabels(adjusted_locs)
    
    plt.grid(True)
    ax_data.legend(bbox_to_anchor=(1.4, 1.05))
    plt.show()