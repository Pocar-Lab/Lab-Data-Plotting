import h5py
import csv
import pandas as pd

#==================================================================================================
### Variables: Set these to indicate the date(s), separation(s), and bias voltage(s) to include in the data.

date_list = ['20190516', '20190424']
bias_voltage = '50V'
separation = '27'

# Change this variable to indicate the path you want to save the csv file to.
# The filename will change to reflect the bias voltage and/or separation indicated by the user.
file_path = 'midpoint_temperature_data.csv'
file_path_bv = 'midpoint_temperature_data_{}.csv'.format(bias_voltage)
file_path_bv_sep = 'midpoint_temperature_data_{}_{}mm.csv'.format(bias_voltage, separation)

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

# Extract the columns needed
df_cols = whole_df[['date', 'separation', 'biasV', 'midpoint', 'midpt_error', 'temperature_avg', 'temperature_rms']]

# Extract the date(s) needed
# This dataframe contains ALL of the data I use for my Midpoint vs. Temperature plots. (Every plot at every bias voltage uses all this data)
df_dates = df_cols[df_cols['date'].isin(date_list)]

#==================================================================================================
### The code below creates csv files for smaller sets of data.
### If there is a data frame that you do not wish to save, simply comment out the appropriate line that has to_csv(). 

df_dates.to_csv(file_path)

# This will produce a smaller dataframe with a single value for bias voltage. It includes both separations.
bv = df_dates['biasV'] == bias_voltage
df_one_bias = df_dates.loc[bv]

df_one_bias.to_csv(file_path_bv)

# This will produce an even smaller dataframe with a single value for bias voltage and a single value for separation.
sep = df_one_bias['separation'] == separation
df_one_separation = df_one_bias.loc[sep]

df_one_separation.to_csv(file_path_bv_sep)
