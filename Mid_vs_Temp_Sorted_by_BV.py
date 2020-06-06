############################### Midpoint vs. Temperature (Sorted by BV) #################################
# This script plots the temperature of the system on the x-axis and the midpoint of the alpha peak histogram
# on the y-axis.
# This version of the plot uses only one separation and bias voltage, however it will color different sets of
# data based on the date they were taken.
# The script will plot the raw data as error bars, and the best fit line for the data will be found and plotted.
# This script does not plot ratios.
#==========================================================================================================

import h5py
import csv
import pandas as pd
import numpy as np
from numpy import sqrt

import scipy
from scipy import odr

from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from Functions import *
from Midpoint_vs_Temperature import plot_data, set_text_str

#==========================================================================================================
### Variables ###

## Change these variables to specify the desired conditions for the plot
date_list = ['20181211']

temperature_ints = np.array([166, 167, 168, 169, 170, 171, 172])
temperature_ints_shifted = temperature_ints-169
bias_voltages = ['47V', '48V', '49V', '50V', '51V', '52V']

alphas_filename = 'alphas.h5'

