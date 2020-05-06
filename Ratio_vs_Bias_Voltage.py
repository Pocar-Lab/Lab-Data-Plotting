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

from Midpoint_vs_Temp_Plotting import create_df, get_final_df, line_func, get_fit_parameters

#==================================================================================================
### Variables

alphas_filename = 'alphas.h5'
date_list = ['20190516', '20190424']



#==================================================================================================
### Functions


#==================================================================================================
### Execute Functions

if __name__ == '__main__':

    # Create the dataframes
    
