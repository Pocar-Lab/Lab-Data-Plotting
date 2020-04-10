# Lab-Data-Plotting
Collection of scripts used to plot data taken from the lab and Chroma

## Midpoint_vs_Temp_Plotting:
This script creates a whole Midpoint vs. Temperature plot. It's written to plot this data at 38mm and 27mm separation. It also plots the ratio (best fit line at 38mm)/(best fit line at 27mm) with error bars on the ratio.

## Extract_Data_to_Text:
This script creates a data frame using parts of the whole data set indicated by the user. The user can decide which dates, separation and bias voltage they want to include in the data frame. The data frame(s) are then saved to a csv (comma separated values) file.

## Mid_vs_Temp_Ratios:
This script plots the ratio line at every bias voltage used in the midpoint vs. temperature plotting script to compare every ratio on the same plot. Only the ratios are included, not the data or best fit lines. 

### To-Do List:
1. The error bars on the ratio lines are huge. We need to figure out how to appropriately calculate the errors on the ratios so that they are not so large.
2. Study the covariance matrix and the correlation term and figure out how these can be used to calculate the ratio errors (or calculate and/or evaluate the best fit line). This might be found as an output in the Python ODR fitting algorithm used in "Mid_vs_Temp_Plotting".
3. Write a script that flips the x and y axes (temperature becomes the y axis and midpoint becomes the x-axis) so that we can find reduced chi-squared values using the temperature errors. 
4. Write a script to plot a set of data at one bias voltage (use 50V) that finds the best fit line parameters with errors, and then plots two possible best fit line along with two possible upper bounds on the best fit line, and two possible lower bounds on the best fit line.
