"""Functions associated with the analysis of the Machine Learning data (observables)."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import dwrangling as DW
import dwrangling.dataframes as DWDF
import analysis.plotting as AP



### Data ###
sns.set(style="white")  # Sets all seaborn plot backgrounds to white.


### Functions ###
def plot_correlation_matrix_heatmap(odataframe, c_range=(None, None), fig_name=None,
                                    collision=None):
    f"""Plots the correlation matrix's heatmap, and returns the correlation matrix.

    Parameters:
     - odataframe: A 'dwrangling.ObservablesDataFrame' which contains all the events 
                   and observables you wish to plot;
     - c_range: A 'tuple' for the range of values for the colour bar, given as 
                (c_min, c_max). If either value is 'None', then they will take the 
                minimum/maximum values of the correlation matrix. [Default 
                '(None, None)'];
     - fig_name: A 'str' if given will save the plot with this name, if 'None' is 
                 given, it will not be saved. [Default 'None'];
     - collision: A 'str' denoting the collision, must be in {DW.collisions}, and 
                  must be given if 'fig_name' is not 'None'. [Default 'None']. 

    Returns:
     - correlation_matrix: A 'pd.DataFrame' containg the correlation matrix of
                           'odataframe'."""
    # Parameter validations
    if not isinstance(fig_name, (str, type(None))):
        raise ValueError("The parameter 'fig_name' must be of either type 'str' " +
                         f"or 'None', not '{type(fig_name)}'.")
    if isinstance(fig_name, str):
        if not isinstance(collision, str):
            raise ValueError("The parameter 'collision' must be given if 'fig_name' " +
                             "is given, and must be of type 'None', " +
                             f"not '{type(collision)}'.")
            
    if not isinstance(c_range, tuple):
        raise ValueError("The parameter 'c_range' must be a 'tuple' of 'floats', " +
                         f"not '{type(c_range)}'.")
    else:
        if len(c_range) != 2:
            raise ValueError("The parameter 'c_range' must have 2 'float' elements, " +
                             f"not '{len(c_range)}' elements.")
        if ((not isinstance(c_range[0], (int, float, type(None))))
        or (not isinstance(c_range[1], (int, float, type(None))))):
            raise ValueError("The elements of 'c_range' must  be of either type 'float' " +
                             f"or 'None', not ('{type(c_range[0])}', '{type(c_range[1])}').")
                                    

    # Get the DataFrame's correlation matrix
    corr_matrix = odataframe.corr()
    
    # Make column labels plot ready, and get correlation matrix for plotting
    cpy = odataframe.copy()
    cpy.columns = AP.latex_label(cpy.columns)
    plot_corr = cpy.corr()
    
    # Create a copy of the correlation matrix to find the minimum and maximum
    cpy = plot_corr.copy()
    
    # Remove the diagonal perfect correlation line and get minimum and maximum
    cpy = np.where((cpy.values <= -1.0) | (cpy.values >= 1.0), 0, cpy.values)
    vmin, vmax = c_range
    if c_range[0] is None:
        vmin = cpy.min()
    if c_range[1] is None:
        vmax = cpy.max()
    print(f"cmin: {vmin}, cmax: {vmax}")

    # Generate a mask for the upper triangle
    mask = np.zeros_like(plot_corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Remove top row and last column from plotting, as these are empty when masked
    plot_corr = plot_corr[ plot_corr.columns[:-1] ][1:]
    mask = mask[1:, :-1]
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(plot_corr, mask=mask, cmap=AP.std_cmap, center=0, square=True,
                vmin=vmin, vmax=vmax, linewidths=.5, cbar_kws={"shrink": .75})

    # Save fig:
    if fig_name is not None:
        DW.collision_check(collision)
        path = AP.get_image_file_paths(fig_name, '.png',
                                       'observables_analysis', collision)
        f.savefig(path, bbox_inches='tight', dpi=f.dpi)
        
    
    # Return correlation matrix with original labels
    return corr_matrix


def plot_correlation_row(odataframe, observable='signal', sort=True,
                         fig_name=None, collision=None):
    f"""Plots the correlation row's heatmap for signal, and returns the correlation row.
    
    Parameters:
     - odataframe: A 'dwrangling.ObservablesDataFrame' which contains all the events
                   and observables you wish to plot;
     - observable: A 'str' with the name of a column from 'odataframe', which you wish
                   to plot the correlation row for [Default is 'signal'];
     - sort: A 'bool' to sort the correlation row [Default is 'True'];
     - fig_name: A 'str' if given will save the plot with this name, if 'None' is 
                 given, it will not be saved. [Default 'None'];
     - collision: A 'str' denoting the collision, must be in {DW.collisions}, and 
                  must be given if 'fig_name' is not 'None'. [Default 'None']. 

    Returns:
     - correlation_matrix: A 'pd.DataFrame' containg the correlation row of
                           'odataframe'."""
    
    # Parameter validations
    if observable not in odataframe.columns:
        raise ValueError("The parameter 'observable' must be a column name in " +
                         f"'odataframe', such as: {list(odataframe.columns)}.")
    
    if not isinstance(fig_name, (str, type(None))):
        raise ValueError("The parameter 'fig_name' must be of either type 'str' " +
                         f"or 'None', not '{type(fig_name)}'.")
    if isinstance(fig_name, str):
        if not isinstance(collision, str):
            raise ValueError("The parameter 'collision' must be given if 'fig_name' " +
                             "is given, and must be of type 'None', " +
                             f"not '{type(collision)}'.")
    
    
    # Get correlation row for signal.
    corr_row = pd.DataFrame(odataframe.corr().loc[observable]).T
    
    # Sort array
    if sort:
        sorted_columns = corr_row.columns[ np.argsort(corr_row.values[0]) ]
        corr_row = corr_row[sorted_columns[::-1]]
    
    # Remove the observable-observable element
    columns = list(corr_row.columns)
    columns.remove(observable)
    corr_row = corr_row[columns]
    
    # Get minimum and maximum for colourbar
    vmin, vmax = corr_row.values.min(), corr_row.values.max()
    
    # Make column labels plot ready
    plot_row = corr_row.copy()
    plot_row.columns = AP.latex_label(plot_row.columns)
    plot_row = plot_row[ plot_row.columns[::-1] ]
    latex_observable = AP.latex_label([observable,])[0]
    plot_row = plot_row.rename(index={observable: latex_observable})
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Draw the heatmap with the correct aspect ratio
    sns.heatmap(plot_row, cmap=AP.std_cmap, center=0, square=True,
                vmin=vmin, vmax=vmax, linewidths=.5,
                cbar_kws={"shrink": 1, "orientation": "horizontal",
                          'pad':-0.081, 'aspect':35})
    
    ax.set_yticklabels(ax.get_yticklabels() ,rotation=0)
    
    # Move colorbar labels to above the colorbar
    c_bar =f.get_axes()[-1]
    c_bar.xaxis.set_ticks_position('top')
    
    # Save fig:
    if fig_name is not None:
        DW.collision_check(collision)
        path = AP.get_image_file_paths(fig_name, '.png',
                                       'observables_analysis', collision)
        f.savefig(path, bbox_inches='tight', dpi=f.dpi)
    
    # Return correlation matrix with original labels
    return corr_row
    
    