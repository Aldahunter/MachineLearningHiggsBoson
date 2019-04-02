"""Functions associated with the analysis of the Machine Learning data \
(observables)."""

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
def plot_correlation_matrix_heatmap(odataframe, collision=None,
                                    fig_name=None, abs_corr=False,
                                    rtn_corr=False, c_range=(None, None),
                                    alt_xticks=False, alt_yticks=False,
                                    fontsize=18, cbar_pad=0.00,
                                    cbar_lpad=14):
    f"""Plots the correlation matrix's heatmap, and returns the \
    correlation matrix.

    Parameters:
     - odataframe: A :class:`dwrangling.ObservablesDataFrame` which contains 
                   all the events and observables you wish to plot;
     - collision: A :class:`str` denoting the collision, must be in
                  {DW.collisions}, and must be given if 'fig_name' is not
                  'None'. [Default 'None'];
     - fig_name: A :class:`str` if given will save the plot with this name,
                 if 'None' is given, it will not be saved. [Default 'None'];
     - abs_corr: A :class:`bool` will plot the absolute correlation values.
                 [Default 'False'];
     - rtn_corr: A :class:`bool` to return the correlation values as well as 
                 the figure.[Default 'False'];
     - c_range: A :class:`tuple` of :class:`float`s for the range of values
                for the colour bar, given as (c_min, c_max). If either value
                is `None`, then they will take the minimum/maximum values of
                the correlation matrix. [Default '(None, None)'];
     - alt_xticks: A :class:`bool` which if 'True' will alternate the x-axis
                   tickmarks, if used with 'alt_yticks' one of every
                   ticklabel is still visible. [Default 'False'];
     - alt_yticks: A :class:`bool` which if 'True' will alternate the y-axis
                   tickmarks, if used with 'alt_xticks' one of every
                   ticklabel is still visible. [Default 'False'];
     - fontsize: A :class:`float` determining the font size of the text in
                 the figure. [Default '18'];
     - cbar_pad: A :class:`float` determining the spacing between the axis
                 and the colorbar. [Default '0.00'];
     - cbar_lpad: A :class:`float` determining the spacing between the
                  colorbar and the 'Correlation' label. [Default '14'].
                 
    Returns:
     - figure: A :class:`matplotlib.figure` instance of the plot;
     - correlation_matrix: A :class:'pandas.DataFrame' containg the
                           correlation matrix of parameter 'odataframe'.
                           [Only returned if 'rtn_corr' is `True`]."""
    
    # Parameter validations
    if not isinstance(fig_name, (str, type(None))):
        raise ValueError("The parameter 'fig_name' must be of either " +
                         f"type 'str' or 'None', not '{type(fig_name)}'.")
    if isinstance(fig_name, str):
        if not isinstance(collision, str):
            raise ValueError("The parameter 'collision' must be given if " +
                             "'fig_name' is given, and must be of type " +
                             f"'None', not '{type(collision)}'.")
            
    if not isinstance(c_range, tuple):
        raise ValueError("The parameter 'c_range' must be a 'tuple' of " +
                         f"'floats', not '{type(c_range)}'.")
    else:
        if len(c_range) != 2:
            raise ValueError("The parameter 'c_range' must have 2 " +
                             f"'float' elements, not '{len(c_range)}' " +
                             "elements.")
        if ((not isinstance(c_range[0], (int, float, type(None))))
        or (not isinstance(c_range[1], (int, float, type(None))))):
            raise ValueError("The elements of 'c_range' must  be of " +
                             "either type 'float' or 'None', not " +
                             f"('{type(c_range[0])}', " +
                             f"'{type(c_range[1])}').")  

    # Get the DataFrame's correlation matrix
    corr_matrix = odataframe.corr()
    
    # If abs corr
    if abs_corr:
        corr_matrix = np.abs(corr_matrix)
    
    # Make column labels plot ready, and get correlation matrix for plotting
    copy = odataframe.copy()
    copy.columns = AP.latex_label(copy.columns)
    plot_corr = copy.corr()
    
    # Obtain minimum and maximum correlation values
    vmin, vmax = c_range
    if (vmin is None) or (vmax is None):
        # Create a copy of the correlation matrix to find min and max
        copy = plot_corr.copy()
        
        # Remove perfect diagonal correlation line and get min and max
        copy = np.where((copy.values <= -1.0) | (copy.values >= 1.0),
                        0, copy.values)
        if vmin is None:
            vmin = copy.min()
        if vmax is None:
            vmax = copy.max()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(plot_corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Remove top row and last column from plotting, since empty when masked
    plot_corr = plot_corr[ plot_corr.columns[:-1] ][1:]
    mask = mask[1:, :-1]
    
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(plot_corr, mask=mask, cmap=AP.std_cmap, center=0,
                square=True, vmin=vmin, vmax=vmax, linewidths=0.0,
                cbar_kws={'shrink': 1.0, 'pad':cbar_pad})
    fig.canvas.draw()

    # Make figure transparent
    fig.patch.set_facecolor('None')
    ax.patch.set_facecolor('None')
    
    # Alternate axis tick marks
    if alt_xticks:
        for n, tick in enumerate(ax.get_xaxis().get_major_ticks()):
            if n % 2 == 0: tick.set_visible(True)
            else: tick.set_visible(False)
    if alt_yticks:
        r = not (len(ax.get_yaxis().get_major_ticks()) % 2)
        for n, tick in enumerate(ax.get_yaxis().get_major_ticks()[::-1]):
            if n == 0: tick.set_visible(True)
            elif n % 2 == r: tick.set_visible(True)
            else: tick.set_visible(False)
    
    # Set ticklabels and fontsize
    ax.tick_params(axis='both', which='major', direction='out',
                   labelsize=fontsize, left=True, bottom=True)
    ax.tick_params(axis='y', which='major', labelrotation=0, pad=0)
    ax.tick_params(axis='x', which='major', labelrotation=90, pad=0)
    
    
    # Get colorbar axes
    cax = fig.get_axes()[-1]
    
    # Set colorbar ticklabels position, fontsize, fontstyle and rotation
    cax.tick_params(axis='y', which='major', labelsize=fontsize*0.94)
    cax.set_ylabel('Correlation',fontproperties=AP.std_font, ha='center',
                   fontsize=fontsize, rotation=270, labelpad=cbar_lpad)
    
    # Add +/- space formatting to axes
    AP.format_ticklabel_signspacing(cax, which='y')
    
    # Save fig:
    if fig_name is not None:
        DW._collision_check(collision)
        path = AP.get_image_file_paths(fig_name, '.png',
                                       'observables_analysis', collision)
        fig.savefig(path, transparent=True, bbox_inches='tight',
                    dpi=fig.dpi)
    
    # Return figure and correlation matrix with original labels
    if rtn_corr:
        return fig, corr_matrix
    return fig


def plot_correlation_row(odataframe, collision=None, fig_name=None,
                         abs_corr=False, sort=True, direction='vertical',
                         rtn_corr=False, c_range=(None, None), fontsize=18,
                         cbar_pad=-0.158, cbar_lpad=9.5):
    f"""Plots the correlation row's heatmap for signal, and returns the \
    correlation row.
    
    Parameters:
     - odataframe: A :class:`dwrangling.ObservablesDataFrame` which contains 
                   all the events and observables you wish to plot;
     - collision: A :class:`str` denoting the collision, must be in
                  {DW.collisions}, and must be given if 'fig_name' is not
                  'None'. [Default 'None'];
     - fig_name: A 'str' if given will save the plot with this name, if
                 'None' is given, it will not be saved. [Default 'None'];
     - abs_corr: A :class:`bool` will plot the absolute correlation values.
                 [Default 'False'];
     - sort: A :class:`bool` to sort the correlation row [Default is
             'True'];
     - direction: A :class:`str` which determines the direction of the of
                  the plot, must either: '`vertical`' or '`horizontal`'.
                  [Default 'vertical'];
     - rtn_corr: A :class:`bool` to return the correlation values as well as 
                 the figure.[Default 'False'];
     - c_range: A :class:`tuple` of :class:`float`s for the range of values
                for the colour bar, given as (c_min, c_max). If either value
                is `None`, then they will take the minimum/maximum values of
                the correlation matrix. [Default '(None, None)'];
     - fontsize: A :class:'float' determing the font size of the text in the
                 figure. [Default '18'];
     - cbar_pad: A :class:'float' determing the spacing between the axis and 
                 the colorbar. [Default '-0.15'];
     - cbar_lpad: A :class:`float` determining the spacing between the
                  colorbar and the 'Correlation' label. [Default '9.5'].

    Returns:
     - figure: A :class:`matplotlib.figure` instance of the plot;
     - correlation_row: A :class:'pandas.DataFrame' containg the correlation
                        row of parameter 'odataframe'. [Only returned if
                        'rtn_corr' is 'True']."""
    
    # Parameter validations
    directions = ['v', 'vertical', 'h', 'horizontal']
    if direction.lower() not in directions:
        raise ValueError("The parameter 'direction' cannot be " +
                         f"'{direction}'. It must be one of: {directions}.")
    
    if 'signal' not in odataframe.columns:
        raise ValueError("'signal' must be a column name in 'odataframe', "
                         + f"such as: {list(odataframe.columns)}.")
    
    if not isinstance(fig_name, (str, type(None))):
        raise ValueError("The parameter 'fig_name' must be of either " + 
                         f"type 'str' or 'None', not '{type(fig_name)}'.")
    if isinstance(fig_name, str):
        if not isinstance(collision, str):
            raise ValueError("The parameter 'collision' must be given " +
                             "if 'fig_name' is given, and must be of " +
                             f"type 'None', not '{type(collision)}'.")
    
    # Setup alignment variables
    if direction in ['v', 'vertical']:
        labelrot, obsv_spad, corr_spad = 90, ' '*3, ' '*4
        invert_cbar = True
        def cax_set_label(*args, **kwargs):
            cax.set_ylabel(*args, **kwargs)
            cax.xaxis.set_label_position('bottom')
    else:
        labelrot, obsv_spad, corr_spad = 0, '', ''
        invert_cbar = False
        def cax_set_label(*args, **kwargs):
            cax.set_xlabel(*args, **kwargs)
            cax.xaxis.set_label_position('top') 
    
    
    # Get correlation row for signal.
    corr_row = pd.DataFrame(odataframe.corr().loc['signal']).T
    
    # If abs corr
    if abs_corr:
        corr_row = np.abs(corr_row)
    
    # Sort array
    if sort:
        sorted_columns = corr_row.columns[ np.argsort(corr_row.values[0]) ]
        c_ord = 1 if invert_cbar else -1
        corr_row = corr_row[sorted_columns[::c_ord]]
        
    
    # Remove the signal-signal element
    columns = list(corr_row.columns)
    columns.remove('signal')
    corr_row = corr_row[columns]
    
    # Get minimum and maximum for colourbar
    vmin, vmax = c_range
    if vmin is None:
        vmin = corr_row.values.min()
    if vmax is None:
        vmax = corr_row.values.max()
    
    # Make column labels plot ready
    plot_row = corr_row.copy()
    plot_row.columns = AP.latex_label(plot_row.columns)
    plot_row = plot_row[ plot_row.columns[::-1] ]
    plot_row = plot_row.rename(index={'signal': 'Signal'})
    
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))

    # Draw the heatmap with the correct aspect ratio
    sns.heatmap(plot_row, cmap=AP.std_cmap, center=0, square=True,
                vmin=vmin, vmax=vmax, linewidths=0.0,
                cbar_kws={"shrink": 1.0, "orientation": "horizontal",
                          'pad':cbar_pad, 'aspect':35})
    
    # Make figure transparent
    fig.patch.set_facecolor('None')
    ax.patch.set_facecolor('None')
    
    # Set 'Signal' spacing, fonttype and alignment
    ax.set_yticklabels([l._text + obsv_spad
                        for l in ax.get_yticklabels()],
                       fontproperties=AP.std_font, va="center")
    
    # Setup x-minor ticks and x-axis fontype and alingments
    ax.set_xticklabels(ax.get_xticklabels(), minor=False, ha="center")
    ax.set_xticks(ax.get_xticks(), minor=True)
    ax.set_xticklabels(ax.get_xticklabels(), minor=True, ha="center")
    [mn.set_x(mj._x) for mj, mn in zip(ax.get_xticklabels(minor=False),
                                       ax.get_xticklabels(minor=True))]
    # Alternate major and minor ticks
    for n, tick in enumerate(ax.get_xaxis().get_major_ticks()):
        if n % 2 == 0: tick.set_visible(False)
        else: tick.set_visible(True)
    for n, tick in enumerate(ax.get_xaxis().get_minor_ticks()):
        if n % 2 != 0: tick.set_visible(False)
        else: tick.set_visible(True)
    
    # Set y-ticklabel and fontsize
    ax.tick_params(axis='y', which='major',
                   labelrotation=labelrot, labelsize=fontsize*0.94)
    # Set x-ticklabels and fontsize
    ax.tick_params(axis='x', which='major', direction='out',
                   length=6, width=1, pad=0, zorder=100,
                   labelrotation=labelrot, labelsize=fontsize,
                   top=False, bottom=True, labeltop=False, labelbottom=True)
    ax.tick_params(axis='x', which='minor', direction='out',
                   length=7.5, width=1, pad=0, zorder=100,
                   labelrotation=labelrot, labelsize=fontsize,
                   top=True, bottom=False, labeltop=True, labelbottom=False)
    
    # Get colorbar axes
    cax = fig.get_axes()[-1]
    
    # Set colorbar ticklabels position, fontsize, fontstyle and rotation
    cax.xaxis.set_ticks_position('top')
    cax.tick_params(axis='x', which='major',
                    labelrotation=labelrot, labelsize=fontsize)
    
    # Set Correlation label
    cax_set_label(corr_spad + 'Correlation', fontproperties=AP.std_font,
                  ha='center', labelpad=cbar_lpad, fontsize=fontsize)
    
    # Add +/- space formatting to axes
    if invert_cbar:
        cax.invert_xaxis()        
    AP.format_ticklabel_signspacing(cax, which='x')
    
    # Save fig:
    if fig_name is not None:
        DW._collision_check(collision)
        path = AP.get_image_file_paths(fig_name, '.png',
                                       'observables_analysis', collision)
        fig.savefig(path, transparent=True, bbox_inches='tight',
                    dpi=fig.dpi)
    
    # Return figure and correlation row with original labels
    if rtn_corr:
        return fig, corr_row
    return fig
    