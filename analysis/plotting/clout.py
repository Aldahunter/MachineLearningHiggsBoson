"""Classifier Outputs - Plotting functions for the Machine Learning \
Outputs from the classifiers"""

from collections import OrderedDict
from typing import Dict, Any, Union, Iterable

import matplotlib.collections as mc
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from numpy.core._multiarray_umath import ndarray
from scipy.stats import norm

import analysis.metrics as AME
import analysis.misc as AMI
import analysis.plotting as AP
import dwrangling as DW


### Functions ###
def plot_confusion_matrix(labels, labels_predicted, p_thres=0.5,
                          collision=None, fig_name=None, rtn_cbar=False,
                          cmap=None, cbar_visible=True, label_fontsize=25,
                          cm_fontsize=40, _axes=None, swap_text_c=False):
    """Plot the confusion matrix as a heatmap.
    
    Parameters:
     - labels: A :class:`list` of :class:`bool` values, determining
               the events class: `True` -> 'signal', `False` ->
               'background';
     - labels_predicted: A :class:`list` of :class:`float` values,
                         giving the classifier's predictions for the
                         events in 'labels', must have the same length
                         as 'labels';
     - p_thres: A :class:`float` between [0,1] for the prediction
                threshold between 'signal' and 'background'. [Default
                '0.5']; 
     - collision: A :class:`str` denoting the collision, must be in
                  {DW.collisions}, and must be given if 'fig_name' is not
                  'None'. [Default 'None'];
     - fig_name: A :class:`str` if given will save the plot with this name,
                 if 'None' is given, it will not be saved. [Default 'None'];
     - rtn_cbar: A :class:`bool` determining whether to return the
                 figure as well as the colorbar;
     - cmap_r: A :class:`bool` determining whether the colormap is
               reversed. [Default 'False'];
     - cmap_middle: A :class:`bool` determining whether the colormap
                    has a third color at the centre. [Default 'False'];
     - cbar_visible: A :class:`bool` determining whether the colorbar
                     is displayed. [Default 'True'];
     - label_fontsize: A :class:`float` giving the fontsize for the
                       ticklabels and axis labels. [Default '30'];
     - cm_fontsize: A :class:`float` giving the fontsize for the
                    confusion matrix annotations. [Default '50'].
    
    Returns:
     - figure: A :class:`matplotlib.figure` instance of the plot;
     - cbar: A :class:`matplotlib.colorbar` instance from the plot.
             [Only returned if 'rtn_cbar' is `True`]."""
    
    # Set-up classes and colormap
    classes = ['Signal', 'Background']
    cmap = AP.std_cmap if (cmap is None) else cmap
    
    # Compute confusion matrix
    cm = AME.sk_confusion_matrix(labels, labels_predicted,
                                 p_threshold=p_thres)[::-1,::-1]
    
    # Normalize confusion matrix if requested
    norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create figure and add confusion matrix plot
    if _axes is None:
        fig, ax = plt.subplots(figsize=(9,9))
    else:
        fig, ax = _axes.figure, _axes
    im = ax.imshow(norm_cm, interpolation='nearest', cmap=cmap)
    
    # Add axes to the right of confusion matrix plot
    cax = make_axes_locatable(ax).append_axes("right", size="7%", pad="0%")
    
    # Add colorbar to new axis
    cbar = colorbar(im, cax=cax)
    
    # Draw figure to get ticks and ticklabels
    fig.canvas.draw()
    
    # Add axes ticks, ticklabels and axis labels
    ax.tick_params(axis='both', which='major', direction='out',
                   top=False, labeltop=False,
                   bottom=False, labelbottom=True,
                   left=False, labelleft=True,
                   right=False, labelright=False)
    
    # Set axes ticks
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    
    # Set axes ticklabels
    ax.set_xticklabels(classes, fontproperties=AP.std_font,
                       rotation=0, fontsize=label_fontsize,
                       va='top', ha='center')
    ax.set_yticklabels(classes, fontproperties=AP.std_font,
                       rotation=90, fontsize=label_fontsize,
                       va='center', ha='right')
    
    # Set axis labels
    ax.set_xlabel("Predicted",
                  fontproperties=AP.std_font, fontsize=label_fontsize)
    ax.set_ylabel("Event",
                  fontproperties=AP.std_font, fontsize=label_fontsize)
    
    # Format Colorbar's y-axis ticklabels
    AP._format_ticklabel_signspacing(cbar.ax, which='y',
                                     fontsize=label_fontsize)

    # Loop over confusion matrix
    thresh = cm.max() / 2.0  # Threshold for text color
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            
            # Determine annotation text color
            if swap_text_c:
                color = "white" if (cm[i,j] > thresh) else "black"
            else:
                color = "white" if (cm[i,j] < thresh) else "black"
            
            # Add annotation for each element in the matrix
            ax.text(j, i,
                    f"${cm[i, j]:d}$\n$({100*norm_cm[i, j]:.1f}\%)$",
                    fontsize=cm_fontsize, color=color,
                    ha="center", va="center")

    # Visibile color bar
    cbar.ax.set_visible(cbar_visible)
    
    # Remove unecessary whitespace
    fig.patch.set_facecolor('None')
    ax.patch.set_facecolor('None')
    cax.patch.set_facecolor('None')
    
    # Save fig:
    if fig_name is not None:
        DW._collision_check(collision)
        path = AP.get_image_file_paths(fig_name, '.png',
                                       'classifier_analysis', collision)
        fig.savefig(path, transparent=True, bbox_inches='tight',
                    dpi=fig.dpi)
    
    if _axes is not None:
        return ax    
    
    # Return figure and colorbar
    if rtn_cbar:
        return fig, cbar
    return fig


def plot_results(classifier, result_pairs, p_threshold=0.5, n_bins='auto',
                 csgn=None, cbkg=None, alpha=0.5, hist=False, fontsize=25.5,
                 observables="", obs_pos=(0,0), cm_pos=(0,0), cm_scale=1,
                 y_max=None, xlabelpad=0.0, ylabelpad=0.0, ydifpad=0.0,
                 x_range=None, bin_edges='auto', **plotstyles):
    
    # Check results dictionary
    AMI._check_results_dict(result_pairs)
    
    # Check p_threshold is a valid value
    if not (0 < p_threshold < 1):
        raise ValueError("'p_threshold' must be between 0 and 1, " +
                         f"not {p_threshold}.")
    
    n_results = len(result_pairs)
    if n_results > 1:
        ax_dimensions = [0,0.25,0.875,0.75]
    else:
        ax_dimensions = [0.05,0.05,0.9,0.9]
    
    # Get bin midpoints
    if hist:
        # Get leaf probabilities
        centers = set()
        for labels, labels_pred in result_pairs.values():
            centers |= set(labels_pred)
        centers = sorted(centers)
        bin_edges = [(p0+p1)/2 for p0, p1 in zip(centers[:-1], centers[1:])]
        bin_edges = np.array([0.0,] + bin_edges + [1.0,])
    
    else:
        all_results = np.concatenate([type_results[1] for type_results
                                      in result_pairs.values()])
        bin_edges = np.histogram_bin_edges(all_results, bins=n_bins,
                                           range=x_range)
        centers = np.array([(l + u) / 2
                            for l, u in zip(bin_edges[:-1], bin_edges[1:])])
        
    
    # Set-up figure and main axis
    fig = plt.figure(figsize=(12.00,10.00))
    ax = fig.add_axes(ax_dimensions)
    
    # Get colors for dictionary items
    csgn = AP.std_cmap(1.0) if (csgn is None) else csgn
    cbkg = AP.std_cmap(0.0) if (cbkg is None) else cbkg
    
    # Get plotting asthetics if not given use standard
    _plotstyles = {'styles':None, 'markers':None, 'square':'mid'}
    _plotstyles.update(plotstyles)
    styles, markers = AP._get_plotting_style(result_pairs.keys(),
                                                      **_plotstyles)
    
    # Iterate through train and test data
    histograms: OrderedDict[Any, Dict[str, Union[ndarray, Iterable, int, float]]] = OrderedDict()
    for n, (key, type_results) in enumerate(result_pairs.items()):
        
        # Seperate signal and background
        is_signal = type_results[0]
        signal_probabilities = type_results[1]
        results = {'s': signal_probabilities[is_signal == True],
                   'b': signal_probabilities[is_signal == False]}
        
        # Create bins for histograms
        histogram = {'s': np.histogram(results['s'],
                                       bins=bin_edges, density = True)[0],
                     'b': np.histogram(results['b'],
                                       bins=bin_edges, density = True)[0]}
        histograms[key] = histogram
        
        # Find bin widths and centers
        widths = bin_edges[1:] - bin_edges[:-1]
        
        # Plot histograms
        if n == 0:
            ax.bar(bin_edges[:-1], histogram['b'], width=widths,
                   align='edge', color=cbkg, label='Background',
                   alpha=alpha, zorder=2*n)
            ax.bar(bin_edges[:-1], histogram['s'], width=widths,
                   align='edge', color=csgn, label='Signal',
                   alpha=alpha, zorder=2*n+1)
        else:
            ax.plot(centers, histogram['b'], #marker=markers[key],
                    ls=styles[key], color=cbkg, #markersize=13,
                    label=f'Background [{key}]', alpha=1.0, zorder=2*n)
            ax.plot(centers, histogram['s'], #marker=markers[key],
                    ls=styles[key], color=csgn, #markersize=13,
                    label=f'Signal [{key}]', alpha=1.0, zorder=2*n+1)
    
    # Add p_threshold line to plot
    ax.axvline(p_threshold, linestyle='--', color='black')
    
    if n_results > 1:
        # Plot residual deviations between first pair and trailing pairs
        ax_res = fig.add_axes([0,0,0.875,0.25], sharex=ax)
        
        c = {'b':cbkg, 's':csgn}
        inital, deviations = list(histograms.keys())[0], {}
        for sb in ['b', 's']:
            
            deviation, N = 0.0, 0.0
            for key in list(histograms.keys())[1:]:
                deviation += histograms[key][sb] - histograms[inital][sb]
                N += 1.0
            deviation /= N
            deviations[sb] = deviation
            
            ax_res.bar(bin_edges[:-1], deviation, width=widths, alpha=.5,
                       align='edge', color=c[sb], zorder=0)
        
        ax_res.axhline(color='black', linestyle='-', zorder=1)
        ax_res.axvline(p_threshold, linestyle='--', color='black', zorder=1)
            
        
        ax_dist = fig.add_axes([0.875,0,0.125,0.25], sharey=ax_res)
        ymin, ymax = ax_res.get_ylim()
        for sb, deviation in deviations.items():
            mu, std = norm.fit(deviation)

            y = np.linspace(ymin, ymax, 100)
            p = norm.pdf(y, mu, std)

            ax_dist.plot(p, y, ls='-', color=c[sb],
                         alpha=1, linewidth=3, zorder=1)
        ax_dist.axhline(color='black', linestyle='-', zorder=0)
    
    # Add observable labels as text
    xobs, yobs = obs_pos
    if observables:
        observables = ' & '.join( AP._latex_label(observables) )
    AP._add_text(0.390 + xobs, 0.960 + yobs, 'Observables:\n'+observables,
                 fontsize=fontsize*0.9, axes=ax)
    
    # Add confusion matrix
    if isinstance(cm_pos, tuple) and (len(cm_pos) == 2):
        (xcm, ycm), scale = cm_pos, 0.325 * cm_scale
        cm_ax = fig.add_axes([0.315 + xcm, 0.487 + ycm, scale, scale]) 
        plot_confusion_matrix(*list(result_pairs.values())[0],
                              p_thres=p_threshold, _axes=cm_ax,
                              cbar_visible=False, cm_fontsize=fontsize*0.85,
                              label_fontsize=fontsize*0.7)
    
    # Set axes limits
    if x_range is not None:
        ax.set_xlim(*tuple(x_range))
    ax.set_ylim(0.0, y_max)
    if n_results > 1:
        ax_dist.set_xlim(0)
        ax_dist.set_ylim(ymin, ymax)
    
    # Draw figure to get ticks and ticklabels
    fig.canvas.draw()
    
    # Add axes labels
    s_classifier = classifier.__str__().split('(')[0]
    ax.set_ylabel("Normalised Counts per Bin", fontproperties=AP.std_font,
                  fontsize=fontsize, labelpad=ylabelpad + 25 + ydifpad)
    if n_results > 1:
        ax_res.set_ylabel("Deviation from Test",
                          fontproperties=AP.std_font, fontsize=fontsize,
                          labelpad=ylabelpad)
        ax_res.set_xlabel(f"Classifier Output [{s_classifier}]",
                          fontproperties=AP.std_font, fontsize=fontsize,
                          labelpad=ylabelpad)
        ax_dist.set_xlabel("Normalised Deviation",
                           fontproperties=AP.std_font, fontsize=fontsize,
                           labelpad=ylabelpad)
    else:
        ax.set_xlabel(f"Classifier Output [{s_classifier}]",
                      fontproperties=AP.std_font, fontsize=fontsize,
                      labelpad=ylabelpad)
    
    # Set axes tick parameters
    AP._format_ticklabel_signspacing(ax,which='both',fontsize=fontsize*0.9)
    AP._format_tickmarks(ax, xmajor='none', ymajor='left',
                         xminor='both', yminor='right')
    if n_results > 1:
        AP._format_ticklabel_signspacing(ax_res, which='both',
                                         fontsize=fontsize*0.9)
        AP._format_tickmarks(ax_res, xmajor='bottom', ymajor='left',
                             xminor='top', yminor='right')
        AP._format_ticklabel_signspacing(ax_dist, which='both',
                                         fontsize=fontsize*0.9)
        AP._format_tickmarks(ax_dist, xmajor='bottom', ymajor='none',
                             xminor='top', yminor='both')
        
        # Remove ticklabels ontop of one another
        yticks = list(ax.get_yticklabels())
        yticks[0]._text = ''
        ax.set_yticklabels(yticks)
        xticks = list(ax_res.get_xticklabels())
        xticks[-1]._text = ''
        ax_res.set_xticklabels(xticks)
    
    # Setup legend
    prop = AP.std_font.copy()
    prop._size = fontsize*0.8
    ax.legend(loc="upper left", prop=prop, framealpha=0.8)
    
    # Reset axes limits
    if x_range is not None:
        ax.set_xlim(*tuple(x_range))
    ax.set_ylim(0.0, y_max)
    if n_results > 1:
        ax_dist.set_xlim(0)
        ax_dist.set_ylim(ymin, ymax)
    
    # Show figure
    return fig


def ROC_curve(result_pairs, n_cuts=100, alpha=0.5, observables="",
              fontsize=30, xlabelpad=5, ylabelpad=-18, **plotstyles):
    """Plots the ROC curve of the classifiers predicted results.
    
    Parameters:
     - labels: A :class:`numpy.array` of length 'n', where a label is:
               'signal':`True` or 'background':`False`;
     - labels_pred: A :class:`numpy.array` of length n, where a label_pres
                    is: the probability output of the algortihm range
                    [0, 1.0];
     - train_labels: A :class:`numpy.array` of length 'n', where a label
                     is: 'signal':`True` or 'background':`False`.
                     [Optional, must be given with 'train_labels_pred'];
     - train_labels_pred: A :class:`numpy.array` of length n, where a
                          label_pres is: the probability output of the
                          algortihm range [0, 1.0]. [Optional,
                          must be given with 'train_labels'];
     - n_cuts: An :class:`int` representing the number of p_threshold cuts
               used (this increases the resolution of graph). [Default
               '100'];
     - observables: A :class:`str` used as the Observables annotation on the
                    graph;
     - fontsize: A :class:`float` giving the fontsize for the ticklabels,
                 axis labels and annotations. [Default '30'];
     - xlabelpad: A :class:`float` giving the x-axis label padding. [Default 
                  '5'];
     - ylabelpad: A :class:`float` giving the y-axis label padding. [Default 
                  '-18'].
    
    Returns:
     - figure: A :class:`matplotlib.figure` instance of the plot."""
    
    # Check results dictionary
    AMI._check_results_dict(result_pairs)
    
    # Create figure and axes instances
    fig = plt.figure(figsize=(10.0, 10.0))
    ax = fig.add_axes([0,0,1,1])
    
    # Get plotting asthetics if not given use standard
    _plotstyles = {'colors':None, 'styles':None, 'hatches':None}
    _plotstyles.update(plotstyles)
    colors, styles, hatches = AP._get_plotting_style(result_pairs.keys(),
                                                     **_plotstyles)
    
    # Iterate over test/train results
    for n, (key, type_results) in enumerate(result_pairs.items()):
        
        # Determine results plot order
        zorder = len(result_pairs) - n
    
        # Get (x, y) coordinates
        scatter = [(0,0),]
        for p_cut in np.linspace(0, 1, n_cuts+1):
            cm = AME.confusion_matrix(*type_results, p_cut)
            x, y = AME.fpr(*cm), AME.recall(*cm)
            scatter.append((x, y))
        scatter += [(1,1),]
        
        # Sort coordinates by x-axis values
        scatter = sorted(scatter, key=lambda xy: abs(xy[0]))
        
        # Plot ROC curve for the results
        ax.plot(*zip(*scatter), color=colors[key], ls=styles[key],
                zorder=zorder, label = key)
        
        # Calculate the Area Under the Curve
        auc = AME.auc(*type_results)
        
        # Plot AUC
        fill = ax.fill_between(*zip(*scatter), alpha=alpha, zorder=zorder,
                               facecolor=AP.lighten_color(colors[key], 1/3), 
                               edgecolor = colors[key], hatch=hatches[key],
                               label=f"{key}: ${auc:.3f}$", linestyle='-')
    
    # Plot random guess line
    ax.plot((0,1),(0,1), ls='--', color='black', zorder=len(result_pairs))
    
    # Draw figure to get ticks and ticklabels
    fig.canvas.draw()
    
    # Add Observables annotation
    if observables:
        AP._add_text(0.98, 0.975,
                     'Observables:\n' + observables.replace(' &',','),
                     axes=ax, fontsize=fontsize*0.8, halign='right')
    
    # Setup axes ticks, ticklabels and labels
    ax.set_xlabel("False Positive Rate", fontproperties=AP.std_font,
                  fontsize=fontsize, labelpad=xlabelpad)
    ax.set_ylabel("True Positive Rate", fontproperties=AP.std_font,
                  fontsize=fontsize, labelpad=ylabelpad)
    
    AP._format_ticklabel_signspacing(ax,  which='both', fontsize=fontsize)
    AP._format_tickmarks(ax, xmajor='bottom', ymajor='left',
                         xminor='top', yminor='right')
    
    # Setup axes limits
    ax.set_xlim(-0.001, 1.001)
    ax.set_ylim(-0.001, 1.001)
    
    # Setup legend
    prop = AP.std_font.copy()
    prop._size = fontsize*0.8
    ax.legend(loc="lower right", prop=prop, framealpha=0.8)
    
    # Return figure
    return fig


# def ROC_displacement(labels, labels_pred, *train_results, n_cuts=100,
#                      return_max=False, text="Observable", fontsize=25):
#
#     AMI.check_label_params(labels, labels_pred, *train_results)
#
#     full_results = {'Test': (labels, labels_pred)}
#     if not train_results:
#         full_results['Train'] = train_results
#
#     fig = plt.figure(figsize=(10.67,7.33))
#     ax = fig.add_axes([0,0,1,1])
#
#     for key, type_results in full_results.items():
#
#         ls = '-' if key == 'Test' else '--'
#         zorder = 2 if key == 'Test' else 1
#
#         displacements = [(0,0),]
#         for p_cut in range(n_cuts+1):
#             cm = AME.confusion_matrix(*type_results, p_cut / 100.0)
#             x, y = AME.fpr(*cm), AME.recall(*cm)
#             disp = np.sqrt(0.5*(x**2 + y**2) - x*y)
#             disp *= 100 if (y >= x) else -100
#             displacements.append((p_cut/100, disp))
#
#         displacements = sorted(displacements, key=lambda xy: abs(xy[0]))
#         displacements.append((1,0))
#         if key == 'Test':
#             max_disp = max(displacements, key=lambda xy: abs(xy[1]))
#
#         ax.plot(*zip(*displacements), ls=ls, color='red',
#                 label=key, zorder=zorder)
#
#     ax.plot((0, 1), (0,0), ls='--', color='grey', zorder=0)
#
#     ax.text(0.64, 0.85, 'Paramters:\n'+text.replace(' &',','),
#             fontsize=fontsize)  # 0.55, 0.85
#
#     ax.set_title("ROC Disp", fontsize=fontsize)
#     ax.set_ylabel("Guess Displacement (%)", fontsize=fontsize)
#     ax.set_ylim(0.0)
#     ax.set_xlabel("Probability Threshold [$P_{cut}$]", fontsize=fontsize)
#     ax.set_xlim(-0.01, 1.01)
#     ax.tick_params(axis='both', labelsize=fontsize)
#     ax.legend(loc="lower right", fontsize=fontsize)
#
#
#     if return_max == True:
#         return fig, max_disp
#     return fig


def SNRatio(result_pairs, n_thresholds=100, leaves_thresholds=False,
            p_threshold=0.5, observables="", fontsize=25, xlabelpad=5,
            ylabelpad=0, x_range=None, **plotstyles):
    """Plots the Signal-to-Noise Ratio curve of the classifiers predicted \
    results.
    
    Parameters:
     - labels: A :class:`numpy.array` of length 'n', where a label is:
               'signal':`True` or 'background':`False`;
     - labels_pred: A :class:`numpy.array` of length n, where a label_pres
                    is: the probability output of the algortihm range
                    [0, 1.0];
     - train_labels: A :class:`numpy.array` of length 'n', where a label
                     is: 'signal':`True` or 'background':`False`.
                     [Optional, must be given with 'train_labels_pred'];
     - train_labels_pred: A :class:`numpy.array` of length n, where a
                          label_pres is: the probability output of the
                          algortihm range [0, 1.0]. [Optional,
                          must be given with 'train_labels'];
     - n_thresholds: An :class:`int` representing the number of p_threshold
                     cuts used (this increases the resolution of graph).
                     [Default '100'];
     - observables: A :class:`str` used as the Observables annotation on the
                    graph;
     - fontsize: A :class:`float` giving the fontsize for the ticklabels,
                 axis labels and annotations. [Default '25'];
     - xlabelpad: A :class:`float` giving the x-axis label padding. [Default 
                  '5'];
     - ylabelpad: A :class:`float` giving the y-axis label padding. [Default 
                  '0'].
    
    Returns:
     - figure: A :class:`matplotlib.figure` instance of the plot."""
    
    # Check results dictionary
    AMI._check_results_dict(result_pairs)
    n_results = len(result_pairs)

    # Create probability threshold range
    x0, x1 = 0.0, 1.0
    if x_range is not None:
        x0, x1 = x_range
    p_cuts = np.linspace(x0, x1, n_thresholds+1)
    if leaves_thresholds:
        
        # Iterate over all predicted values
        leaves = {}
        for _, labels_pred in result_pairs.values():
        
            # Add leaf probabilities set to total leaf probabilities set
            leaves.update({leaf: 0 for leaf in set(labels_pred)})
        
        # Get union of thresholds and leaves
        p_cuts = set(p_cuts) | set(leaves)
        p_cuts = sorted(p_cuts)
    
    # Get plotting asthetics if not given use standard
    _plotstyles = {'colors':None, 'styles':None}
    _plotstyles.update(plotstyles)
    colors, styles = AP._get_plotting_style(result_pairs.keys(),
                                            **_plotstyles)

    # Create figure and axis instances
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    # Iterate over test/train results
    for n, (key, type_results) in enumerate(result_pairs.items()):
        
        # Calculate SNRs for every probability threshold 
        sns = []
        for p_cut in p_cuts:
            sn = AME.S_B_ratio(*type_results, p_threshold=p_cut)
            sns.append(sn)
            
            # If threshold is a leaf probability, and the SNR is
            # greater than the current value, update the SNR
            if leaves_thresholds and (p_cut in leaves):
                leaf_sn = leaves[p_cut]
                leaves[p_cut] = sn if (sn > leaf_sn) else leaf_sn

        # Plot testing signal-noise curve
        ax.plot(p_cuts, sns, color=colors[key], label=key,
                linestyle=styles[key], zorder=(n_results - n))
    
    # Obtain the probability threshold coordinates from the second set of
    # results, iif just one set use the first
    labels, labels_pred = list(result_pairs.values())[n_results > 2]
    thres = (p_threshold,  AME.S_B_ratio(labels, labels_pred, p_threshold))

    # Plot the best signal-noise and probability threshold lines
    linestyle = [6, 2, 3, 2, 3, 2]
    line = ax.plot((thres[0],)*2, (0, thres[1]), color='black',
                   zorder=0, label='P$_{Threshold}$', dashes=linestyle)
    line = ax.plot((0, thres[0]), (thres[1],)*2, color='black', zorder=0,
                   dashes=linestyle)
    
    
    # Iterate through leaves and plot line the top SNR curve
    if leaves_thresholds:
        leaves_ls = [6, 2, 1, 2, 1, 2]
        leaves = list(leaves.items())
        p, snr = leaves[0]
        line = ax.plot((p,p), (0,snr), color='grey', dashes=leaves_ls,
                       zorder=0, label='Leaves')
        for p, snr in leaves[1:]:
            line = ax.plot((p,p), (0,snr), color='grey',
                           dashes=leaves_ls, zorder=0)
            
    # Set the axes limits
    if x_range is not None: ax.set_xlim(*tuple(x_range))
    else: ax.set_xlim(-0.001, 1.001)
    ax.set_ylim(0.0)

    # Draw figure to get ticks and ticklabels
    fig.canvas.draw()
    
    # Add Observables annotation
    if observables:
        AP._add_text(0.98, 0.975,
                     'Observables:\n' + observables.replace(' &',','),
                     axes=ax, fontsize=fontsize*0.8, halign='right')
    
    # Setup axis ticks and ticklabels
    AP._format_ticklabel_signspacing(ax, fontsize=fontsize)
    AP._format_tickmarks(ax, xmajor='bottom', ymajor='left',
                         xminor='top', yminor='right')
    
    # Setup axis labels
    ax.set_xlabel('$P_{Threshold}$', fontproperties=AP.std_font,
                  fontsize=fontsize, labelpad=xlabelpad)
    ax.set_ylabel('Signal-Noise Ratio', fontproperties=AP.std_font,
                  fontsize=fontsize, labelpad=ylabelpad)

    # Setup legend
    prop = AP.std_font.copy()
    prop._size = fontsize*0.8
    ax.legend(loc="lower right", prop=prop, framealpha=0.8)
    
    # Set the axes limits
    if x_range is not None: ax.set_xlim(*tuple(x_range))
    else: ax.set_xlim(-0.001, 1.001)
    ax.set_ylim(0.0)
    
    # Return figure
    return fig


def parallel_coord_path(path_sample_generator, path_depth, alpha=None,
                        xfontsize=15, yfontsize=13, xlabelpad=15,
                        split_xpad=0.02, split_ypad=0.005, linewidths=2):
    
    # Check alpha is a function which takes the impurity as an input
    if alpha is None:
        alpha =  lambda imp: (-1.08 * imp) + 0.55
    if not callable(alpha):
        raise ValueError("'alpha' must be 'None' or a callable funtion, " +
                         "which takes the node's impurity as a 'float' " +
                         "and returns a value between [0,1].")
    
    # Check kwargs are all numerical
    for str_param, param in {'xfontsize': xfontsize,
                             'yfontsize': yfontsize,
                             'xlabelpad': xlabelpad,
                             'split_xpad': split_xpad,
                             'split_ypad': split_ypad}.items():
        
        if not isinstance(param, (int, float)):
            raise ValueError(f"'{str_param}' must be of type 'int' or " +
                             f"'float', not '{type(param)}'.")
    
    # Generate normalise fns, y-ticks, y-labels and fontsize
    def get_norm_fn_yparams(dataframe, observable):
        # If signal, change y label scale and positions so signal
        # and background are evenly spaced in the middle
        if observable == 'signal':
            norm_fn = AMI.create_normalise_fn(np.array([-1, 2]))
            y_ticks = norm_fn(np.array([0, 1]))
            y_labels = ['B', 'S']
            math = False
            fontsize = int(yfontsize * 1.25)
        # Otherwise scale observable data normally
        else:
            df_values = dataframe[observable].values
            norm_fn = AMI.create_normalise_fn(df_values)
            y_ticks = np.array([0, 1])
            y_labels = [f"{df_values.min():.3f}", f"{df_values.max():.3f}"]
            math = True
            fontsize = yfontsize
        return norm_fn, y_ticks, y_labels, math, fontsize
    
    # Generate Parallel Coordinates figure
    fig, axs = AP.get_parallel_coord_axes(path_depth)

    # Get first node
    cur, nxt = {}, {}
    obs, bra, split, impu, df = next(path_sample_generator)
    cur['obs'], cur['bra'], cur['split'] = obs, bra, split
    cur['impu'], cur['df'] = impu, df

    # Create functions for initial
    norm_fn, y_ticks, y_labels, math, fontsize = \
                                  get_norm_fn_yparams(cur['df'], cur['obs'])
    
    # Set first axis ticks to be longer
    axs[0].tick_params(axis='y', direction='out', length=12.5, width=5)
    
    # Iterate through nodes and axes
    for ax, next_ in zip(axs, path_sample_generator):
    
        # Get current and next node values and dataframes
        nxt['obs'], nxt['bra'], nxt['split'], nxt['impu'], nxt['df'] = next_

        # Change axis ticks and labels
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontproperties=AP.std_font,
                           fontsize=fontsize)
        AP._format_ticklabel_signspacing(ax, which='y', math=math)
        ax.set_xlim((0, 1))
        ax.set_xlabel(AP._latex_label(cur['obs']), x=0, y=1,
                      fontproperties=AP.std_font, fontsize=xfontsize,
                      labelpad=xlabelpad)
        ax.xaxis.set_label_position('bottom')


        # If node can split, add partiton data and add partition line
        if cur['bra'] is not None:
            branch = 0 if (cur['bra'] == '<') else 1
            cur['df'] = cur['df'].partition(cur['obs'],
                                            cur['split'])[branch]

            ax.axhline(y = norm_fn(cur['split']), xmin = 0, xmax = 0.49,
                       ls = '--', c = 'black', zorder = 3)

        # Get current node singal and background y values
        cur['sys'] = cur['df'].get_s()[ cur['obs'] ].values
        cur['bys'] = cur['df'].get_b()[ cur['obs'] ].values


        # Generate normalise fns and y-ticks and -labels
        nxt_norm_fn, y_ticks, y_labels, math, fontsize = \
                                  get_norm_fn_yparams(nxt['df'], nxt['obs'])

        # Get current node singal and dataframe y values
        nxt['sys'] = nxt['df'].get_s()[ nxt['obs'] ].values
        nxt['bys'] = nxt['df'].get_b()[ nxt['obs'] ].values

        # If next node can split, add partiton line and label
        if nxt['bra'] is not None:
            split_val = nxt['split']
            norm_split_val = nxt_norm_fn(split_val)

            ax.axhline(y = norm_split_val, xmin = 0.51, xmax = 1,
                       ls = '--', c = 'black', zorder = 3)

            ax.text(y=(norm_split_val + split_ypad), x=(1.0 - split_xpad),
                    s=f"${split_val:.3f}$", fontproperties=AP.std_font,
                    fontsize=fontsize, zorder=4,
                    horizontalalignment='right', 
                    verticalalignment='baseline')

        # Create line start and end coordinate functions
        line_start = lambda y_0: (0.0, y_0)
        line_end = lambda y_1: (1.0, y_1)

        # Obtain signal segments
        sstarts = map(line_start, norm_fn(cur['sys']))
        sends = map(line_end, nxt_norm_fn(nxt['sys']))
        ssegments = zip(sstarts, sends)

        # Obtain background segments
        bstarts = map(line_start, norm_fn(cur['bys']))
        bends = map(line_end, nxt_norm_fn(nxt['bys']))
        bsegments = zip(bstarts, bends)


        # Obtain line collections for signal and background
        scolor = mcolors.to_rgba('orange', alpha = alpha( cur['impu'] ))
        slc = mc.LineCollection(ssegments, colors = scolor,
                                linewidths = linewidths, zorder = 2)
        bcolor = mcolors.to_rgba('blue', alpha = alpha( cur['impu'] ))
        blc = mc.LineCollection(bsegments, colors = bcolor,
                                linewidths = linewidths, zorder = 2)

        # Plot line collections
        ax.add_collection(slc)
        ax.add_collection(blc)

        # Swap next to current
        cur = nxt.copy()
        norm_fn = nxt_norm_fn

    # Configure final axis 
    axs[-1].set_ylim((0, 1))
    axs[-1].set_yticks(y_ticks)
    axs[-1].set_yticklabels(y_labels, fontproperties=AP.std_font,
                            fontsize=fontsize)
    axs[-1].tick_params(axis='y', direction='out', length=10, width=5)
    axs[-1].set_xlim((0, 1))
    axs[-1].set_xlabel(AP._latex_label(cur['obs']), x = 1, y = 1,
                       fontproperties = AP.std_font, fontsize = xfontsize,
                       labelpad = xlabelpad)
    

    # Create twin for bottom right label
    twin = AMI.phantom_twin_xaxis(axs[-1])
    twin.set_xlabel(AP._latex_label(cur['obs']), x = 1, y = 1,
                    fontproperties = AP.std_font, fontsize = xfontsize,
                    labelpad = xlabelpad)
    twin.xaxis.set_label_position('bottom')

    # Mirror x-axis labels ontop
    AMI.mirror_axes_above(axs, labelpad = xlabelpad + 2)
    
    # Return figure
    return fig
    
 