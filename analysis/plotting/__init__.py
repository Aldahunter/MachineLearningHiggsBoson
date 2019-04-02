"""Plotting - Module for Analysis Plots of Machine Learning Classifier's and their outputs."""

import os
import colorsys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
from itertools import cycle
from string import ascii_letters

import dwrangling as DW
import analysis.metrics as AME
import analysis.misc as AMI
from analysis.plotting._stdcmap import StdCmap



### Data ###
comma_space = ', '
matplotlib.rcParams['mathtext.fontset'] = 'cm'
std_font = fm.FontProperties(fname=r'C:\Windows\Fonts\computermodern.ttf')
std_figsize = (9, 9)


### Setup standard color map ###
_cmap_sequence = ['navy', 'blue',
                  0.20, 'blue', 'royalblue',
                  0.40, 'royalblue', 'white',
                  0.50, 'white', 'orange',
                  0.60, 'orange', 'darkorange',
                  0.80, 'darkorange', 'orangered']
std_cmap = StdCmap.from_sequence('Standard Colormap', _cmap_sequence)


### Functions ###
def get_image_file_paths(name, ext, *directories):
    f"""Returns the file location as a string, for both signal and \
    background.
    
    Parameters:
     - name: The name of the file as type 'str';
     - ext: The file extension/type as type 'str', e.g. 'jpg';
     - directories: The directories for the file as 'str's in the correct
                    order."""

    # Validate and clean inputs
    name, ext = name.strip(), ext.strip()
    directories = [dir_.strip() for dir_ in directories]
    ext = ext if (ext[0] == '.') else '.' + ext

    # Obtain the file's path and return
    file_loc = os.path.join(os.getcwd(),
                            "Figures", *directories, name + ext)
    return file_loc


def _add_text(x, y, label, axes=plt.gca(), fontsize=20, legend_style=True,
             halign='left', valign='top', bbox_args={}, **kwargs):
    setup = {'horizontalalignment': halign, 'verticalalignment': valign,
             'weight': 'semibold', 'transform': axes.transAxes,
             'fontsize': fontsize}
    setup.update(kwargs)
    
    string = ',\n'.join([line.strip() for line in label.split('|')])
    
    bbox = {}
    if legend_style:
        bbox = {"boxstyle":"round", "facecolor":"white",
                "edgecolor":"None", "alpha":0.8}
        bbox.update(bbox_args)
    
    font_prop = std_font.copy()
    font_prop._size = fontsize
    
    txt = plt.text(x, y, string, fontproperties=font_prop,
                   bbox=bbox, **setup)

def _get_plotting_style(keys, **plotsyles):
    ps = ['cmap', 'colors', 'styles', 'markers', 'hatches', 'square']
    ps = {}.fromkeys(ps, False)
    ps.update({k:v for k,v in plotsyles.items() if (k in ps)})
    
    # List to hold styles to return
    asthetics = []
    
    # Get colors for dictionary items
    if ps['colors'] is None:
        cmap = std_cmap if (not ps['cmap']) else ps['cmap']
        fracition = np.linspace(0, 1, len(keys))
        ps['colors'] = {key: cmap(f)for key, f in zip(keys, fracition)}
    if ps['colors']: asthetics.append(ps['colors'])
    
    # Get linestyles for dictionary items
    if ps['styles'] is None:
        style_cycle = cycle(['-', '--', '-.', ':'])
        ps['styles'] = {key: next(style_cycle) for key in keys}
    
    # Apply step-fitting for dictionary item   
    if ps['square']:
        square = str(ps['square']).lower().strip()
        sqsty = {'True':'mid', 'front':'pre', 'middle':'mid', 'back':'post'}
        if square not in sqsty.values():
            square = sqsty[square]
        ps['styles'] = {key:f'steps-{square}{ls}'
                        for key,ls in ps['styles'].items()}
    if ps['styles']: asthetics.append(ps['styles'])       
    
    # Get markers for dictionary items
    if ps['markers'] is None:
        marker_cycle = cycle(['.', '+', 'x', ',', 'o', 'v', '^', '<', '>',
                              '1', '2', '3', '4', '8', 's', 'p', '*', 'h',
                              'H', 'D', 'd', '|', '_', 'TICKUP', 'TICKDOWN',
                              'TICKLEFT', 'TICKRIGHT', 'CARETUP',
                              'CARETDOWN', 'CARETLEFT', 'CARETRIGHT',
                              *[f'${c}$' for c in ascii_letters ]])
        ps['markers'] = {key: next(marker_cycle) for key in keys}
    if ps['markers']: asthetics.append(ps['markers'])
    
    # Get hatches for dictionary items
    if ps['hatches'] is None:
        hatch_cycle = cycle(['/', '\\', '|', '-', 'o',
                             'O', '.', '*', 'x', '+'])
        ps['hatches'] = {key: next(hatch_cycle) for key in keys}
    if ps['hatches']: asthetics.append(ps['hatches'])
    
    return asthetics


def _latex_label(observables):
    try:
        iterable = iter(observables)
        if not isinstance(next(iterable), str): raise TypeError
    except TypeError:
        raise ValueError("You must give a 'str' or an iterable object of " +
                         f"'str's, not '{type(observables)}'.")
    
    # Draw figure to get ticks and ticklabels
    fig = plt.gcf()
    fig.canvas.draw()
    
    input_str = False
    if isinstance(observables, str):
        input_str = True
        observables = [observables,]
    
    latex_observables = []
    for s in observables:
        ss = (s.replace('mu', '\mu').replace('del', '\Delta ')
              .replace('phi', '\phi').replace('pT', 'p_{T}'))
        if s.startswith('Z_'):
            ss = '$'+ss.replace('_','^{',1).replace('_','}_{',1)+'}$'
        elif s == 'signal':
            ss = 'Signal'
        else:
            ss = ss.replace('-', '^{-}',1).replace('+', '^{+}',1)
            ss = '$'+ss.replace('_','_{',1)+'}$'
        latex_observables.append(ss)
    
    if input_str:
        return latex_observables[0]
    return latex_observables


def _format_ticklabel_signspacing(axis, which='both', math=True, **kwargs):
    """Formats the axis ticklabels so that positive values are spaced to \
    lineup with negative values.
    
    Parameters:
     - axis: A :class:`matplotlib.axes.axis` instance holding the tick
             labels to be formatted;
     - which: A :class:`str` to dictate which spines to format, can be
              one of 'x', 'y' or 'both'. [Default 'both'];
     - math: A :class:`bool` to indicate if labels should be LaTeX math
             formatted. [Default 'True'];
     - kwargs: Any keyword arguments to pass into the
               `matplotlib.set_xticklabels` or `matplotlib.set_yticklabels`
               methods."""
    
    if which.lower() not in ['x', 'y', 'both']:
        raise ValueError("Parameter 'which' must be either 'x', 'y' or " +
                         f"'both', not '{which}'.")
    
    S = '$' if math else ''
    
    axis_ticklabels, axis_setlabels = [], []
    if which.lower() in ['x', 'both']:
        axis_ticklabels.append( axis.get_xticklabels() )
        axis_setlabels.append(lambda labels: axis.set_xticklabels(labels,
                                                                  **kwargs))
    if which.lower() in ['y', 'both']:
        axis_ticklabels.append( axis.get_yticklabels() )
        axis_setlabels.append(lambda labels: axis.set_yticklabels(labels,
                                                                 **kwargs))
    
    for ticklabels, setlabels in zip(axis_ticklabels, axis_setlabels):
        all_positive = True
        for tick in ticklabels:
            if ord(tick._text[0]) in [8722, 45]:
                all_positive = False
                break

        labels = []
        spacing = '' if all_positive else '\;\;\;'
        for tick in ticklabels:
            text = (tick._text if (ord(tick._text[0]) in [8722, 45])
                    else spacing +tick._text)
            labels.append(S+text+S)
        setlabels(labels)

        
def _format_tickmarks(axis, xmajor='bottom', ymajor='left', xminor='top',
                     yminor='right', xlabels=True, ylabels=True, **kwargs):
    """Formats the axes ticks so that major ticks are 'inout' and minor \
    ticks are 'in' and reflect major ticks.
    
    Parameters:
     - axis: A :class:`matplotlib.axes.axis` instance holding the tick
             labels to be formatted;
     - xmajor: A :class:`str` to dictate which spines the x-axis major ticks
               should appear, can be one of 'none', 'top', 'bottom' or
               'both'. [Default 'bottom'];
     - ymajor: A :class:`str` to dictate which spines the y-axis major ticks
               should appear, can be one of 'none', 'left', 'right' or
               'both'. [Default 'left'];
     - xminor: A :class:`str` to dictate which spines the x-axis minor ticks
               should appear, can be one of 'none', 'top', 'bottom' or
               'both'. [Default 'top'];
     - yminor: A :class:`str` to dictate which spines the y-axis minor ticks
               should appear, can be one of 'none', 'left', 'right' or
               'both'. [Default 'right'];
     - xlabels: A :class:`bool` to dictate if the x-axis ticklabels are
                visible. [Default 'True'];
     - ylabels: A :class:`bool` to dictate if the y-axis ticklabels are
                visible. [Default 'True'];
     - kwargs: Any keyword arguments to pass into the
               `matplotlib.axes.Axes.tick_params`, except:
               + 'bottom', 'top', 'left', 'right';
               + 'labelbottom', 'labeltop', 'labelleft', 'labelright'."""
    
    # Check no directions are given as kwargs
    directions = ['bottom', 'top', 'labelbottom', 'labeltop',
                  'left', 'right', 'labelleft', 'labelright']
    for direction in directions:
        if direction in kwargs:
            raise ValueError(f"The kwarg '{direction}' cannot be used, " +
                             "please use: 'major_xdirection', " +
                             "'major_ydirection', 'minor_xdirection' and " +
                             "'minor_ydirection' instead.")
    for excluded_kwarg in ['axis', 'which', 'direction']:
        if excluded_kwarg in kwargs:
            raise ValueError(f"The kwarg '{excluded_kwarg}' cannot " +
                             "be used.")
    
    # Check major/minor x-directions are valid entries
    xdirections = {'xmajor': xmajor.lower(), 'xminor': xminor.lower()}
    for sdirection, xdirection in xdirections.items():
        if xdirection not in ['top', 'bottom', 'both', 'none']:
            raise ValueError(f"Parameter {sdirection} must be either " +
                             "'top', 'bottom', 'both' or 'none', not " +
                             f"'{xdirection}'.")
    
    # Check major/minor y-directions are valid entries
    ydirections = {'ymajor': ymajor.lower(), 'yminor': yminor.lower()}
    for sdirection, ydirection in ydirections.items():
        if ydirection not in ['left', 'right', 'both', 'none']:
            raise ValueError(f"Parameter {sdirection} must be either " +
                             "'left', 'right', 'both' or 'none', not " +
                             f"'{ydirection}'.")
    
    # Create and update _kwargs
    _kwargs = {'length':5, 'width':1.25}
    _kwargs.update(kwargs)
    
    # Setup minor ticks to reflect majors ticks
    axis.set_xticks(axis.get_xticks(), minor=True)
    axis.set_yticks(axis.get_yticks(), minor=True)
    
    # Create dictionary for minor tick directions
    minor_directions = {direction:False for direction in directions}
    if xdirections['xminor'] in ['bottom', 'both']:
        minor_directions['bottom'] = True
    if xdirections['xminor'] in ['top', 'both']:
        minor_directions['top'] = True
    if ydirections['yminor'] in ['left', 'both']:
        minor_directions['left'] = True
    if ydirections['yminor'] in ['right', 'both']:
        minor_directions['right'] = True
    
    # Update minor tick params
    axis.tick_params(axis='both', which='minor', direction='in',
                     **minor_directions, **_kwargs)
    
    # Update _kwargs length for major ticks as they are in and out
    _kwargs['length'] *= 2.0
    
    # Create dictionary for major tick directions
    major_directions = {direction:False for direction in directions}
    if xdirections['xmajor'] in ['bottom', 'both']:
        major_directions['bottom'] = True
        major_directions['labelbottom'] = xlabels
    if xdirections['xmajor'] in ['top', 'both']:
        major_directions['top'] = True
        major_directions['labeltop'] = xlabels
    if ydirections['ymajor'] in ['left', 'both']:
        major_directions['left'] = True
        major_directions['labelleft'] = ylabels
    if ydirections['ymajor'] in ['right', 'both']:
        major_directions['right'] = True
        major_directions['labelright'] = ylabels
    
    # Update major tick params
    axis.tick_params(axis='both', which='major', direction='inout',
                     **major_directions, **_kwargs)


def get_parallel_coord_axes(n_dim):
    fig = plt.figure(figsize=(15,12))
    n_axis = n_dim - 1
    sep = 0.9 / (n_axis - 1)

    axs = [fig.add_axes((0.05, 0.05, sep, 0.9)),]
    for n in range(1, n_axis):
        axs.append(fig.add_axes((0.05 + n*sep, 0.05, sep, 0.9)))
    axs.append(axs[-1].twinx())


    for n, ax in enumerate(axs):
        ax.tick_params(axis='x',  which='both', bottom=False, top=False,
                       labelbottom=False, labeltop=False) 
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_linewidth(4)

        if n < len(axs) - 2:
            ax.spines['right'].set_visible(False)
        if n == len(axs) -1:
            ax.spines['right'].set_linewidth(4)
        
        ax.yaxis.set_tick_params(width=5)
    
    return fig, axs



# def plot_P_observable(observables, labels, labels_pred, *train_results, p_cut=0.5,
#                       n_obs = 0, xlabel="Observable"):
    
#     AMI.check_label_params(observables, labels, labels_pred, *train_results,
#                        observables=True)
    
#     if len(train_results) == 3:
#         observables = np.append(observables, train_results[0], axis=0)
#         labels = np.append(labels, train_results[1])
#         labels_pred = np.append(labels_pred, train_results[2])
    
#     is_signal = labels
#     obvs_and_signal_probs = np.hstack((observables, labels_pred))
    
#     results = {'s': obvs_and_signal_probs[is_signal is True],
#                'b': obvs_and_signal_probs[is_signal is False]}
        
#     results['s'] = results['s'][ np.argsort(results['s'][:,-1]) ]
#     results['b'] = results['b'][ np.argsort(1.0 - results['b'][:,-1]) ]
#     xmax = max(results['s'][-1,-1], results['b'][0,-1])
    
#     plt.plot((0,xmax), (p_cut,p_cut), ls='--', color='grey')
#     plt.plot(*zip(*results['s'][:,[n_obs, -1]]), color='orange', alpha=.5, label='Signal')
#     plt.plot(*zip(*results['b'][:,[n_obs, -1]]), color='blue', alpha=.5, label='Background')
    
#     add_text(0.98, 0.92, xlabel)

#     plt.xlabel(xlabel)
#     plt.ylabel("P(X = Signal/Background)") # Probability that sig/bkg is sig/bkg
#     plt.legend(loc="upper left")
#     plt.show()


# def hist_observable(data, n_bins=50,  xlabel="Observable", xlim=None, ylim=None, figsize=(16,11)):
#     ddata = {'s': [], 'b': []}
#     for datum in data:
#         if datum[1]:
#             ddata['s'].append(datum)
#         else:
#             ddata['b'].append(datum)
    
#     bins = np.linspace(min(data, key=lambda r: r[0])[0],
#                        max(data, key=lambda r: r[0])[0],
#                        n_bins + 1)
#     histogram = {'s': np.histogram(ddata['s'], bins)[0],
#                  'b': np.histogram(ddata['b'], bins)[0]}
    
#     width = bins[1] - bins[0]
#     center = (bins[:-1] + bins[1:]) / 2
    
#     histogram['s'] = AMI.hist_normalise(histogram['s'], width)
#     histogram['b'] = AMI.hist_normalise(histogram['b'], width)
    
#     plt.figure(figsize=figsize)
    
#     plt.bar(center, histogram['b'], align='center', color='blue', 
#             label='Background', width=width, alpha=.5)
#     plt.bar(center, histogram['s'], align='center', color='orange',
#             label='Signal', width=width, alpha=.5)
#     add_text(0.98, 0.74, xlabel)

#     if xlim:
#         plt.xlim(bins[0], xlim)
#     else:
#         plt.xlim(bins[0], bins[-1])
#     if ylim:
#         plt.ylim(0, ylim)
    
#     plt.xlabel(xlabel)
#     plt.ylabel("Normalised Counts per Bin")
#     plt.legend(loc="upper right")
#     plt.show()


# def get_particle_observables_order(num_each_particle, observable_order):
#     particle_order = ()
#     particle_observables_order = {}
    
#     # Cycle through the particles for each observable
#     for observable in observable_order:
#         for particle in num_each_particle:

#             # Check whether the particle mathces the observable
#             if observable.replace('$', '').startswith(particle):

#                 # Add particle to the particle ordering
#                 if not particle in particle_order:
#                     particle_order += (particle,)

#                 # Add the observable to the particle's observable ordering
#                 if not particle in particle_observables_order:
#                     particle_observables_order[particle] = (observable,)
#                 else:
#                     particle_observables_order[particle] += (observable,)

#                 # Observable's particle has been found so break particle loop
#                 break
#     return particle_order, particle_observables_order


# def organise_data_by_observables(data, num_each_particle, particle_order, particle_observables_order):
#     # Create dictionary for each particle's observables
#     observable_data = {}
#     for particle in particle_order:
#             for observable in particle_observables_order[particle]:
#                 observable_data[observable] = {'s':[], 'b':[]}
    
#     # Populate this dictionary with the values
#     for datum in data:
#         # Obtain values for datum
#         observable_values = datum[0]
#         s_or_b = 's' if datum[1] else 'b'
        
#         # Cycle through the values for each particle to add to dictionary
#         particle_start = 0
#         for particle in particle_order:
#             # Calculate number of observables for this particle
#             n_particle_obs = len(particle_observables_order[particle])
            
#             # Cycle through this particle's observables to add to dictionay
#             for i, observable in enumerate(particle_observables_order[particle]):
#                 # Calculate start and end indices for this particle's observable values in datum
#                 # this may change depending on whether there is 1 or 2 (p + anti-p) particles
#                 s_index = particle_start + i
#                 e_index = s_index + (num_each_particle[particle] - 1) * n_particle_obs + 1
                
#                 # Obtain the particle's observable's values and add to the dictionary
#                 values = observable_values[s_index : e_index : n_particle_obs]
#                 observable_data[observable][s_or_b] += values
            
#             # Determine how many indices to skip in datum for next particle's observable's values
#             particle_start += n_particle_obs * num_each_particle[particle]
#     return observable_data


# def hist_multiobservable(data, n_bins=50,  observable_order=("Observable",),
#                          num_each_particle={"Particle":1}, figsize=(16,11), subplot_kwargs=[{},]):
    
#     # Determine the order of the particles in each datum
#     # and the order of that particle's observables
#     particle_order, particle_observables_order = get_particle_observables_order(num_each_particle,
#                                                                                 observable_order)
    
#     # Get a dictionary of data for each observable
#     observable_data = organise_data_by_observables(data, num_each_particle, particle_order,
#                                                    particle_observables_order)
        
#     # Create bins for each observable
#     fig = plt.figure(figsize=figsize)
#     subplot_rows = len(particle_order)  # Determine number of rows for figure

#     # Cycle through each particle in order
#     subplot_num = 0
#     for n, particle in enumerate(particle_order):
#         particle_obsvs = particle_observables_order[particle]
#         subplot_cols = len(particle_obsvs)  # Determine number of columns for this row
        
#         # Cycle through each observable of this particle
#         for m, observable in enumerate(particle_obsvs):

#             # Obtain the signal & bkground values for this observable
#             observable_values = observable_data[observable]
#             all_values = observable_values['s'] + observable_values['b']

#             # Create bins and histograms for this observable
#             sp_n_bins = subplot_kwargs[subplot_num].get('n_bins', False)
#             sp_n_bins = sp_n_bins if sp_n_bins else n_bins
#             bins = np.linspace(min(all_values), max(all_values), sp_n_bins + 1)
#             histogram = {'s': np.histogram(observable_values['s'], bins)[0],
#                          'b': np.histogram(observable_values['b'], bins)[0]}

#             # Determine the bins' widths and centers
#             width = bins[1] - bins[0]
#             center = (bins[:-1] + bins[1:]) / 2

#             # Create subplot to hold this observable's histogram
#             subplot_index = (n * subplot_cols) + (m + 1)  # Determine postion in figure
#             subplot = fig.add_subplot(subplot_rows, subplot_cols, subplot_index)
            
#             # Normalise the histograms
#             histogram['s'] = AMI.hist_normalise(histogram['s'], width)
#             histogram['b'] = AMI.hist_normalise(histogram['b'], width)
            
#             # Plot the histograms
#             subplot.bar(center, histogram['b'], align='center', color='blue', 
#                         label='Background', width=width, alpha=.5)
#             subplot.bar(center, histogram['s'], align='center', color='orange',
#                         label='Signal', width=width, alpha=.5)
            
#             # Label the subplots
#             units_label = subplot_kwargs[subplot_num].get('units', False)
#             units_label = (' (' + units_label + ')') if units_label else ''
#             subplot.set_xlabel(observable + units_label)
#             subplot.xaxis.set_label_position('top') 
#             if subplot.is_first_col():
#                 subplot.set_ylabel("Normalised Counts per Bin")
            
#             # Add subplot limits
#             xlim = subplot_kwargs[subplot_num].get('xlim')
#             if isinstance(xlim, (int, float, tuple, list)):
#                 if isinstance(xlim, (tuple, list)):
#                     if len(xlim) == 1:
#                         subplot.set_xlim(bins[0], xlim[0])
#                     else:
#                         subplot.set_xlim(xlim[0], xlim[1])
#                 else:
#                     subplot.set_xlim(bins[0], xlim)
            
#             ylim = subplot_kwargs[subplot_num].get('ylim')
#             if isinstance(ylim, (int, float, tuple, list)):
#                 if isinstance(ylim, (tuple, list)):
#                     if len(ylim) == 1:
#                         subplot.set_ylim(0.0, ylim[0])
#                     else:
#                         subplot.set_ylim(ylim[0], ylim[1])
#                 else:
#                     subplot.set_ylim(0.0, ylim)
            
#             # Increment subplot counter by one
#             subplot_num += 1

#     # Show plots
#     fig.show()


def lighten_color(color, frac=0.5):
    """Returns the RGB values for a lighter version of the given color.
    
    Parameters:
     - color: A :class:`str` or RGB(A) :class:`tuple` which matplotlib can
              interpret;
     - frac: A :class:`float` between [-1,1] giving the relative change in
             brightness. [Default '0.5'].
             
    Returns:
     - RGBtuple: A RGB :class:`tuple`."""
    
    # Determine functions to use for increasing/decreasing brightness
    lum = ((lambda l: l + frac*(1 - l)) if (frac > 0)
           else (lambda l: l + frac*l))
    
    # Convert str to color
    try: color = mcolors.cnames[color]
    except: pass
    
    # Convert RGB color to HLS, then increase/decrease
    # luminosity and convert back
    color = colorsys.rgb_to_hls(*mcolors.to_rgb(color))
    return colorsys.hls_to_rgb(color[0], lum(color[1]), color[2])

def darken_color(color, frac=0.5):
    """Returns the RGB values for a darker version of the given color.
    
    Parameters:
     - color: A :class:`str` or RGB(A) :class:`tuple` which matplotlib can
              interpret;
     - frac: A :class:`float` between [-1,1] giving the relative change in
             brightness. [Default '0.5'].
             
    Returns:
     - RGBtuple: A RGB :class:`tuple`."""
    return lighten_color(color, frac=-frac)