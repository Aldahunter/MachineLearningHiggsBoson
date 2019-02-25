"""Plotting - Module for Analysis Plots of Machine Learning Classifier's and their outputs."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import dwrangling as DW
import analysis.metrics as AME
import analysis.misc as AMI



### Data ###
comma_space = ', '
c = mcolors.ColorConverter().to_rgb
cmap_input = [c('navy'), c('blue'), 0.20, c('blue'), c('royalblue'), 0.40,
              c('royalblue'), c('white'), 0.50, c('white'), c('orange'), 0.60,
              c('orange'), c('darkorange'), 0.80, c('darkorange'), c('orangered')]


### Functions ###
def get_image_file_paths(name, ext, *directories):
    f"""Returns the file location as a string, for both signal and background.
    
    Parameters:
     - name: The name of the file as type 'str';
     - ext: The file extension/type as type 'str', e.g. 'jpg';
     - directories: The directories for the file as 'str's in the correct order."""

    # Validate and clean inputs
    name, ext = name.strip(), ext.strip()
    directories = [dir_.strip() for dir_ in directories]
    ext = ext if (ext[0] == '.') else '.' + ext

    # Obtain the file's path and return
    file_loc = os.path.join(os.getcwd(), "Figures", *directories, name + ext)
    return file_loc


def add_text(x, y, label, axes=plt.gca(), **kwargs):
    string = ',\n'.join([line.strip()
                         for line in label.split('|')])
    plt.text(x, y, string,
             horizontalalignment='right',
             verticalalignment='center',
             weight='semibold',
             transform=axes.transAxes,
             **kwargs)


def latex_label(observables):
    latex_observables = []
    for s in observables:
        ss = s.replace('mu', '\mu').replace('pT', 'p_{T}')
        if s.startswith('Z_'):
            ss = '$'+ss.replace('_','^{',1).replace('_','}_{',1)+'}$'
        elif s == 'signal':
            ss = 'Signal'
        else:
            ss = ss.replace('-', '^{-}',1).replace('+', '^{+}',1)
            ss = '$'+ss.replace('_','_{',1)+'}$'
        latex_observables.append(ss)
    return latex_observables


def make_colormap(name, seq):
    """Returns a LinearSegmentedColormap.
    
    Parameters:
     - name: A string for the name of the color map.
     - seq: A sequence of 'float's and RGB 'tuple's. The floats should be 
            increasing and in the interval (0,1), as they give the
            proportions between the RGB tuples. """
    
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
            
    return mcolors.LinearSegmentedColormap(name, cdict)



def plot_results(classifier, labels, labels_pred, *train_results, n_bins=50,
                 text="Observable", p_threshold = None, show_cm=True):
    AMI.check_label_params(labels, labels_pred, *train_results)
    
    # Add test and train results to dictionary for iterating
    full_results = {'Test': (labels, labels_pred)}
    if train_results:
        full_results['Train'] = train_results
        ax_dimensions = [0,0.25,0.8,0.75]
    else:
        ax_dimensions = [0.05,0.05,0.9,0.9]
    
    # Set-up figure and main axis
    fig = plt.figure(figsize=(11.00,9.77)) #(8.0, 7.33)
    ax = fig.add_axes(ax_dimensions)
    
    # Iterate through train and test data
    histograms = {}
    markersize = 18
    for key, type_results in full_results.items():
        
        # Seperate signal and background
        is_signal = type_results[0]
        signal_probabilities = type_results[1]
        results = {'s': signal_probabilities[is_signal == True],
                   'b': signal_probabilities[is_signal == False]}

        # Create bins for histograms
        bins = np.linspace(0, 1, n_bins + 1)
        histogram = {'s': np.histogram(results['s'], bins)[0],
                     'b': np.histogram(results['b'], bins)[0]}
        
        # Find bin widths and centers
        width = bins[1] - bins[0]
        center = (bins[:-1] + bins[1:]) / 2
        
        # Normalise bins
        histogram['s'] = AMI.hist_normalise(histogram['s'], width)
        histogram['b'] = AMI.hist_normalise(histogram['b'], width)
        
        histograms[key] = histogram

        # Plot histograms
        if key == 'Test':
            ax.bar(center, histogram['b'], align='center', color='blue', 
                    label='Background', width=width, alpha=.5, zorder=1)
            ax.bar(center, histogram['s'], align='center', color='orange',
                    label='Signal', width=width, alpha=.5, zorder=1)
        elif key == 'Train':
            ax.plot(center, histogram['b'], marker='x', ls='', color='blue',
                    markersize=markersize, label='Background [Train]', alpha=1.0, zorder=2)
            ax.plot(center, histogram['s'], marker='x', ls='', color='orange',
                    markersize=markersize, label='Signal [Train]', alpha=1.0, zorder=2)
    
    if train_results:
        # Plot residual deviations between test and train data
        ax_res = fig.add_axes([0,0,0.8,0.25], sharex=ax)
        c = {'s':'orange','b':'blue'}
        deviations = {}
        for sb in ['b', 's']:
            deviation = histograms['Train'][sb] - histograms['Test'][sb]
            deviations[sb] = deviation
            ax_res.bar(center, deviation, align='center', color=c[sb],
                       width=width, alpha=.5)
        ax_res.axhline(color='grey', linestyle='--', zorder=0)

        # Plot distribution of deviations
        ax_dist = fig.add_axes([0.8,0,0.2,0.25], sharey=ax_res)
        ymin, ymax = ax_res.get_ylim()
        for sb in ['b', 's']:
            mu, std = norm.fit(deviations[sb])

            y = np.linspace(ymin, ymax, 100)
            p = norm.pdf(y, mu, std)

            ax_dist.plot(p, y, ls='-', color=c[sb], alpha=0.6, linewidth=3, zorder=1)
        ax_dist.axhline(color='grey', linestyle='--', zorder=0)
    
    if p_threshold is not None:
        if not (0 < p_threshold < 1):
            raise ValueError("'p_threshold' must be between 0 and 1, " +
                             f"not {p_threshold}.")
        ax.axvline(p_threshold, linestyle='--', color='black')
        
        if train_results:
            ax_res.axvline(p_threshold, linestyle='--', color='black')
    
    
    # Add observable labels as text
    fontsize = 'x-large'                    #  {'x-small', 'medium', 'xx-large'}
    add_text(0.39, 0.95, 'Observables:\n'+text.replace(' &',','), fontsize=fontsize, axes=ax) #35
    
    if show_cm:
        p_threshold = 0.5 if (p_threshold is None) else p_threshold
        tn, fp, fn, tp = AME.confusion_matrix(labels, labels_pred, p_threshold=p_threshold)
        str_cm = f"TP: {tp:5d}; FP: {fp:5d};\nFN: {fn:5d}; TN: {tn:5d};"
        add_text(0.39, 0.88, 'Confusion Matrix:\n'+str_cm, fontsize='medium', axes=ax)

    
    # Add x labels
    s_classifier = str(type(classifier))[8:-2].split('.')[-1]
    ax.set_xlim(0.0, 1.0)
    if train_results:
        ax_res.set_xlabel(f"Classifier Output [{s_classifier}]", fontsize=fontsize)
        ax_dist.set_xlabel("Normalised Deviation", fontsize=fontsize)
    else:
        ax.set_xlabel(f"Classifier Output [{s_classifier}]", fontsize=fontsize)
    
    # Add y labels
    ax.set_ylabel("Normalised Counts per Bin", fontsize=fontsize)
    if train_results:
        ax_res.set_ylabel("Deviation from Training", fontsize=fontsize)
        ax_dist.set_ylim(ymin, ymax)
        ax_dist.yaxis.tick_right()
    
    # Set tick parameters
    ax.tick_params(axis='y', bottom=True, top=True, left=True, right=True,
                   labelsize=fontsize, direction='inout')
    ax.tick_params(axis='x', bottom=True, top=True, left=True, right=True,
                   direction='in', labelbottom=False)
    if train_results:
        ax_res.tick_params(labelsize=fontsize, bottom=True, top=True, left=True,
                           right=True, direction='inout')
        ax_dist.tick_params(labelsize=fontsize, bottom=True, top=True, left=True,
                            right=True, direction='inout')
    
    # Add legend
    ax.legend(loc="upper left", fontsize=fontsize)
    
    # Show figure
    fig.show()


def ROC_curve(labels, labels_pred, *train_results, n_cuts=100,
              text="Observables"):
    """The parameters must be in the order:
    - labels: numpy.array of length n, where a label is: \
    signal (True) or background (False); 
    - labels_pred: numpy.array of length n, where a label_pres \
    is: the probability output of the algortihm range [0, 1.0];
    - train_labels [optional]: numpy.array of length n, where a \
    label is: signal (True) or background (False); 
    - train_labels_pred [optional]: numpy.array of length n, \
    where a label_pres is: the probability output of the algortihm \
    range [0, 1.0]; 

kwargs:
    - n_cuts is the number of p_threshold cuts used (this \
increases resolution of graph)
    - text: the label on the graph (top right corner)"""
    AMI.check_label_params(labels, labels_pred, *train_results)
    
    full_results = {'Test': (labels, labels_pred)}
    if not train_results:
        full_results['Train'] = train_results
    
    fig = plt.figure(figsize=(10.67,7.33))
    ax = fig.add_axes([0,0,1,1])
    
    plots = []
    for key, type_results in full_results.items():
        
        ls = '-' if key == 'Test' else '--'
        zorder = 1 if key == 'Test' else 0
    
        scatter = [(0,0),]
        for p_cut in range(n_cuts+1):
            cm = AME.confusion_matrix(*type_results, p_cut / 100.0)
            x, y = AME.fpr(*cm), AME.recall(*cm)
            scatter.append((x, y))
        
        scatter = sorted(scatter, key=lambda xy: abs(xy[0]))
        scatter.append((1,1))

        ax.plot(*zip(*scatter), color='red', ls=ls, zorder=zorder,
                label = key)
        
        
        auc = metrics.roc_auc_score(*type_results)
        print(f"{key} AUC = {auc}")
        
        fill = ax.fill_between(*zip(*scatter), alpha=.25, zorder=zorder, color='red',
                               label=f"{key}: {auc:.3f}")
        if key == 'Train':
            fill.set_hatch(r'//')
        else:
            fill.set_hatch(r'\\')
    
    ax.plot((0,1),(0,1), ls='--', color='grey', zorder=3)
    
    fontsize = 'x-large'                    #  {'x-small', 'medium', 'xx-large'}
    ax.text(0.58, 0.008, 'Paramters:\n'+text.replace(' &',','), fontsize=fontsize)
    
    ax.set_ylabel("True Positive Rate [Recall]", fontsize=fontsize)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False Positive Rate", fontsize=fontsize)
    ax.set_xlim(-0.01, 1.01)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.set_title("ROC Curve", fontsize=fontsize)
    frame = ax.legend(*list(zip(*plots)),loc="lower right", fontsize=fontsize,
                      framealpha=1.0)
    fig.show()


def ROC_displacement(labels, labels_pred, *train_results, n_cuts=100, return_max=False,
                    text="Observable"):
    AMI.check_label_params(labels, labels_pred, *train_results)
    
    full_results = {'Test': (labels, labels_pred)}
    if not train_results:
        full_results['Train'] = train_results
    
    fig = plt.figure(figsize=(10.67,7.33))
    ax = fig.add_axes([0,0,1,1])
    
    for key, type_results in full_results.items():
        
        ls = '-' if key == 'Test' else '--'
        zorder = 2 if key == 'Test' else 1
    
        displacements = [(0,0),]
        for p_cut in range(n_cuts+1):
            cm = AME.confusion_matrix(*type_results, p_cut / 100.0)
            x, y = AME.fpr(*cm), AME.recall(*cm)
            disp = np.sqrt(0.5*(x**2 + y**2) - x*y)
            disp *= 100 if (y >= x) else -100
            displacements.append((p_cut/100, disp))

        displacements = sorted(displacements, key=lambda xy: abs(xy[0]))
        displacements.append((1,0))
        if key == 'Test':
            max_disp = max(displacements, key=lambda xy: abs(xy[1]))

        ax.plot(*zip(*displacements), ls=ls, color='red', label=key, zorder=zorder)
    
    ax.plot((0, 1), (0,0), ls='--', color='grey', zorder=0)
    
    fontsize = 'x-large'                    #  {'x-small', 'medium', 'xx-large'}
    ax.text(0.64, 0.85, 'Paramters:\n'+text.replace(' &',','), fontsize=fontsize)  # 0.55, 0.85

    ax.set_title("ROC Disp", fontsize=fontsize)
    ax.set_ylabel("Guess Displacement (%)", fontsize=fontsize)
    ax.set_ylim(0.0)
    ax.set_xlabel("Probability Threshold [$P_{cut}$]", fontsize=fontsize)
    ax.set_xlim(-0.01, 1.01)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.legend(loc="lower right", fontsize=fontsize)
    fig.show()
    
    if return_max == True:
        return max_disp


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


### Functional Data ###
std_cmap = make_colormap('RGB', cmap_input)