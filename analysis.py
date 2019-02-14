### Imports ###
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from random import random, seed
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score


### Functions ###
## Data ##
def split_data(data, train_frac = 0.67):
    # data must be a pd.DataFrame
    train_data = []
    test_data = []

    for _, datum in data.iterrows():
        if random() <= train_frac:
            train_data.append(datum)
        else:
            test_data.append(datum)
    
    return {'train': pd.DataFrame(train_data).reset_index(drop=True),
            'test': pd.DataFrame(test_data).reset_index(drop=True)}


def test_classifier(test_data, classifier):
    results = []
    for datum in test_data:  # datum = (observable, label)
        p = classifier.classify(datum)
        results.append((*datum, p))
    return results


## Machine Learning Analysis ##
def accuracy(tp, fp, fn, tn):
    return (tp + tn) / (tp + fp + fn + tn)


def precision(tp, fp, fn, tn):
    return tp / (tp + fp)


def recall(tp, fp, fn, tn):
    return tp / (tp + fn)


def fpr(tp, fp, fn, tn):
    """False Positive Rate"""
    return fp / (fp + tn)


def f1_score(tp, fp, fn, tn):
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)
    return (2 * p * r) / (p + r)


def confusion_matrix(labelled_predictions):
    """labelled_predictions = [(label, predict),...]"""
    label_pred_counts = Counter(labelled_predictions)
        
    true_positive = label_pred_counts[(True, True)]
    false_positive = label_pred_counts[(False, True)]
    false_negative = label_pred_counts[(True, False)]
    true_negative = label_pred_counts[(False, False)]

    return (true_positive, false_positive,
            false_negative, true_negative)


def get_confusion_matrix(test_results, p_cut=0.5):
    labelled_predictions = [(label, p >= p_cut)
                            for value, label, p in test_results]
    return confusion_matrix(labelled_predictions)


def S_B_ratio(p_cut, test_results):
    sigma_s = 0.204436505
    sigma_b = 12.25108192
    luminosity = 100
    
    cm = get_confusion_matrix(test_results, p_cut)
    epsilon_b, epsilon_s = fpr(*cm), recall(*cm)
    
    try:
        sb_ratio = ( (sigma_s * np.sqrt(luminosity) * epsilon_s) 
                     / np.sqrt(sigma_b * epsilon_b) )
    except ZeroDivisionError:
        sb_ratio = 0.0
    
    return sb_ratio


def find_p_cut(test_results):
    
    def minimise_fn(p_cut):
        try:
            score = 1.0 / S_B_ratio(p_cut, test_results)
        except ZeroDivisionError:
            score = float('inf')
        return score

    optimize_result = minimize_scalar(minimise_fn, method='Bounded', bounds=(0.0, 1.0))
    
    return optimize_result.x


def get_scores(test_results, p_cut=0.5):
    scores = {}
    tp, fp, fn, tn = get_confusion_matrix(test_results, p_cut)

    scores["a"] = accuracy(tp, fp, fn, tn)
    scores["p"] = precision(tp, fp, fn, tn)
    scores["r"] = recall(tp, fp, fn, tn)
    scores["f1"] = f1_score(tp, fp, fn, tn)

    return scores


## Analysis Plotting ##
def latex_label(observables):
    latex_observables = []
    for s in observables:
        ss = s.replace('mu', '\mu').replace('pT', 'p_{T}')
        if s.startswith('Z_'):
            ss = '$'+ss.replace('_','^{',1).replace('_','}_{',1)+'}$'
        elif s == 'signal':
            continue
        else:
            ss = ss.replace('-', '^{-}',1).replace('+', '^{+}',1)
            ss = '$'+ss.replace('_','_{',1)+'}$'
        latex_observables.append(ss)
    return ' & '.join(latex_observables)


def hist_normalise(histogram, bin_width):
        total_area = 0.0
        for hist_bin in histogram:
            total_area += hist_bin * bin_width
        return histogram / total_area

def add_text(x, y, label):
    string = ',\n'.join([line.strip()
                         for line in label.split('|')])
    plt.text(x, y, string,
             horizontalalignment='right',
             verticalalignment='center',
             weight='semibold',
             transform=plt.gca().transAxes)

def ROC_curve(test_results, n_cuts=100, text="Observable", train_results=None):
    """test_results = [(value, label, p), (value, label, p), \
...] where:
    - value: the observable inputs to the algorithm;
    - label: signal (True) or background (False);
    - p: probability output of algortihm in range [0, 1.0];

kwargs:
    - n_cuts is the number of p_threshold cuts used (this \
increases resolution of graph)
    - text: the label on the graph (top right corner)"""
    
    full_results = {'Test':test_results, 'Train':train_results}
    
    fig = plt.figure(figsize=(10.67,7.33))
    ax = fig.add_axes([0,0,1,1])
    
    plots = []
    for key, type_results in full_results.items():
        if type_results == None: continue
        
        ls = '-' if key == 'Test' else '--'
        zorder = 1 if key == 'Test' else 0
    
        scatter = [(0,0),]
        for p_cut in range(n_cuts+1):
            cm = get_confusion_matrix(type_results, p_cut / 100)
            x, y = fpr(*cm), recall(*cm)
            scatter.append((x, y))
        scatter = sorted(scatter, key=lambda xy: abs(xy[0]))
        scatter.append((1,1))

        ax.plot(*zip(*scatter), color='red', ls=ls, zorder=zorder,
                label = key)
        
        # Remove the values so only have [list(is_signal,), list(p_signal)]
        type_results = list(zip(*type_results))[1:]
        
        auc = roc_auc_score(*type_results)
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
#     fig.show()

def ROC_displacement(test_results, n_cuts=100, return_max=False,
                    text="Observable", train_results=None):
    
    full_results = {'Test':test_results, 'Train':train_results}
    
    fig = plt.figure(figsize=(10.67,7.33))
    ax = fig.add_axes([0,0,1,1])
    
    for key, type_results in full_results.items():
        if type_results == None: continue
        
        ls = '-' if key == 'Test' else '--'
        zorder = 2 if key == 'Test' else 1
    
        displacements = [(0,0),]
        for p_cut in range(n_cuts+1):
            cm = get_confusion_matrix(type_results, p_cut /100)
            x, y = fpr(*cm),recall(*cm)
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
#     fig.show()
    
    if return_max == True: return max_disp

def plot_results(test_results, classifier, n_bins=50,
                 text="Observable", train_results=None):
    full_results = {'test':test_results, 'train':train_results}
    
    fig = plt.figure(figsize=(10.67,9.77))
    ax = fig.add_axes([0,0.25,0.8,0.75])
    
    histograms = {}
    markersize = 18
    for key, type_results in full_results.items():
        if type_results == None: continue
        
        results = {'s': [], 'b': []}

        for result in type_results:
            if result[1]:
                results['s'].append(result[-1])
            else:
                results['b'].append(result[-1])

        bins = np.linspace(0, 1, n_bins + 1)
        histogram = {'s': np.histogram(results['s'], bins)[0],
                     'b': np.histogram(results['b'], bins)[0]}
        
        width = bins[1] - bins[0]
        center = (bins[:-1] + bins[1:]) / 2

        histogram['s'] = hist_normalise(histogram['s'], width)
        histogram['b'] = hist_normalise(histogram['b'], width)
        
        histograms[key] = histogram

        if key == 'train':
            ax.bar(center, histogram['b'], align='center', color='blue', 
                    label='Background [Train]', width=width, alpha=.5, zorder=1)
            ax.bar(center, histogram['s'], align='center', color='orange',
                    label='Signal [Train]', width=width, alpha=.5, zorder=1)
        elif key == 'test':
            ax.plot(center, histogram['b'], marker='x', ls='', color='blue',
                    markersize=markersize, label='Background', alpha=1.0, zorder=2)
            ax.plot(center, histogram['s'], marker='x', ls='', color='orange',
                    markersize=markersize, label='Signal', alpha=1.0, zorder=2)
    
    ax_res = fig.add_axes([0,0,0.8,0.25], sharex=ax)
    c = {'s':'orange','b':'blue'}
    deviations = {}
    for sb in ['b', 's']:
        deviation = histograms['train'][sb] - histograms['test'][sb]
        deviations[sb] = deviation
        ax_res.bar(center, deviation, align='center', color=c[sb],
                   width=width, alpha=.5)
    ax_res.axhline(color='grey', linestyle='--', zorder=0)
    
    ax_dist = fig.add_axes([0.8,0,0.2,0.25], sharey=ax_res)
    ymin, ymax = ax_res.get_ylim()
    for sb in ['b', 's']:
        mu, std = norm.fit(deviations[sb])
        
        y = np.linspace(ymin, ymax, 100)
        p = norm.pdf(y, mu, std)
        
        ax_dist.plot(p, y, ls='-', color=c[sb], alpha=0.6, linewidth=3, zorder=1)
    ax_dist.axhline(color='grey', linestyle='--', zorder=0)
    
    
    fontsize = 'x-large'                    #  {'x-small', 'medium', 'xx-large'}
    ax.text(0.33, 35, 'Paramters:\n'+text.replace(' &',','), fontsize=fontsize)

    s_classifier = str(type(classifier))[8:-2].split('.')[-1]
    ax_res.set_xlim(0.0, 1.0)
    ax_res.set_xlabel(f"Classifier Output [{s_classifier}]", fontsize=fontsize)
    ax_dist.set_xlabel("Normalised Deviation", fontsize=fontsize)
    
    ax.set_ylabel("Normalised Counts per Bin", fontsize=fontsize)
    ax_res.set_ylabel("Deviation from Training", fontsize=fontsize)
    ax_dist.set_ylim(ymin, ymax)
    ax_dist.yaxis.tick_right()
    
    
    ax.tick_params(axis='y', bottom=True, top=True, left=True, right=True,
                   labelsize=fontsize, direction='inout')
    ax.tick_params(axis='x', bottom=True, top=True, left=True, right=True,
                   direction='in', labelbottom=False)
    ax_res.tick_params(labelsize=fontsize, bottom=True, top=True, left=True,
                       right=True, direction='inout')
    ax_dist.tick_params(labelsize=fontsize, bottom=True, top=True, left=True,
                        right=True, direction='inout')
    ax.legend(loc="upper left", fontsize=fontsize)
#     fig.show()

def rms_deviation(test_results, train_results, n_bins=50):
    full_results = {'test':test_results, 'train':train_results}
    
    histograms = {}
    for key, type_results in full_results.items():
        results = {'s': [], 'b': []}

        for result in type_results:
            if result[1]:
                results['s'].append(result[-1])
            else:
                results['b'].append(result[-1])

        bins = np.linspace(0, 1, n_bins + 1)
        histogram = {'s': np.histogram(results['s'], bins)[0],
                     'b': np.histogram(results['b'], bins)[0]}
        
        width = bins[1] - bins[0]
        histogram['s'] = hist_normalise(histogram['s'], width)
        histogram['b'] = hist_normalise(histogram['b'], width)
        
        histograms[key] = histogram
        
    rmsq = {}
    for sb in ['b', 's']:
        deviation = (histograms['train'][sb] - histograms['test'][sb])**2
        rmsq[sb] = np.sqrt( deviation.sum() / len(deviation) )
    
    return rmsq


def plot_P_observable(test_results, p_cut=0.5, xlabel="Observable"):
    results = {'s': [], 'b': []}
    
    for result in test_results:
        if result[1]:
            results['s'].append((result[0], result[2]))
        else:
            results['b'].append((result[0], result[2]))
        
    results['s'] = sorted(results['s'], key=lambda result: result[0])
    results['b'] = sorted(results['b'], key=lambda result: 1.0 - result[0])
    xmax = max(results['s'][-1][0], results['b'][0][0])
    
    plt.plot((0,xmax), (p_cut,p_cut), ls='--', color='grey')
    plt.plot(*zip(*results['s']), color='orange', alpha=.5, label='Signal')
    plt.plot(*zip(*results['b']), color='blue', alpha=.5, label='Background')
    
    add_text(0.98, 0.92, xlabel)

    plt.xlabel(xlabel)
    plt.ylabel("P(X = Signal/Background)") # Probability that sig/bkg is sig/bkg
    plt.legend(loc="upper left")
    plt.show()

def hist_observable(data, n_bins=50,  xlabel="Observable", xlim=None, ylim=None, figsize=(16,11)):
    ddata = {'s': [], 'b': []}
    for datum in data:
        if datum[1]:
            ddata['s'].append(datum)
        else:
            ddata['b'].append(datum)
    
    bins = np.linspace(min(data, key=lambda r: r[0])[0],
                       max(data, key=lambda r: r[0])[0],
                       n_bins + 1)
    histogram = {'s': np.histogram(ddata['s'], bins)[0],
                 'b': np.histogram(ddata['b'], bins)[0]}
    
    width = bins[1] - bins[0]
    center = (bins[:-1] + bins[1:]) / 2
    
    histogram['s'] = hist_normalise(histogram['s'], width)
    histogram['b'] = hist_normalise(histogram['b'], width)
    
    plt.figure(figsize=figsize)
    
    plt.bar(center, histogram['b'], align='center', color='blue', 
            label='Background', width=width, alpha=.5)
    plt.bar(center, histogram['s'], align='center', color='orange',
            label='Signal', width=width, alpha=.5)
    add_text(0.98, 0.74, xlabel)

    if xlim:
        plt.xlim(bins[0], xlim)
    else:
        plt.xlim(bins[0], bins[-1])
    if ylim:
        plt.ylim(0, ylim)
    
    plt.xlabel(xlabel)
    plt.ylabel("Normalised Counts per Bin")
    plt.legend(loc="upper right")
    plt.show()

def get_particle_observables_order(num_each_particle, observable_order):
    particle_order = ()
    particle_observables_order = {}
    
    # Cycle through the particles for each observable
    for observable in observable_order:
        for particle in num_each_particle:

            # Check whether the particle mathces the observable
            if observable.replace('$', '').startswith(particle):

                # Add particle to the particle ordering
                if not particle in particle_order:
                    particle_order += (particle,)

                # Add the observable to the particle's observable ordering
                if not particle in particle_observables_order:
                    particle_observables_order[particle] = (observable,)
                else:
                    particle_observables_order[particle] += (observable,)

                # Observable's particle has been found so break particle loop
                break
    return particle_order, particle_observables_order

def organise_data_by_observables(data, num_each_particle, particle_order, particle_observables_order):
    # Create dictionary for each particle's observables
    observable_data = {}
    for particle in particle_order:
            for observable in particle_observables_order[particle]:
                observable_data[observable] = {'s':[], 'b':[]}
    
    # Populate this dictionary with the values
    for datum in data:
        # Obtain values for datum
        observable_values = datum[0]
        s_or_b = 's' if datum[1] else 'b'
        
        # Cycle through the values for each particle to add to dictionary
        particle_start = 0
        for particle in particle_order:
            # Calculate number of observables for this particle
            n_particle_obs = len(particle_observables_order[particle])
            
            # Cycle through this particle's observables to add to dictionay
            for i, observable in enumerate(particle_observables_order[particle]):
                # Calculate start and end indices for this particle's observable values in datum
                # this may change depending on whether there is 1 or 2 (p + anti-p) particles
                s_index = particle_start + i
                e_index = s_index + (num_each_particle[particle] - 1) * n_particle_obs + 1
                
                # Obtain the particle's observable's values and add to the dictionary
                values = observable_values[s_index : e_index : n_particle_obs]
                observable_data[observable][s_or_b] += values
            
            # Determine how many indices to skip in datum for next particle's observable's values
            particle_start += n_particle_obs * num_each_particle[particle]
    return observable_data

def hist_multiobservable(data, n_bins=50,  observable_order=("Observable",),
                         num_each_particle={"Particle":1}, figsize=(16,11), subplot_kwargs=[{},]):
    
    # Determine the order of the particles in each datum
    # and the order of that particle's observables
    particle_order, particle_observables_order = get_particle_observables_order(num_each_particle,
                                                                                observable_order)
    
    # Get a dictionary of data for each observable
    observable_data = organise_data_by_observables(data, num_each_particle, particle_order,
                                                   particle_observables_order)
        
    # Create bins for each observable
    fig = plt.figure(figsize=figsize)
    subplot_rows = len(particle_order)  # Determine number of rows for figure

    # Cycle through each particle in order
    subplot_num = 0
    for n, particle in enumerate(particle_order):
        particle_obsvs = particle_observables_order[particle]
        subplot_cols = len(particle_obsvs)  # Determine number of columns for this row
        
        # Cycle through each observable of this particle
        for m, observable in enumerate(particle_obsvs):

            # Obtain the signal & bkground values for this observable
            observable_values = observable_data[observable]
            all_values = observable_values['s'] + observable_values['b']

            # Create bins and histograms for this observable
            sp_n_bins = subplot_kwargs[subplot_num].get('n_bins', False)
            sp_n_bins = sp_n_bins if sp_n_bins else n_bins
            bins = np.linspace(min(all_values), max(all_values), sp_n_bins + 1)
            histogram = {'s': np.histogram(observable_values['s'], bins)[0],
                         'b': np.histogram(observable_values['b'], bins)[0]}

            # Determine the bins' widths and centers
            width = bins[1] - bins[0]
            center = (bins[:-1] + bins[1:]) / 2

            # Create subplot to hold this observable's histogram
            subplot_index = (n * subplot_cols) + (m + 1)  # Determine postion in figure
            subplot = fig.add_subplot(subplot_rows, subplot_cols, subplot_index)
            
            # Normalise the histograms
            histogram['s'] = hist_normalise(histogram['s'], width)
            histogram['b'] = hist_normalise(histogram['b'], width)
            
            # Plot the histograms
            subplot.bar(center, histogram['b'], align='center', color='blue', 
                        label='Background', width=width, alpha=.5)
            subplot.bar(center, histogram['s'], align='center', color='orange',
                        label='Signal', width=width, alpha=.5)
            
            # Label the subplots
            units_label = subplot_kwargs[subplot_num].get('units', False)
            units_label = (' (' + units_label + ')') if units_label else ''
            subplot.set_xlabel(observable + units_label)
            subplot.xaxis.set_label_position('top') 
            if subplot.is_first_col():
                subplot.set_ylabel("Normalised Counts per Bin")
            
            # Add subplot limits
            xlim = subplot_kwargs[subplot_num].get('xlim')
            if isinstance(xlim, (int, float, tuple, list)):
                if isinstance(xlim, (tuple, list)):
                    if len(xlim) == 1:
                        subplot.set_xlim(bins[0], xlim[0])
                    else:
                        subplot.set_xlim(xlim[0], xlim[1])
                else:
                    subplot.set_xlim(bins[0], xlim)
            
            ylim = subplot_kwargs[subplot_num].get('ylim')
            if isinstance(ylim, (int, float, tuple, list)):
                if isinstance(ylim, (tuple, list)):
                    if len(ylim) == 1:
                        subplot.set_ylim(0.0, ylim[0])
                    else:
                        subplot.set_ylim(ylim[0], ylim[1])
                else:
                    subplot.set_ylim(0.0, ylim)
            
            # Increment subplot counter by one
            subplot_num += 1

    # Show plots
    fig.show()

    
## Machine Learning Analysis ##
def df_to_ML_input(df):
    ML_input = []
    
    for index, row in df.loc[:, df.columns != 'signal'].iterrows():
        ML_input.append( (tuple(row), df.signal[index]) )
    
    return tuple(ML_input)


def SN_classifier(classifier, data_df, classifier_kwargs, rseed=None):
    """Trains and tests the classifer with the events data_df panda.DataFrame \
    (containing only the desired observables and signal columns ) using the \
    given classifier_kwargs."""
    if rseed is not None:
        seed(rseed)
    
    data_df = split_data(data_df, train_frac=0.67)
    
    input_data = {}
    input_data['train'] = df_to_ML_input(data_df['train'])
    input_data['test'] = df_to_ML_input(data_df['test'])
    
#     print("Training Classifier")
    classifier.train(input_data['train'], **classifier_kwargs)
    
#     print("Testing Classifier")
    test_results = test_classifier(input_data['test'], classifier)
    train_results = test_classifier(input_data['train'], classifier)

    rmsq = rms_deviation(test_results, train_results, n_bins=50)
    p_cut = find_p_cut(test_results)
    sn_ratio = S_B_ratio(p_cut, test_results)
#     print(f'S/N : {sn_ratio}')
    
    return classifier, sn_ratio, rmsq


def analyse_classifier(classifier, sgn_df, bkg_df, input_observables, classifier_kwargs):
    """Analyses the given classifer with the panda.DataFrame events, using the 
    input_observables and classifier_kwargs."""
    
    sgn_data = sgn_df[input_observables]
    bkg_data = bkg_df[input_observables]
    label = latex_label(input_observables)
    
    data = pd.concat([sgn_data,bkg_data])
    data = split_data(data, train_frac=0.67)
    
    print("train size:", len(data['train']))
    print("test size:", len(data['test']))
    
    input_data = {}
    input_data['train'] = df_to_ML_input(data['train'])
    input_data['test'] = df_to_ML_input(data['test'])
    
    print("Training Classifier")
    classifier.train(input_data['train'], **classifier_kwargs)
    
    print("Testing Classifier")
    test_results = test_classifier(input_data['test'], classifier)
    train_results = test_classifier(input_data['train'], classifier)

    print("Analysing Classifier")
    ROC_curve(test_results, text=label, train_results=train_results)
    ROC_displacement(test_results, text=label, train_results=train_results)
    plot_results(test_results, classifier, n_bins=50, text=label, train_results=train_results)
    
    p_cut = find_p_cut(test_results)
    
    print("Probability Threshold:", p_cut)
    for k, e in get_scores(test_results, p_cut).items(): print(k,':',e)
    
    rmsq = rms_deviation(test_results, train_results, n_bins=50)
    print(f"rmsq[s]: {rmsq['s']}")
    print(f"rmsq[b]: {rmsq['b']}")
    
    sn_ratio = S_B_ratio(p_cut, test_results)
    print(f'S/N : {sn_ratio}')
    
    return classifier