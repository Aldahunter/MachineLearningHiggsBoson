### Functions ###
def hist_normalise(histogram, bin_width):
    total_area = 0.0
    for hist_bin in histogram:
        total_area += hist_bin * bin_width
    return histogram / total_area

def create_normalise_fn(values):
    """Creates a function to normalise values."""
    x_max = values.max()
    x_min = values.min()
    
    def normalise_fn(x):
        return (x - x_min)/(x_max - x_min)
    
    return normalise_fn


        
def print_cm(confusion_matrix):
    """Print the confusion matrix to stdout."""
    tn, fp, fn, tp = confusion_matrix
    
    s_total = float(fn + tp)
    b_total = float(tn + fp)
    a, b, c, d = tn/b_total, fp/b_total, fn/s_total, tp/s_total

    print( "          -------------------------")
    print( "          |         Actual        |")
    print( "          |-----------------------|")
    print( "          |     S     |     B     |")
    print( "----------|-----------|-----------|")
    print( "|   |     |           |           |")
    print( "|   |     |  TP:      |  FP:      |")
    print(f"| P |  S  | {tp:7d}   | {fp:7d}   |")
    print(f"| r |     |   ({d:.2f})  |   ({b:.2f})  |")
    print( "| e |     |           |           |")
    print( "| d |-----|-----------|-----------|")
    print( "| i |     |           |           |")
    print( "| c |     |  FN:      |  TN:      |")
    print(f"| t |  B  | {fn:7d}   | {tn:7d}   |")
    print(f"|   |     |   ({c:.2f})  |   ({a:.2f})  |")
    print( "|   |     |           |           |")
    print( "-----------------------------------")


def _check_results_dims(*results, observables=False):
    """Checks the dimensions of [observables,] labels and labels_predicted)\
    satisfy each other. To check more sets, give each array as a parameter \
    with each set in the correct order."""
    
    if observables is False:
        if not (len(results) % 2) == 0:
            raise ValueError(f"Your must give both the 'labels' and " +
            "'labels_pred' IN THIS ORDER for each pair, or neither. You " +
            f"have given {len(results)//2} sets and 1 unpaired set.")
        
        pair = range(len(results) // 2)
        for n, lbs, lbs_pred in zip(pair, results[::2], results[1::2]):
            if len(lbs) != len(lbs_pred):
                raise ValueError("Your labels and labels_pred should be " +
                f"the same length. In pair {n} you have given lengths " +
                f"{len(lbs)} and {len(lbs_pred)}, respectively.")
    
    else:        
        if not (len(results) % 3) == 0:
            raise ValueError(f"Your must give both the 'observables', " +
            f"'test_labels' and 'test_labels_pred' IN THIS ORDER for " +
            f"each triplet. or neither. You have given {len(results)//3} " +
            f"sets and {len(results) % 3} extra items.")

        triplet = range(len(results) // 3)
        n_observables = results[0].shape[1]
        triplets = lambda sets: (sets[::3], sets[1::3], sets[2::3])
        for n, obs, lbs, lbs_pred in zip(pair, *triplets(results)):
            if not (len(obs) == len(lbs) == len(lbs_pred)):
                raise ValueError("Your 'observables', 'labels' and " +
                                 "'labels_pred' should be the same " +
                                 f"length. In triplet {n} you have given " +
                                 f"lengths {len(obs)}, {len(lbs)} and " +
                                 f"{len(lbs_pred)}, respectively.")
                    
            if obs.shape[1] != n_observables:
                raise ValueError(f"The number of observables in each " +
                "triplet should be equal. You have given " +
                f"{n_observables} observables in set 0, but " +
                f"{obs.shape[1]} observables in set {n}.")

def _check_results_dict(result_dictionary, observables=False):
    """Checks the a dictionary of structure {'key': '([observables,] labels, 
    labels_predicted)', such that 'key' is a 'str', the dictionaries 
    values are 'tuple's of length '2'[or '3'] , and the 'array's within the 
    tuples satisfy each others dimensions."""
    
    # Check its actually a dictionary
    len_tup = 3 if (observables is True) else 2
    if not isinstance(result_dictionary, dict):
        raise ValueError("Parameter 'results_pairs' must be a 'dict' of " +
                         "length 2 'tuple's, where the keys are 'str's " +
                         f"used as legend labels. Not type " +
                         f"'{type(result_dictionary)}'.")
    
    # Check keys are strings
    if sum(map(lambda k: not isinstance(k,str), result_dictionary.keys())):
        raise ValueError("The keys to 'result_pairs' should all be of " +
                         "type 'str'.")
    
    # Check values are tuples of length 2
    if sum(map(lambda v: not (isinstance(v, tuple) and len(v) == len_tup),
               result_dictionary.values())):
        raise ValueError("The elemts in 'result_pairs' should all be " +
                         "'tuple's of length '2'.")
    
    # Check dimensions of tuples' elements
    _check_results_dims(*[item for pair in result_dictionary.values()
                               for item in pair],
                        observables=observables)
                    
                    
def phantom_twin_xaxis(axis):
    """Returns an invisible x-axis twinned to axis."""
    
    # Create twin on y axis
    twin = axis.twiny()
    
    # Remove ticks, which reappear on original, and on new twin
    axis.tick_params(axis='x',  which='both', bottom=False, top=False,
                     labelbottom=False, labeltop=False)
    twin.tick_params(axis='x',  which='both', bottom=False, top=False,
                     labelbottom=False, labeltop=False)
    
    # Remove new twin axis' borders
    for position in ['top', 'bottom', 'left', 'right']:
        twin.spines[position].set_visible(False)
    
    # Return twin axis 
    return twin

                
def mirror_axes_above(axes, labelpad=10):
    """Mirrors the x-axis labels given in axes."""
    
    # Iterate through axes
    for ax in axes:
        
        # Create twin for axis labels
        twin = phantom_twin_xaxis(ax)
        
        # Get x labels' parameters
        bottom_label = ax.xaxis.get_label()
        text = bottom_label.get_text()
        props = bottom_label.get_fontproperties()
        fontsize = bottom_label.get_fontsize()
        pos = bottom_label.get_position()
        
        # Apply orignal parameters to new twinned axis
        twin.set_xlabel(text, fontproperties=props, fontsize=fontsize,
                        position=pos, labelpad=labelpad)

        