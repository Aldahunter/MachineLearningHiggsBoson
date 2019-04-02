import collections
import pandas as pd
from random import seed
from timeit import Timer
from humanfriendly import format_timespan

import dwrangling as DW
import dwrangling.dataframes as DWDF
import dwrangling.pipelines as DWP
from trees._BaseClasses import BaseClassifier
import analysis.metrics as AME
import analysis.plotting as AP
import analysis.plotting.clout as APC


# Create Class for Ordering Results #
class OrderedDict(collections.OrderedDict):
    def rename(self, key, new_key):
        if (key not in self) or (new_key in self): return
        edited_dict = [(k, value) if k!=key else (new_key, value)
                        for k, value in self.items()]
        self.clear(); self.update(edited_dict)

        
        
### Functions ###
def analyse_classifier(collision, classifier, fit_kwargs, predict_kwargs,
                     train_frac=0.8, val_frac=0.75, n_bins=100, rseed=None):
    """Trains and tests the classifer with the events data_df \
    panda.DataFrame (containing only the desired observables and signal \
    columns) using the given classifier_kwargs. Returns the classifier and \
    a dictionary or calculated metrics."""
    
    txt = ' Obtaining DataFrame '; print(f"{txt:=^90}", end='\n\n')
    if rseed is not None:
        seed(rseed)
    
    df = DWP.get_opt_dataframe(collision)
    train_df, test_df = df.train_test_sets(train_frac)
    if not isinstance(classifier, BaseClassifier):
        train_df, valid_df = train_df.train_test_sets(val_frac)
        
    
    txt = ' Training Classifier '; print(f"{txt:=^90}", end='\n\n')
    observables, labels = {}, {}
    observables['Train'], labels['Train'] = train_df.ML_input()
    observables['Test'],  labels['Test']  = test_df.ML_input()
    if not isinstance(classifier, BaseClassifier):
        observables['Validate'],  labels['Validate'] = valid_df.ML_input()
    
    print(classifier, '\n')
    def fit(): classifier.fit(observables['Train'], labels['Train'],
                              **fit_kwargs)
    t = Timer(fit, globals=globals()).timeit(1)
    print(classifier, '\n')
    print('Time Taken:', format_timespan(t, detailed=True), '\n')
    
    txt = ' Testing Classifier '; print(f"{txt:=^90}", end='\n\n')
    predictions = OrderedDict()
    predictions['Test'] = classifier.predict_prob(observables['Test'],
                                                  **predict_kwargs)
    predictions['Train'] = classifier.predict_prob(observables['Train'],
                                                   **predict_kwargs)
    if not isinstance(classifier, BaseClassifier):
        predictions['Validate']= classifier.predict(observables['Validate'],
                                                    **predict_kwargs)
    else:
        _predictions = OrderedDict()
        _predictions['Test'] = classifier.predict_from_prob(
                                                       predictions['Test'],
                                                       **predict_kwargs)
        _predictions['Train']= classifier.predict_from_prob(
                                                       predictions['Train'],
                                                       **predict_kwargs)
    
    results =OrderedDict((k,(labels[k],p)) for k,p in  predictions.items())
    _results=OrderedDict((k,(labels[k],p)) for k,p in _predictions.items())
    
    
    txt=' Analysing Classifier Probabilities ';print(f"{txt:=^90}",end='\n')
    if not isinstance(classifier, BaseClassifier):
        p_threshold = AME.find_p_threshold(*_predictions['Validate'])
    else:
        p_threshold = classifier.threshold
    
    scores = AME.get_scores(results, p_threshold=p_threshold, n_bins=n_bins)
    txt = ' Results '; print(f'{txt:-^90}')
    print(f"Probability Threshold: {p_threshold}")
    for k, v in scores.items(): print(k,':',v)

    if isinstance(classifier, BaseClassifier):
        _scores = AME.get_scores(_results, p_threshold, n_bins)
        txt = ' Classified Results '; print(f'{txt:~^90}')
        for k in ['auc', 'rmsd']:
            print(k,':',_scores[k])
            scores[k+'_prob'] = _scores[k]
    
    scores['p_thres'] = p_threshold
    results.update({'_'+k:v for k, v in _results.items()})
    return classifier, results, scores


def plot_classifier(classifier, results, p_thres=0.5, cmap=None, cmid=None,
                    labels="", n_bins=100, snr_resolution=200, leaves=False,
                    hist=False, y_max=None):
    """Analyses the given classifer with the panda.DataFrame events, using \
    the input_observables and classifier_kwargs."""
    
    txt = ' Plotting Results '; print(f"{txt:=^90}")
    _results = OrderedDict()
    for key in list(results.keys()):
        if key.startswith('_'):
            _results[ key[1:] ] = results.pop(key)
    
    try: (y_max[0], y_max[-1])
    except: y_max = (y_max,)
    cmap = cmap if cmap else AP.std_cmap.copy().set_middle(False)
    if not isinstance(cmap, AP.StdCmap): cmap = AP.StdCmap(cmap)
    
    def plot(): APC.plot_confusion_matrix(*results['Test'], p_thres,
                                          cmap=cmap)
    t = Timer(plot).timeit(1)
    txt = ' Confusion Matrix '; print(f'{txt:-^90}')
    print('Time Taken:', format_timespan(t, detailed=True))
    
    
    cmap = cmap.copy().set_middle(True, cmid) if cmid else cmap
    
    def plot(): APC.ROC_curve(results, cmap=cmap, alpha=0.45)
    t = Timer(plot).timeit(1)
    txt = ' ROC Curve '; print(f'{txt:-^90}')
    print('Time Taken:', format_timespan(t, detailed=True))
    if _results:
        def plot(): APC.ROC_curve(_results, cmap=cmap, alpha=0.45)
        t = Timer(plot).timeit(1)
        txt = ' ROC Curve '; print(f'{txt:~^90}')
        print('Time Taken:', format_timespan(t, detailed=True))
    
    
    def plot(): APC.SNRatio(results, leaves_thresholds=leaves, cmap=cmap,
                            n_thresholds=snr_resolution, square='back',
                            p_threshold=p_thres)
    t = Timer(plot).timeit(1)
    txt = ' SNR Curve '; print(f'{txt:-^90}')
    print('Time Taken:', format_timespan(t, detailed=True))
    if _results:
        def plot(): APC.SNRatio(_results, leaves_thresholds=True, cmap=cmap,
                                n_thresholds=2, square='back',
                                p_threshold=p_thres)
        t = Timer(plot).timeit(1)
        txt = ' SNR Curve '; print(f'{txt:~^90}')
        print('Time Taken:', format_timespan(t, detailed=True))
    
    
    for key, value in results.items():
        if key != 'Test': results.rename(key, '['+key+']')
    for key, value in _results.items():
        if key != 'Test': _results.rename(key, '['+key+']')
    
    def plot(): APC.plot_results(classifier,results, p_threshold=p_thres,
                                 hist=hist, n_bins=n_bins, cm_scale=1,
                                 observables=labels, y_max=y_max[0])
    t = Timer(plot).timeit(1)
    txt = ' Histogram Distribution '; print(f'{txt:-^90}')
    print('Time Taken:', format_timespan(t, detailed=True))
    if _results:
        def plot(): APC.plot_results(classifier, _results, hist=True,
                                     p_threshold=p_thres, cm_scale=1,
                                     observables=labels, y_max=y_max[-1])
        t = Timer(plot).timeit(1)
        txt = ' Histogram Distribution '; print(f'{txt:~^90}')
        print('Time Taken:', format_timespan(t, detailed=True))


def analyse(collision, classifier, fit_kwargs={}, predict_kwargs={},
            train_frac=0.8, val_frac=0.75, n_bins=100, cmap=None, cmid=None,
            labels="", leaves=False, hist=False, y_max=None, rseed=None,
            snr_resolution=200):
    
    classifier, results, scores = analyse_classifier(collision, classifier,
                                                     fit_kwargs,
                                                     predict_kwargs,
                                                     train_frac, val_frac,
                                                     n_bins, rseed)
    
    plot_classifier(classifier, results, scores['p_thres'], cmap, cmid,
                    labels, n_bins, snr_resolution, leaves, hist, y_max)
                    
    
    return results, scores


def show_tree_paths(tree, probability, dataframe, **kwargs):
    """Takes a fitted :class:`trees.classifier.BinaryTreeClassifier` and a \
    probability' for a leaf (or leaves), and returns a :class:`list` of \
    :class:`matplotlib.figure` objects, visually showing the observable \
    partitions of the :class:`dwrangling.dataframes.ODataFrame` to reach \
    the probability."""
    
    # Validate dataframe type
    if not isinstance(dataframe, DWDF.ODataFrame):
        raise ValueError("The dataframe given must be of type "
                         + "'dwrangling.dataframe.ODataFrame', "
                         + f"not type '{type(dataframe)}'.")
    # Validate probability given
    if probability not in tree.get_probabilities():
        raise ValueError(f"The probability given '{probability}', is not " + 
                         "valid for a leaf within the tree given.")
    
    # Find paths to the leaf node
    prob_paths = tree.get_path(probability)
    
    # Find lengths of paths associated with the generators
    path_lens = prob_paths.get_paths()
    path_lens = [(len(path) + 2) for path in path_lens]
    
    # Create path sample generators
    generators = prob_paths.get_path_sample_generators(dataframe)
    
    # Iterate through each generater and put output figures into list
    figures = []
    for path_len, generator in zip(path_lens, generators):
        figures.append(APC.parallel_coord_path(generator, path_len,
                                               **kwargs))
    
    # Return figures
    return figures
    