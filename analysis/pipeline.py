import pandas as pd
from random import seed

import dwrangling as DWrang
import analysis.metrics as AME
import analysis.plotting as AP

### Functions ###
def SN_classifier(classifier, data_df, classifier_kwargs, train_frac=0.67, rseed=None):
    """Trains and tests the classifer with the events data_df panda.DataFrame \
(containing only the desired observables and signal columns) using the given \
classifier_kwargs."""
    if rseed is not None:
        seed(rseed)
    
    data_df = DWrang.split_data(data_df, train_frac)
    
    observables, labels = {}, {}
    observables['train'], labels['train'] = DWrang.df_to_ML_input(data_df['train'])
    observables['test'],  labels['test']  = DWrang.df_to_ML_input(data_df['test'])

    classifier.train(observables['train'], labels['train'], **classifier_kwargs)
    
    test_predictions = classifier.predict(observables['test'])
    train_predictions = classifier.predict(observables['test'])

    rmsq = AME.rms_deviation(labels['test'], test_predictions,
                             labels['train'], train_predictions,
                             n_bins=50)
    p_threshold = AME.find_p_threshold(labels['test'], test_predictions)
    sn_ratio = AME.S_B_ratio(labels['test'], test_predictions, p_threshold)
    
    return classifier, sn_ratio, rmsq


def analyse_classifier(classifier, sgn_df, bkg_df, input_observables, classifier_kwargs):
    """Analyses the given classifer with the panda.DataFrame events, using the \
input_observables and classifier_kwargs."""
    
    sgn_data = sgn_df[input_observables]
    bkg_data = bkg_df[input_observables]
    label = AP.latex_label(input_observables)
    
    data = pd.concat([sgn_data,bkg_data])
    data = DWrang.split_data(data, train_frac=0.67)
    
    print("train size:", len(data['train']))
    print("test size:", len(data['test']))
    
    observables, labels = {}, {}
    observables['train'], labels['train'] = DWrang.df_to_ML_input(data['train'])
    observables['test'],  labels['test']  = DWrang.df_to_ML_input(data['test'])
    
    print("Training Classifier")
    classifier.train(observables['train'], labels['train'], **classifier_kwargs)
    
    print("Testing Classifier")
    test_predictions = classifier.predict(observables['test'])
    train_predictions = classifier.predict(observables['test'])

    print("Analysing Classifier")
    AP.ROC_curve(labels['test'], test_predictions, labels['train'], train_predictions,
                 text=label)
    AP.ROC_displacement(labels['test'], test_predictions, labels['train'], train_predictions,
                        text=label)
    AP.plot_results(classifier, labels['test'], test_predictions, labels['train'], train_predictions,
                    n_bins=50, text=label)
    
    p_threshold = AME.find_p_threshold(labels['test'], test_predictions)
    
    print("Probability Threshold:", p_threshold)
    for k, e in AME.get_scores(labels['test'], test_predictions, p_threshold).items(): print(k,':',e)
    
    rmsq = AME.rms_deviation(labels['test'], test_predictions,
                             labels['train'], train_predictions,
                             n_bins=50)
    print("rmsq[s]: {0}".format(rmsq['s']));
    print("rmsq[b]: {0}".format(rmsq['b']));
    
    sn_ratio = AME.S_B_ratio(labels['test'], test_predictions, p_threshold)
    print(f'S/N : {sn_ratio}')
    
    return classifier