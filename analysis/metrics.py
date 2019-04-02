import numpy as np
import sklearn.metrics as skmetrics
from scipy.optimize import minimize_scalar

import analysis.misc as AMI

### Functions ###
def sk_confusion_matrix(labels, labels_predicted, p_threshold=0.5):
    bool_predicted = labels_predicted >= p_threshold
    return skmetrics.confusion_matrix(labels, bool_predicted,
                                      labels=[False, True])
    
def confusion_matrix(labels, labels_predicted, p_threshold=0.5):
    return list(sk_confusion_matrix(labels, labels_predicted,
                                    p_threshold=p_threshold).ravel())


def accuracy(tn, fp, fn, tp):
    return (tp + tn) / (tp + fp + fn + tn)


def precision(tn, fp, fn, tp):
    return tp / (tp + fp)


def recall(tn, fp, fn, tp):
    return tp / (tp + fn)


def fpr(tn, fp, fn, tp):
    """False Positive Rate"""
    return fp / (fp + tn)


def f1_score(tn, fp, fn, tp):
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)
    return (2 * p * r) / (p + r)


def auc(labels, labels_pred, **kwargs):
    return skmetrics.roc_auc_score(labels, labels_pred, **kwargs)


def find_p_threshold(labels, labels_predicted):
    sigma_sb_ratio = 59.926097745
    
    # Get a set of all possible probabilities
    p_set = sorted(set(labels_predicted))
    
    # Iterate over every pair of neighbouring probabilities
    max_snr = 0
    for p0, p1 in zip(p_set[:-1], p_set[1:]):
        
        # Calculate the mid-point probability
        p_mid = (p0 + p1) / 2.0
        
        # Obtain Machine Learning results.
        cm = confusion_matrix(labels, labels_predicted, p_threshold=p_mid)
        epsilon_b, epsilon_s = fpr(*cm), recall(*cm)
        
        # Get a scaled value for the signal-to-noise ratio of this mid-point
        snr = epsilon_s / np.sqrt(epsilon_s + (sigma_sb_ratio * epsilon_b))

        # If signal-to-noise is greater than the current max
        if snr > max_snr:
            
            # Make this the new maximum and remember probability
            max_snr = snr
            p_thres = p_mid

    # Return maximum signal-to-noise ratio's probability
    return p_thres

def S_B_ratio(es_labels, eb_labels_pred, p_threshold=0.5):
    # Define physics values.
    sigma_s = 0.204436505 #fb
    sigma_b = 12.25108192 #fb
    luminosity = 100.0 #fb^-1
    
    # Obtain Machine Learning values.
    try:
        cm = confusion_matrix(es_labels, eb_labels_pred, p_threshold)
        es_labels, eb_labels_pred = recall(*cm), fpr(*cm)
    except: pass
    
    eff_sigma_s = sigma_s * es_labels
    eff_sigma_b = sigma_b * eb_labels_pred
    
    # Calculate Signal-Noise numerator.
    numerator = eff_sigma_s * np.sqrt(luminosity)
    if numerator == 0: return 0
    
    # Calculate Signal-Noise ratio.
    sb_ratio = numerator / np.sqrt(eff_sigma_s + eff_sigma_b)
    
    # Return Signal-Noise ratio.
    return sb_ratio


def rms_deviation(labels, labels_pred, *extra_results, n_bins=50):
    AMI._check_results_dims(labels, labels_pred, *extra_results)
    
    results = {0: (labels, labels_pred)}
    results.update({n:(l, l_p)
                    for n, (l, l_p) in enumerate(zip(extra_results[0::2],
                                                 extra_results[1::2]), 1)})
    
    histograms = {}
    for n, (is_signal, signal_prob) in results.items():
        
        _results = {'s': [], 'b': []}
        _results = {'s': signal_prob[is_signal == True],
                    'b': signal_prob[is_signal == False]}

        bins = np.linspace(0, 1, n_bins + 1)
        histogram = {'s': np.histogram(_results['s'], bins)[0],
                     'b': np.histogram(_results['b'], bins)[0]}
        
        width = bins[1] - bins[0]
        histogram['s'] = AMI.hist_normalise(histogram['s'], width)
        histogram['b'] = AMI.hist_normalise(histogram['b'], width)
        
        histograms[n] = histogram
        
    rmsq = {}
    for sb in ['b', 's']:
        sq_deviation = 0
        for n, histogram in histograms.items():
            if n == 0: continue
            sq_deviation += (histogram[sb] - histograms[0][sb])**2
        rmsq[sb] = np.sqrt( sq_deviation.sum() / len(sq_deviation) )
    
    return rmsq


def get_scores(results_dict, p_threshold=0.5, n_bins=50):
    
    # Check results dictionary
    AMI._check_results_dict(results_dict)
    
    # Get keys and inital item
    results = iter(results_dict.values())
    labels, labels_pred = next(results)
    
    scores = {}
    tn, fp, fn, tp = confusion_matrix(labels, labels_pred, p_threshold)

    scores["cm"] = {'tn':tn, 'fp':fp, 'fn':fn, 'tp':tp}
    scores["a"] = accuracy(tn, fp, fn, tp)
    scores["p"] = precision(tn, fp, fn, tp)
    scores["tpr"] = recall(tn, fp, fn, tp)
    scores["fpr"] = fpr(tn, fp, fn, tp)
    scores["f1"] = f1_score(tn, fp, fn, tp)
    scores["auc"] = auc(labels, labels_pred)
    scores["s/b"] = S_B_ratio(scores["tpr"], scores["fpr"], p_threshold)
    if len(results_dict) > 1:
        scores["rmsd"] = rms_deviation(*sum(results_dict.values(), ()),
                                       n_bins=n_bins)
    
    return scores